import base64
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from openai import OpenAI

from settings import LLMConfig
from modules.interaction_detector import InteractionResult
from modules.tracker import TrackedBBox
from storage.database import LogEntry
from utils.image import draw_normalized_bbox
from nlp.prompts import DETECT_SYSTEM_PROMPT, SNAPSHOT_VISION_PROMPT, SNAPSHOT_TRACKING_PROMPT

logger = logging.getLogger(__name__)

DIRECTION_MAP = {
    (337.5, 360): "up",
    (0, 22.5): "up",
    (22.5, 67.5): "up and right",
    (67.5, 112.5): "right",
    (112.5, 157.5): "down and right",
    (157.5, 202.5): "down",
    (202.5, 247.5): "down and left",
    (247.5, 292.5): "left",
    (292.5, 337.5): "up and left",
}


@dataclass
class CameraSnapshot:
    camera_id: str
    target_present: bool
    timestamp: float
    tracked_list: List[TrackedBBox] = field(default_factory=list)
    interactions: List[InteractionResult] = field(default_factory=list)
    image_b64: Optional[str] = None
    images: List[str] = field(default_factory=list)
    target_coordinate: Optional[List[float]] = None


class LLMCallDebouncer:
    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown = cooldown_seconds
        self._last_call: Dict[str, float] = {}

    def should_call(self, key: str) -> bool:
        now = time.time()
        last = self._last_call.get(key, 0.0)
        if now - last < self.cooldown:
            return False
        self._last_call[key] = now
        return True


def _angle_to_direction(speed: float, angle: float) -> str:
    if speed < 5:
        return "stationary"
    for (lo, hi), direction in DIRECTION_MAP.items():
        if lo <= angle < hi:
            return direction
    return "unknown"


def _state_to_movement(state, direction: str) -> str:
    if state is None:
        return "UNKNOWN"
    name = state.name
    if name == "STOPPED":
        return "stopped"
    if name == "ROTATING":
        return "rotating"
    if name in ("DASHING", "FAST_MOVE"):
        return f"moving quickly {direction}" if direction not in ("stationary", "unknown") else "moving quickly"
    if name == "SLOW_MOVE":
        return f"moving slowly {direction}" if direction not in ("stationary", "unknown") else "moving slowly"
    return "UNKNOWN"


class SpaceLogger:
    def __init__(self, config: LLMConfig, log_dir: str = "logs", repo=None, event_bus=None):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._repo = repo
        self._event_bus = event_bus
        self.debouncer = LLMCallDebouncer(cooldown_seconds=5.0)
        self.client: Optional[OpenAI] = None
        self._lock = threading.Lock()

    def _ensure_client(self):
        if self.client is None:
            if self.config.api_key:
                self.client = OpenAI(
                    base_url=self.config.api_base_url,
                    api_key=self.config.api_key,
                )
                logger.debug("[space_logger] OpenAI client created (base=%s)", self.config.api_base_url)
            else:
                logger.debug("[space_logger] api_key empty, client remains None")

    def _db_insert(self, log_type: str, timestamp: str | None = None,
                   batch_id: str = "", subject_id: str | None = None,
                   target_present: bool | None = None,
                   description: str | None = None, target_coordinate: list | None = None,
                   raw_json: str | None = None):
        if self._repo is None:
            return
        import json as _json
        try:
            ts = datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)
        entry = LogEntry(
            batch_id=batch_id,
            timestamp=ts,
            log_type=log_type,
            subject_id=subject_id,
            target_present=target_present,
            description=description,
            target_coordinate=_json.dumps(target_coordinate) if target_coordinate is not None else None,
            raw_json=raw_json,
            created_at=datetime.now(timezone.utc),
        )
        self._repo.save(entry)
        if self._event_bus:
            import json as _json_ev
            ts_str = timestamp or datetime.now(timezone.utc).isoformat()
            self._event_bus.publish({
                "type": "log",
                "data": {
                    "id": entry.id,
                    "timestamp": ts_str,
                    "log_type": log_type,
                    "subject_id": subject_id,
                    "target_present": target_present,
                    "description": description,
                },
            })

    @staticmethod
    def _parse_json_response(text: str, context: str = "") -> Optional[Dict]:
        candidates = [text]
        stripped = text.strip()
        if stripped != text:
            candidates.append(stripped)
        for fence in ["```json", "```JSON", "```"]:
            if fence in text:
                parts = text.split(fence, 1)
                if len(parts) > 1:
                    after_fence = parts[1].rsplit("```", 1)[0].strip()
                    candidates.append(after_fence)
        for t in candidates:
            try:
                parsed = json.loads(t)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                continue
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            extracted = m.group(0)
            for cleaned in [extracted, re.sub(r',\s*}', '}', extracted), re.sub(r',\s*\]', ']', extracted)]:
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        logger.warning("[parse_json] Recovered via regex extraction. context=%s", context or "unknown")
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    continue
        logger.warning("[parse_json] Failed to parse LLM output as JSON. context=%s, text=%r", context or "unknown", text[:500])
        return None

    def vision_detect(self, camera_id: str, image_b64: str, llm_system_prompt: str | None,
                      target_classes: list[str] | None) -> dict | None:
        self._ensure_client()
        if self.client is None:
            return None

        timestamp = datetime.now(timezone.utc).isoformat()
        context = [f"Timestamp: {timestamp}", f"Camera: {camera_id}"]
        if target_classes:
            context.append(f"Target objects: {', '.join(target_classes)}")
        context.append("\nDetermine if any target object is present in this single image.")
        context_prompt = "\n".join(context)

        user_messages = [
            {"type": "text", "text": context_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]

        parts = [DETECT_SYSTEM_PROMPT]
        if llm_system_prompt:
            parts.append(f"Additional instructions:\n{llm_system_prompt}")

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "\n".join(parts)},
                    {"role": "user", "content": user_messages},
                ],
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Vision detect LLM call failed for %s: %s", camera_id, e)
            return None

        parsed = self._parse_json_response(text, f"detect_{camera_id}")
        if not parsed:
            logger.debug("[detect:%s] parse failed, raw: %s", camera_id, text[:200])
            return None

        target_present = parsed.get("target_present", False)
        reasoning = parsed.get("reasoning", "")
        if target_present:
            logger.info("[detect:%s] target_present=True reason=%s", camera_id, reasoning)
        else:
            logger.debug("[detect:%s] target_present=False reason=%s", camera_id, reasoning)
        return parsed

    def space_snapshot(self, space_id: str, space_name: str,
                       snapshots: Dict[str, CameraSnapshot],
                       vision_enabled: bool,
                       target_classes: List[str] | None = None,
                       llm_system_prompt: str | None = None) -> Optional[str]:
        timestamp = datetime.now(timezone.utc).isoformat()
        batch_id = uuid4().hex
        self._ensure_client()

        if self.client is None:
            return self._snapshot_fallback(space_id, snapshots, timestamp, space_name, batch_id)

        user_messages = []
        if vision_enabled:
            user_messages.append({"type": "text", "text": f"Timestamp: {timestamp}\nSpace: {space_name}"})
            for cam_id in sorted(snapshots.keys()):
                snap = snapshots[cam_id]
                user_messages.append({"type": "text", "text": f"\n--- [{cam_id}] ---"})
                if snap.images:
                    for img_b64 in snap.images:
                        user_messages.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        })
                elif snap.image_b64:
                    user_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{snap.image_b64}"}
                    })
                tracking_str = self._build_tracking_summary(snap.tracked_list, snap.interactions)
                if tracking_str:
                    coord_str = f" | bbox {snap.target_coordinate}" if snap.target_coordinate else ""
                    user_messages.append({"type": "text", "text": tracking_str + coord_str})
            if target_classes:
                unique_classes = list(dict.fromkeys(target_classes))
                user_messages.append({"type": "text", "text": f"\nTarget objects: {', '.join(unique_classes)}"})
            parts = [SNAPSHOT_VISION_PROMPT]
            if llm_system_prompt:
                parts.append(f"Additional instructions:\n{llm_system_prompt}")
            system_prompt = "\n".join(parts)
        else:
            prompt_lines = [f"Timestamp: {timestamp}", f"Space: {space_name}", ""]
            if target_classes:
                unique_classes = list(dict.fromkeys(target_classes))
                prompt_lines.append(f"Target objects: {', '.join(unique_classes)}")
                prompt_lines.append("")
            for cam_id in sorted(snapshots.keys()):
                snap = snapshots[cam_id]
                if not snap.target_present:
                    prompt_lines.append(f"- {cam_id}: no target detected")
                    continue
                tracking_str = self._build_tracking_summary(snap.tracked_list, snap.interactions)
                coord_str = f" | bbox {snap.target_coordinate}" if snap.target_coordinate else ""
                prompt_lines.append(f"- {cam_id}: {tracking_str}{coord_str}")
            system_prompt = SNAPSHOT_TRACKING_PROMPT
            user_messages = [{"type": "text", "text": "\n".join(prompt_lines)}]

        if not self.debouncer.should_call(f"{space_id}_snapshot"):
            logger.debug("[space:%s] snapshot debounce suppress", space_id)
            return None

        text = None
        try:
            response_kwargs = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_messages},
                ],
                "max_tokens": 500,
            }
            try:
                kw_with_format = dict(response_kwargs)
                kw_with_format["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kw_with_format)
                text = response.choices[0].message.content.strip()
            except Exception as e:
                if "'response_format'" in str(e):
                    logger.warning("Model does not support json_object, falling back to text-only")
                    response = self.client.chat.completions.create(**response_kwargs)
                    text = response.choices[0].message.content.strip()
                else:
                    raise
        except Exception as e:
            logger.error("Snapshot LLM API call failed for %s: %s", space_id, e)
            return self._snapshot_fallback(space_id, snapshots, timestamp, space_name, batch_id)

        parsed = self._parse_json_response(text, f"snapshot_{space_id}")
        if not parsed:
            logger.warning("[snapshot:%s] parse failed, raw: %s", space_id, text[:400])
            return self._snapshot_fallback(space_id, snapshots, timestamp, space_name, batch_id)

        cameras_resp = parsed.get("cameras", {})
        reasoning = parsed.get("reasoning", "")

        per_camera_present: List[bool] = []
        for cam_id, snap in snapshots.items():
            cam_resp = cameras_resp.get(cam_id, {})
            if isinstance(cam_resp, str):
                desc = cam_resp
                coord = None
                llm_present = False
            elif isinstance(cam_resp, dict):
                desc = cam_resp.get("description", "") or cam_resp.get("reasoning", "")
                coord = cam_resp.get("target_coordinate") or snap.target_coordinate
                llm_present = cam_resp.get("target_present", False)
            else:
                desc = f"target={snap.target_present}"
                coord = snap.target_coordinate
                llm_present = False

            merged_present = snap.target_present or llm_present
            per_camera_present.append(merged_present)

            if snap.image_b64 or snap.images:
                img_to_save = snap.image_b64 or snap.images[0]
                self._save_snapshot_image(img_to_save, space_name, cam_id, timestamp, coord)

            self._db_insert(
                log_type="detect",
                timestamp=timestamp,
                batch_id=batch_id,
                subject_id=cam_id,
                target_present=merged_present,
                description=desc,
                target_coordinate=coord,
            )

        target_present_all = any(per_camera_present)
        log_entry = {
            "target_present": target_present_all,
            "cameras": {cam_id: {"description": desc, "target_coordinate": snap.target_coordinate}
                        for cam_id, snap in snapshots.items()},
            "reasoning": reasoning,
        }
        log_text = json.dumps(log_entry)
        self._db_insert(
            log_type="space",
            timestamp=timestamp,
            batch_id=batch_id,
            subject_id=space_id,
            target_present=target_present_all,
            description=reasoning,
            raw_json=log_text,
        )

        logger.info("[snapshot:%s] cameras=%d reasoning=%s", space_id, len(snapshots), reasoning)
        return log_text

    def _build_tracking_summary(self, tracked_list: List[TrackedBBox],
                                interactions: List[InteractionResult]) -> str:
        parts = []
        for t in tracked_list:
            movement = _state_to_movement(t.state, _angle_to_direction(t.speed, t.direction_angle))
            parts.append(f"{t.class_name}: {movement}")
        if interactions:
            for ir in interactions:
                rel = {"interacting": "touching", "contact": "touching", "nearby": "near"}.get(ir.relation_type, ir.relation_type)
                parts.append(f"nearby: {ir.class_name} ({rel})")
        return " | ".join(parts)

    def _save_snapshot_image(self, image_b64: str, space_name: str,
                             cam_id: str, timestamp: str, coord: Optional[List[float]] = None):
        try:
            capture_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S")
            filename = f"{space_name}_{cam_id}_{capture_ts}.jpg"
            img_data = image_b64
            if coord:
                img_data = draw_normalized_bbox(image_b64, coord, label=space_name)
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / filename).write_bytes(base64.b64decode(img_data))
        except Exception as e:
            logger.error("Failed to save snapshot image %s: %s", filename, e)

    def _snapshot_fallback(self, space_id: str, snapshots: Dict[str, CameraSnapshot],
                           timestamp: str, space_name: str = "",
                           batch_id: str = "") -> str:
        if not batch_id:
            batch_id = uuid4().hex
        reasoning_parts = []
        for cam_id, snap in snapshots.items():
            desc = "no target detected"
            coord = None
            if snap.target_present and snap.tracked_list:
                desc = self._build_tracking_summary(snap.tracked_list, snap.interactions)
                coord = snap.target_coordinate
            reasoning_parts.append(f"{cam_id}: {desc}")

            if snap.image_b64 or snap.images:
                img_to_save = snap.image_b64 or snap.images[0]
                self._save_snapshot_image(img_to_save, space_name or space_id, cam_id, timestamp, coord)

            self._db_insert(
                log_type="detect",
                timestamp=timestamp,
                batch_id=batch_id,
                subject_id=cam_id,
                target_present=snap.target_present,
                description=desc,
                target_coordinate=coord,
            )

        target_present_all = any(s.target_present for s in snapshots.values())
        reasoning = " | ".join(reasoning_parts)
        log_entry = {
            "target_present": target_present_all,
            "cameras": {cam_id: {"description": desc} for cam_id, desc in
                       zip(snapshots.keys(), [r.split(": ", 1)[1] for r in reasoning_parts])},
            "reasoning": reasoning,
        }
        log_text = json.dumps(log_entry)
        self._db_insert(
            log_type="space",
            timestamp=timestamp,
            batch_id=batch_id,
            subject_id=space_id,
            target_present=target_present_all,
            description=reasoning,
            raw_json=log_text,
        )
        return log_text
