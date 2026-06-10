import base64
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI

from settings import LLMConfig
from modules.interaction_detector import InteractionResult
from modules.tracker import MovementState, TrackedBBox
from storage.database import LogEntry
import cv2
from utils.image import draw_normalized_bbox

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an object behavior observation specialist. "
    "You track the movement of objects in a single camera feed.\n\n"
    "Output ONLY valid JSON with these fields:\n"
    '- "description": One concise, objective sentence describing the movement and behavior of the target object. '
    "Use natural expressions like 'moving left/right/up/down', 'rotating', "
    "'moving quickly', 'moving slowly', 'stopped' — never use pixel values or numerical measurements. "
    "If nearby objects are listed in the input, you MUST include them in your description. "
    "Do not omit any nearby objects that are provided.\n"
    '- "reasoning": Same as description (single camera).\n\n'
    "RULES:\n"
    "- Never invent objects or relationships that are not in the input.\n"
    "- No emotions, no speculation.\n"
    "- Output ONLY valid JSON. No markdown, no code fences.\n"
)

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
class _LogTask:
    tracked_list: List[TrackedBBox]
    camera_id: str
    interaction_results: List[InteractionResult] | None
    image_b64: Optional[str] = None
    space_logger: Optional["SpaceLogger"] = None
    space_id: Optional[str] = None
    target_coordinate: Optional[List[float]] = None
    target_classes: Optional[List[str]] = None


@dataclass
class _VisionLogTask:
    images: List[str]
    camera_id: str
    llm_system_prompt: str | None
    target_classes: List[str] | None


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


class NLPLogger:
    def __init__(self, config: LLMConfig, log_dir: str = "logs", output_dir: str = "output", repo=None):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self._repo = repo
        self.debouncer = LLMCallDebouncer(config.cooldown_seconds)
        self.client: Optional[OpenAI] = None
        self._queue = queue.Queue()
        self._vision_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="nlp-worker")
        self._worker_thread.start()
        self._vision_worker_thread = threading.Thread(target=self._vision_worker, daemon=True, name="vision-nlp-worker")
        self._vision_worker_thread.start()

    def _ensure_client(self):
        if self.client is None and self.config.api_key:
            self.client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
            )

    def _db_insert(self, log_type: str, timestamp: str | None = None, camera_id: str | None = None,
                   space_id: str | None = None, target_present: bool | None = None,
                   description: str | None = None, target_coordinate: list | None = None,
                   reasoning: str | None = None, raw_json: str | None = None):
        if self._repo is None:
            return
        import json as _json
        try:
            ts = datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)
        entry = LogEntry(
            timestamp=ts,
            log_type=log_type,
            camera_id=camera_id,
            space_id=space_id,
            target_present=target_present,
            description=description,
            target_coordinate=_json.dumps(target_coordinate) if target_coordinate is not None else None,
            reasoning=reasoning,
            raw_json=raw_json,
            created_at=datetime.now(timezone.utc),
        )
        self._repo.save(entry)

    def _save_image(self, image_b64: str, camera_id: str):
        try:
            cam_dir = self.output_dir / camera_id
            cam_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filepath = cam_dir / f"{timestamp}.jpg"
            image_bytes = base64.b64decode(image_b64)
            filepath.write_bytes(image_bytes)
        except Exception as e:
            logger.error("Image save failed for %s: %s", camera_id, e)

    def log(
        self, tracked_list: List[TrackedBBox], frame: np.ndarray, camera_id: str = "cam_01",
        interaction_results: List[InteractionResult] | None = None,
        space_logger: Optional["SpaceLogger"] = None, space_id: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not tracked_list:
            return None
        h, w = frame.shape[:2]
        t0 = tracked_list[0]
        target_coordinate = [t0.x1 / w, t0.y1 / h, t0.x2 / w, t0.y2 / h]

        self._ensure_client()
        debounce_key = f"{camera_id}_batch"
        if not self.debouncer.should_call(debounce_key):
            logger.debug("[logger] debounce suppress: %s", debounce_key)
            return None

        if self.client is None:
            return self._log_fallback(tracked_list, camera_id, interaction_results, space_logger, space_id, target_coordinate)

        image_b64 = None
        if self.config.vision_enabled:
            vis = frame.copy()
            if self.config.vision_max_width > 0 and vis.shape[1] > self.config.vision_max_width:
                scale = self.config.vision_max_width / vis.shape[1]
                vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.config.vision_quality])
            raw_b64 = base64.b64encode(buf).decode("utf-8")
            if target_coordinate:
                image_b64 = draw_normalized_bbox(raw_b64, target_coordinate, label=t0.class_name)
            else:
                image_b64 = raw_b64
            self._save_image(image_b64, camera_id)
        self._queue.put(_LogTask(
            tracked_list, camera_id, interaction_results, image_b64,
            space_logger, space_id, target_coordinate, target_classes,
        ))
        logger.debug("[logger] enqueue: %s (qsize=%d, vision=%s)", debounce_key, self._queue.qsize(), "on" if image_b64 else "off")
        return None

    def _log_fallback(self, tracked_list: List[TrackedBBox], camera_id: str,
                       interaction_results: List[InteractionResult] | None,
                       space_logger: Optional["SpaceLogger"], space_id: Optional[str],
                       target_coordinate: List[float]) -> str:
        import json as _json
        timestamp = datetime.now(timezone.utc).isoformat()
        desc_parts = []
        for t in tracked_list:
            movement = _state_to_movement(t.state, _angle_to_direction(t.speed, t.direction_angle)) if t.state is not None else "detected"
            desc_parts.append(f"{t.class_name} {movement}")
        description = " | ".join(desc_parts)
        log_entry = {
            "target_present": True,
            "cameras": {
                camera_id: {
                    "description": description,
                    "target_coordinate": target_coordinate,
                }
            },
            "reasoning": description,
        }
        text = _json.dumps(log_entry)
        self._save_log(text, timestamp, camera_id)
        if space_logger and space_id:
            space_logger.collect(space_id, camera_id, text)
            space_logger.try_flush(space_id, space_id)
        logger.info("[%s] target_present=true desc=%s bbox=%s", camera_id, description, target_coordinate)
        return text

    def vision_log(self, images: List[str], camera_id: str, llm_system_prompt: str | None, target_classes: List[str] | None):
        self._ensure_client()
        if self.client is None:
            return None
        debounce_key = f"{camera_id}_vision"
        if not self.debouncer.should_call(debounce_key):
            logger.debug("[logger] vision debounce suppress: %s", debounce_key)
            return None
        self._vision_queue.put(_VisionLogTask(images, camera_id, llm_system_prompt, target_classes))
        logger.debug("[logger] vision enqueue: %s (qsize=%d)", camera_id, self._vision_queue.qsize())

    def _process_vision_task(self, task: _VisionLogTask):
        timestamp = datetime.now(timezone.utc).isoformat()
        context_parts = [f"Timestamp: {timestamp}", f"Camera: {task.camera_id}"]
        if task.target_classes:
            context_parts.append(f"Target objects: {', '.join(task.target_classes)}")
        context_parts.append("")
        context_parts.append("Analyze the images in chronological order (earliest first) and describe:")
        context_parts.append("1. The behavior of each target object across the frames")
        context_parts.append("2. Any interactions between objects")
        context_parts.append("3. Notable changes in the scene")
        context_prompt = "\n".join(context_parts)

        parts = [SYSTEM_PROMPT]
        if task.llm_system_prompt:
            parts.append(f"Additional instructions:\n{task.llm_system_prompt}")
        system_content = "\n".join(parts)
        user_messages = []
        for img_b64 in task.images:
            user_messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
        user_messages.insert(0, {"type": "text", "text": context_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_messages},
                ],
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Vision LLM API call failed: %s", e)
            return None

        self._db_insert(
            log_type="vision",
            timestamp=timestamp,
            camera_id=task.camera_id,
            description=text,
            raw_json=text,
        )
        logger.info("[vision:%s] %s", task.camera_id, text)
        return text

    def _process_vision_batch_space(self, images: List[tuple[str, str]], space_name: str,
                                     llm_system_prompt: str | None, target_classes: List[str] | None,
                                     valid_camera_ids: List[str] | None = None,
                                     camera_health: Dict[str, str] | None = None):
        self._ensure_client()
        if self.client is None:
            return None
        timestamp = datetime.now(timezone.utc).isoformat()
        capture_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S")

        user_messages = [{"type": "text", "text": f"Timestamp: {timestamp}\nSpace: {space_name}"}]

        current_cam = None
        seen_cameras: set[str] = set()
        for cam_id, img_b64 in images:
            seen_cameras.add(cam_id)
            if cam_id != current_cam:
                user_messages.append({"type": "text", "text": f"\n--- [{cam_id}] ---"})
                current_cam = cam_id
            user_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        if camera_health:
            for cam_id in sorted(camera_health):
                if cam_id in seen_cameras:
                    continue
                health = camera_health[cam_id]
                if health == "degraded":
                    user_messages.append({"type": "text", "text": f"\n--- [{cam_id}] ---\n[stale image — no fresh frame available]"})
                    seen_cameras.add(cam_id)
                elif health == "dead":
                    user_messages.append({"type": "text", "text": f"\n--- [{cam_id}] ---\n[camera offline]"})
                    seen_cameras.add(cam_id)

        if target_classes:
            unique_classes = list(dict.fromkeys(target_classes))
            user_messages.append({"type": "text", "text": f"\nTarget objects: {', '.join(unique_classes)}"})

        parts = [VISION_SPACE_SYSTEM_PROMPT]
        if llm_system_prompt:
            parts.append(f"Additional instructions:\n{llm_system_prompt}")

        messages = [
            {"role": "system", "content": "\n".join(parts)},
            {"role": "user", "content": user_messages},
        ]

        response_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": 500,
        }

        text = None
        try:
            kw_with_format = dict(response_kwargs)
            kw_with_format["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kw_with_format)
            text = response.choices[0].message.content.strip()
        except Exception as e:
            if "'response_format'" in str(e):
                logger.warning("Model does not support json_object, falling back to text-only")
                try:
                    response = self.client.chat.completions.create(**response_kwargs)
                    text = response.choices[0].message.content.strip()
                except Exception as e2:
                    logger.error("Space vision LLM API call failed (fallback): %s", e2)
                    return None
            else:
                logger.error("Space vision LLM API call failed: %s", e)
                return None

        parsed = self._parse_json_response(text, space_name)
        if not parsed:
            logger.warning("[vision:%s] parse failed, raw: %s", space_name, text[:400])
            return None

        target_present = parsed.get("target_present", False)
        cameras = parsed.get("cameras", {})
        reasoning = parsed.get("reasoning", "")
        if not reasoning:
            logger.warning("[vision:%s] reasoning field missing in LLM response", space_name)

        logger.debug("[vision:%s] parsed: target=%r cameras_keys=%r", space_name, target_present, list(cameras.keys()))

        bbox_coords: dict[str, list[float]] = {}
        for cam_id, cam_val in cameras.items():
            clean_cam_id = cam_id.strip("[]")
            if valid_camera_ids and clean_cam_id not in valid_camera_ids:
                logger.warning("[vision:%s] unknown camera_id %r (not in %r), skipping", space_name, cam_id, valid_camera_ids)
                continue
            if isinstance(cam_val, str):
                desc = cam_val.strip()
                self._save_log(desc, timestamp, clean_cam_id)
            elif isinstance(cam_val, dict):
                desc = cam_val.get("description", "")
                if desc:
                    self._save_log(desc.strip(), timestamp, clean_cam_id)
                coords = cam_val.get("target_coordinate")
                if coords and isinstance(coords, list) and len(coords) == 4:
                    bbox_coords[clean_cam_id] = [float(c) for c in coords]
            else:
                logger.warning("[vision:%s] camera %s value has unexpected type (%s), skipping", space_name, cam_id, type(cam_val).__name__)
                continue

        if target_present:
            last_per_cam: dict[str, str] = {}
            for cam_id, img_b64 in images:
                last_per_cam[cam_id] = img_b64
            for cam_id, img_b64 in last_per_cam.items():
                filename = f"{space_name}_{cam_id}_{capture_ts}.jpg"
                try:
                    img_to_save = img_b64
                    if cam_id in bbox_coords:
                        img_to_save = draw_normalized_bbox(img_b64, bbox_coords[cam_id], label=space_name)
                    (self.output_dir / filename).write_bytes(base64.b64decode(img_to_save))
                except Exception as e:
                    logger.error("Failed to save target capture %s: %s", filename, e)

            self._db_insert(
                log_type="vision_space",
                timestamp=timestamp,
                space_id=space_name,
                target_present=target_present,
                reasoning=reasoning,
                raw_json=text,
            )

        logger.info("[vision:%s] target_present=%s cameras=%d %s", space_name, target_present, len(cameras), reasoning)
        return text

    @staticmethod
    def _parse_json_response(text: str, context: str = "") -> Optional[Dict]:
        import re
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
        """Synchronous single-image detection call. No debounce (caller manages timing)."""
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

    def stop(self):
        self._stop_event.set()
        self._worker_thread.join(timeout=5)
        self._vision_worker_thread.join(timeout=5)

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_task(task)

    def _vision_worker(self):
        while not self._stop_event.is_set():
            try:
                task = self._vision_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_vision_task(task)

    def _process_task(self, task: _LogTask):
        changes = self._build_state_changes(task.tracked_list)
        if not changes:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        prompt = self._build_prompt(changes, timestamp, task.camera_id, task.interaction_results,
                                     task.target_coordinate, task.target_classes)
        logger.debug("[logger] LLM call: camera=%s prompt=%d chars", task.camera_id, len(prompt))

        import json as _json
        text = None
        try:
            if task.image_b64:
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{task.image_b64}"}},
                ]
            else:
                content = prompt
            response_kwargs = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 200,
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
            logger.error("LLM API call failed: %s", e)
            return None

        parsed = self._parse_json_response(text, task.camera_id)
        if parsed:
            llm_desc = parsed.get("description", "") or parsed.get("reasoning", "") or text
        else:
            llm_desc = text

        cam_data: dict = {"description": llm_desc}
        if task.target_coordinate:
            cam_data["target_coordinate"] = task.target_coordinate
        log_entry = {
            "target_present": True,
            "cameras": {task.camera_id: cam_data},
            "reasoning": llm_desc,
        }
        log_text = _json.dumps(log_entry)

        self._save_log(log_text, timestamp, task.camera_id)
        if task.space_logger and task.space_id:
            task.space_logger.collect(task.space_id, task.camera_id, log_text)
            task.space_logger.try_flush(task.space_id, task.space_id)

        logger.info("[%s] target_present=true desc=%s bbox=%s", task.camera_id, llm_desc, task.target_coordinate)
        return log_text

    def _build_state_changes(self, tracked_list: List[TrackedBBox]) -> List[Dict]:
        changes = []
        for t in tracked_list:
            direction_angle = getattr(t, "direction_angle", 0.0)
            change = {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "current_state": t.state,
                "prev_state": None,
                "speed": t.speed,
                "direction": _angle_to_direction(t.speed, direction_angle),
            }
            if t.prev_bbox is not None:
                change["prev_state"] = None
                change["is_new"] = False
            else:
                change["is_new"] = True
            changes.append(change)
        return changes

    def _build_prompt(self, changes: List[Dict], timestamp: str, camera_id: str,
                       interaction_results: List[InteractionResult] | None = None,
                       target_coordinate: Optional[List[float]] = None,
                       target_classes: Optional[List[str]] = None) -> str:
        lines = [f"Timestamp: {timestamp}", f"Camera: {camera_id}", ""]
        lines.append("Tracked objects:")

        for c in changes:
            state_str = c["current_state"].name if c["current_state"] else "UNKNOWN"
            direction = c.get("direction", "unknown")
            if c.get("is_new"):
                lines.append(
                    f"- {c['class_name']}: APPEARED"
                )
            else:
                movement = _state_to_movement(c["current_state"], direction)
                lines.append(
                    f"- {c['class_name']}: {movement}"
                )

        if interaction_results:
            lines.append("")
            lines.append("Nearby objects (include these in your description):")
            for ir in interaction_results:
                rel = {"interacting": "touching", "contact": "touching", "nearby": "near"}.get(ir.relation_type, ir.relation_type)
                lines.append(f"- {ir.class_name}: {rel}")

        if target_coordinate:
            lines.append("")
            lines.append(f"Detected bounding box (normalized): {target_coordinate}")
        if target_classes:
            lines.append(f"Target classes: {', '.join(target_classes)}")

        lines.append("")
        if interaction_results:
            lines.append("Describe the state changes AND the object's relationship with nearby objects.")
        else:
            lines.append("Describe the current state changes.")
        lines.append('Output ONLY valid JSON: {"description": "one sentence", "reasoning": "same one sentence"}')
        return "\n".join(lines)

    def _save_log(self, text: str, timestamp: str, camera_id: str, log_type: str = "detect"):
        import json as _json
        target_present = None
        description = None
        target_coordinate = None
        reasoning = None
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, dict):
                target_present = parsed.get("target_present")
                reasoning = parsed.get("reasoning")
                cameras = parsed.get("cameras", {})
                if isinstance(cameras, dict):
                    first_cam = next(iter(cameras.values()), {})
                    if isinstance(first_cam, dict):
                        description = first_cam.get("description")
                        target_coordinate = first_cam.get("target_coordinate")
        except (_json.JSONDecodeError, TypeError):
            description = text
        self._db_insert(
            log_type=log_type,
            timestamp=timestamp,
            camera_id=camera_id,
            target_present=target_present,
            description=description,
            target_coordinate=target_coordinate,
            reasoning=reasoning,
            raw_json=text,
        )


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
    if name == "DASHING":
        return f"moving quickly {direction}" if direction not in ("stationary", "unknown") else "moving quickly"
    if name == "FAST_MOVE":
        return f"moving quickly {direction}" if direction not in ("stationary", "unknown") else "moving quickly"
    if name == "SLOW_MOVE":
        return f"moving slowly {direction}" if direction not in ("stationary", "unknown") else "moving slowly"
    return "UNKNOWN"



SPACE_SYSTEM_PROMPT = (
    "You are an object behavior observation specialist. "
    "Given observations from multiple cameras in the same space, "
    "synthesize them into one concise, objective sentence describing the overall situation. "
    "No emotions, no speculation.\n\n"
    'Output ONLY valid JSON: {"reasoning": "one sentence"}\n'
    "No markdown, no code fences.\n"
)

DETECT_SYSTEM_PROMPT = (
    "You are a target detection specialist. Your task is to determine whether a specific "
    "target object is present in a single image.\n\n"
    "RULES:\n"
    "1) Look for the target objects listed in 'Target objects:'. If none listed, detect any living creature.\n"
    "2) If the target is visible (even partially, e.g. a tail, paw, or ear peeking from behind furniture), "
    "set target_present to true.\n"
    "3) Do NOT confuse inanimate objects (toys, cushions, shadows) with the living target.\n"
    "4) When in doubt, default to false. Never guess.\n\n"
    'OUTPUT: Respond with ONLY valid JSON: {"target_present": true/false, "reasoning": "one sentence summary"}\n'
    "No markdown, no code fences, no trailing commas.\n"
)

VISION_SPACE_SYSTEM_PROMPT = (
    "You are an object behavior observation specialist analyzing a space from multiple camera angles. "
    "Each group of images is labeled with its camera ID in brackets, e.g. '[livingroom]'. "

    "\n\nRULES:\n"
    "1) CAMERA IDs: In your JSON output, use ONLY the bare camera IDs — 'livingroom', NOT '[livingroom]', "
    "NOT 'livingroom_2'. Never invent camera IDs that don't appear in the input.\n"
    "2) TARGET-ONLY DESCRIPTIONS: Describe what the target is DOING — its action, pose, location, and any object it interacts with. "
    "If a target is not visible from a camera, write 'No target detected.' as the description.\n"
    "3) ONE SENTENCE PER CAMERA: Each description must be one short sentence (under 20 words).\n"
     "4) BOUNDING BOX (REQUIRED when target visible): When the target IS visible from a camera, "
     "include its tight bounding box as normalized coordinates [x1, y1, x2, y2] "
     "where (0,0) is top-left and (1,1) is bottom-right. "
     "The box MUST tightly enclose the visible body — no padding around it. "
     "Aim for 2-3 decimal places (e.g. [0.25, 0.35, 0.55, 0.65]). "
     "If target is not visible from a camera, set target_coordinate to null.\n"

    "\n\nDETECTION RULES:\n"
    "- Set target_present to true if you see the target (even partially) — look for body parts peeking from furniture/beds, "
    "movement, or interactions with objects.\n"
    "- Do NOT confuse inanimate objects (toys, cushions, shadows) with the living target. "
    "But do not dismiss a real partially-hidden animal as an inanimate object.\n"
    "- When in doubt, default to false. Never guess.\n"

    "\n\nOUTPUT FORMAT (REQUIRED FIELDS):\n"
    "Your response must be ONLY valid JSON. No markdown, no code fences, no trailing commas, "
    "no additional text before or after the JSON object.\n\n"
    "REQUIRED fields in your JSON:\n"
    '- "target_present": true or false (REQUIRED — you MUST include this)\n'
    '- "cameras": object with camera IDs as keys (REQUIRED — one entry per camera)\n'
    '- "reasoning": string (REQUIRED — you MUST include this field. '
    'One short sentence summarizing your conclusion.)\n\n'

    "CAMERAS FORMAT:\n"
    "Each camera value MUST be an object (NOT a string) with these fields:\n"
    '- "description": string (REQUIRED — one short sentence)\n'
    '- "target_coordinate": [x1, y1, x2, y2] or null (REQUIRED — '
    'normalized 0.0-1.0 bounding box when target visible, null otherwise)\n\n'

    "Example when target IS present:\n"
    '{\n'
    '  "target_present": true,\n'
    '  "cameras": {\n'
    '    "hallway": {\n'
    '      "description": "No target detected.",\n'
    '      "target_coordinate": null\n'
    '    },\n'
    '    "livingfront": {\n'
    '      "description": "Cat sitting on the sofa, facing the camera, grooming its paw.",\n'
    '      "target_coordinate": [0.25, 0.35, 0.55, 0.65]\n'
    '    }\n'
    '  },\n'
    '  "reasoning": "Cat sitting on the sofa, grooming its left front paw with its tongue."\n'
    "}\n"

    "Example when target IS NOT present:\n"
    '{\n'
    '  "target_present": false,\n'
    '  "cameras": {\n'
    '    "hallway": {\n'
    '      "description": "No target detected.",\n'
    '      "target_coordinate": null\n'
    '    },\n'
    '    "livingfront": {\n'
    '      "description": "No target detected.",\n'
    '      "target_coordinate": null\n'
    '    }\n'
    '  },\n'
    '  "reasoning": "No living cat visible from any camera; only inanimate objects present."\n'
    "}\n"
)


class SpaceLogger:
    def __init__(self, config: LLMConfig, log_dir: str = "logs", flush_threshold: int = 0, repo=None):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._repo = repo
        self.debouncer = LLMCallDebouncer(cooldown_seconds=5.0)
        self.client: Optional[OpenAI] = None
        self._buffer: Dict[str, Dict[str, List[str]]] = {}
        self._flush_threshold = flush_threshold
        self._camera_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._vision_buffer: Dict[str, Dict[str, dict]] = {}  # space_id -> camera_id -> {"images": [...], "submitted": bool}

    def _ensure_client(self):
        if self.client is None and self.config.api_key:
            self.client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
            )

    def _db_insert(self, log_type: str, timestamp: str | None = None, camera_id: str | None = None,
                   space_id: str | None = None, target_present: bool | None = None,
                   description: str | None = None, target_coordinate: list | None = None,
                   reasoning: str | None = None, raw_json: str | None = None):
        if self._repo is None:
            return
        import json as _json
        try:
            ts = datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)
        entry = LogEntry(
            timestamp=ts,
            log_type=log_type,
            camera_id=camera_id,
            space_id=space_id,
            target_present=target_present,
            description=description,
            target_coordinate=_json.dumps(target_coordinate) if target_coordinate is not None else None,
            reasoning=reasoning,
            raw_json=raw_json,
            created_at=datetime.now(timezone.utc),
        )
        self._repo.save(entry)

    def set_camera_count(self, space_id: str, count: int):
        self._camera_counts[space_id] = count

    def cleanup_space(self, space_id: str):
        with self._lock:
            self._buffer.pop(space_id, None)
            self._vision_buffer.pop(space_id, None)
            self._camera_counts.pop(space_id, None)

    def collect(self, space_id: str, camera_id: str, text: str):
        logger.debug("[space:%s] collect from %s", space_id, camera_id)
        with self._lock:
            if space_id not in self._buffer:
                self._buffer[space_id] = {}
            if camera_id not in self._buffer[space_id]:
                self._buffer[space_id][camera_id] = []
            self._buffer[space_id][camera_id].append(text)

    def vision_collect(self, space_id: str, camera_id: str, images: List[str]):
        """버퍼에 최신 이미지만 저장 (LLM 호출 없음 — flush_vision이 주기적으로 처리)"""
        logger.debug("[space:%s][vision] collect from %s (%d images)", space_id, camera_id, len(images))
        with self._lock:
            if space_id not in self._vision_buffer:
                 self._vision_buffer[space_id] = {}
            self._vision_buffer[space_id][camera_id] = {"images": images, "submitted": False}

    def flush_vision(self, space_id: str, space_name: str, nlp_logger: "NLPLogger",
                        llm_system_prompt: str | None = None, target_classes: List[str] | None = None,
                        valid_camera_ids: List[str] | None = None,
                        camera_health: Dict[str, str] | None = None,
                        override_images: List[tuple[str, str, float]] | None = None):
        """Submit all camera images for space-level LLM analysis.

        Two call paths:
        - Periodic timer (legacy): reads from self._vision_buffer
        - Scheduler (new): caller provides override_images + camera_health directly

        Each image tuple: (camera_id, image_b64, captured_wall_clock)
        """
        logger.debug("[flush_vision] called space_id=%s", space_id)
        if override_images is not None:
            all_images = override_images
            if not all_images and not camera_health:
                return None
        else:
            with self._lock:
                entries = self._vision_buffer.get(space_id, {})
                if not entries:
                    return None
                all_images = []
                now = time.time()
                for cam_id in sorted(entries.keys()):
                    entry = entries[cam_id]
                    for img in entry["images"]:
                        all_images.append((cam_id, img, now))
                    entry["submitted"] = True

        if not nlp_logger.debouncer.should_call(f"{space_id}_vision"):
            logger.debug("[space:%s][vision] debounce suppress", space_id)
            return None

        return nlp_logger._process_vision_batch_space(
            images=all_images, space_name=space_name,
            llm_system_prompt=llm_system_prompt, target_classes=target_classes,
            valid_camera_ids=valid_camera_ids,
            camera_health=camera_health,
        )

    def flush(self, space_id: str, space_name: str) -> Optional[str]:
        with self._lock:
            entries = self._buffer.pop(space_id, {})
        if not entries:
            return None
        timestamp = datetime.now(timezone.utc).isoformat()

        import json as _json
        cameras: Dict[str, dict] = {}
        all_descs: list[str] = []
        for cam_id, texts in sorted(entries.items()):
            for t in texts:
                try:
                    parsed = _json.loads(t)
                    cam_entry = parsed.get("cameras", {}).get(cam_id, {})
                    desc = cam_entry.get("description", t)
                    coord = cam_entry.get("target_coordinate")
                except (_json.JSONDecodeError, TypeError):
                    desc = t
                    coord = None
                if cam_id not in cameras:
                    cam_data: dict = {"description": desc}
                    if coord:
                        cam_data["target_coordinate"] = coord
                    cameras[cam_id] = cam_data
                all_descs.append(f"{cam_id}: {desc}")

        self._ensure_client()
        if self.client is not None and self.debouncer.should_call(space_id):
            prompt_lines = [f"Timestamp: {timestamp}", f"Space: {space_name}", ""]
            for cam_id, cam_data in sorted(cameras.items()):
                prompt_lines.append(f"- {cam_id}: {cam_data['description']}")
            prompt_lines.append("")
            prompt = "\n".join(prompt_lines)
            try:
                use_format = {"type": "json_object"}
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": SPACE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=150,
                    response_format=use_format,
                )
                text = response.choices[0].message.content.strip()
                parsed = _json.loads(text)
                reasoning = parsed.get("reasoning", "")
            except Exception as e:
                if "'response_format'" in str(e):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.config.model_name,
                            messages=[
                                {"role": "system", "content": SPACE_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=150,
                        )
                        reasoning = response.choices[0].message.content.strip()
                    except Exception as e2:
                        logger.error("Space LLM fallback failed: %s", e2)
                        reasoning = " | ".join(all_descs)
                else:
                    logger.error("Space LLM API call failed: %s", e)
                    reasoning = " | ".join(all_descs)
        else:
            reasoning = " | ".join(all_descs)

        log_entry = {
            "target_present": True,
            "cameras": cameras,
            "reasoning": reasoning,
        }
        log_text = _json.dumps(log_entry)

        self._db_insert(
            log_type="space",
            timestamp=timestamp,
            space_id=space_id,
            target_present=log_entry.get("target_present"),
            reasoning=reasoning,
            raw_json=log_text,
        )
        logger.info("[%s] target_present=true cameras=%d reasoning=%s", space_id, len(cameras), reasoning)
        return log_text

    def try_flush(self, space_id: str, space_name: str) -> Optional[str]:
        if self._flush_threshold <= 0:
            return None
        with self._lock:
            entries = self._buffer.get(space_id, {})
            if len(entries) < self._flush_threshold:
                logger.debug("[space:%s] try_flush skipped: %d < %d", space_id, len(entries), self._flush_threshold)
                return None
        return self.flush(space_id, space_name)
