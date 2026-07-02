import base64
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from openai import OpenAI

from settings import LLMConfig
from storage.database import LogEntry
from utils.image import draw_normalized_bbox
from nlp.prompts import DETECT_SYSTEM_PROMPT, SNAPSHOT_VISION_PROMPT, SNAPSHOT_TRACKING_PROMPT

logger = logging.getLogger(__name__)

_LANG_NAMES = {"ko": "Korean (한국어)", "ja": "Japanese (日本語)", "en": "English"}

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






class SpaceLogger:
    def __init__(self, config: LLMConfig, log_dir: str = "logs", repo=None, event_bus=None, minio_config=None, log_config=None):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._repo = repo
        self._event_bus = event_bus
        self.debouncer = LLMCallDebouncer(cooldown_seconds=5.0)
        self.client: Optional[OpenAI] = None
        self._lock = threading.Lock()
        self._minio = None
        self._minio_config = minio_config
        self._log_config = log_config
        if minio_config and minio_config.is_configured:
            from minio import Minio
            self._minio = Minio(
                minio_config.endpoint,
                access_key=minio_config.access_key,
                secret_key=minio_config.secret_key,
                secure=True,
            )
        self._cleanup_stop = threading.Event()
        self._last_cleanup_time: float = 0.0
        self._cleanup_thread = None
        self._start_cleanup_scheduler()

    def _start_cleanup_scheduler(self):
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        self._cleanup_stop.clear()
        t = threading.Thread(target=self._run_cleanup_scheduler, daemon=True, name="snapshot-cleanup")
        t.start()
        self._cleanup_thread = t

    def _run_cleanup_scheduler(self):
        while not self._cleanup_stop.is_set():
            now = datetime.now(timezone.utc)
            seconds_to_next_hour = (59 - now.minute) * 60 + (59 - now.second) - now.microsecond / 1e6
            remaining = seconds_to_next_hour
            while remaining > 0 and not self._cleanup_stop.is_set():
                wait = min(remaining, 300)
                self._cleanup_stop.wait(wait)
                remaining -= wait
            if self._cleanup_stop.is_set():
                break
            now_ts = time.time()
            if now_ts - self._last_cleanup_time < 60:
                continue
            self._last_cleanup_time = now_ts
            logger.info("[cleanup] Running hourly cleanup at %s", datetime.now(timezone.utc).isoformat())
            if self._minio:
                self._do_cleanup_minio()
            else:
                self._do_cleanup_local()

    def _do_cleanup_minio(self):
        if not self._minio or not self._minio_config:
            return
        if not self._minio_config.cleanup_enabled:
            logger.debug("[cleanup] MinIO cleanup disabled")
            return
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._minio_config.retention_hours)
        deleted = 0
        try:
            for obj in self._minio.list_objects(self._minio_config.bucket):
                if obj.last_modified < cutoff:
                    self._minio.remove_object(self._minio_config.bucket, obj.object_name)
                    logger.debug("[cleanup] Deleted MinIO object: %s (modified: %s)", obj.object_name, obj.last_modified)
                    deleted += 1
        except Exception as e:
            logger.error("[cleanup] MinIO cleanup failed: %s", e)
        logger.info("[cleanup] MinIO: deleted %d objects older than %dh", deleted, self._minio_config.retention_hours)

    def _do_cleanup_local(self):
        if not self._log_config:
            return
        if not self._log_config.cleanup_enabled:
            logger.debug("[cleanup] local cleanup disabled")
            return
        output_dir = Path("output")
        if not output_dir.exists():
            return
        cutoff_ts = time.time() - self._log_config.retention_hours * 3600
        deleted = 0
        try:
            for f in output_dir.iterdir():
                if f.is_file() and f.suffix == ".jpg":
                    if f.stat().st_mtime < cutoff_ts:
                        f.unlink()
                        logger.debug("[cleanup] Deleted local file: %s", f.name)
                        deleted += 1
        except Exception as e:
            logger.error("[cleanup] Local cleanup failed: %s", e)
        logger.info("[cleanup] output/: deleted %d files older than %dh", deleted, self._log_config.retention_hours)

    def stop(self):
        """Stop the cleanup scheduler thread."""
        logger.info("[cleanup] Stopping cleanup scheduler")
        self._cleanup_stop.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=305)

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
                   raw_json: str | None = None, image_path: str | None = None,
                   visual_evidence: list | None = None):
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
            image_path=image_path,
            visual_evidence=_json.dumps(visual_evidence) if visual_evidence is not None else None,
            created_at=datetime.now(timezone.utc),
        )
        self._repo.save(entry)
        if self._event_bus:
            import json as _json_ev
            ts_str = timestamp or datetime.now(timezone.utc).isoformat()
            data: dict = {
                "id": entry.id,
                "timestamp": ts_str,
                "log_type": log_type,
                "subject_id": subject_id,
                "target_present": target_present,
                "description": description,
            }
            if image_path:
                data["image_url"] = f"/api/logs/{entry.id}/image"
            self._event_bus.publish({
                "type": "log",
                "data": data,
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
        start = text.find('{')
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        extracted = text[start:i+1]
                        for cleaned in [extracted, re.sub(r',\s*}', '}', extracted), re.sub(r',\s*\]', ']', extracted)]:
                            try:
                                parsed = json.loads(cleaned)
                                if isinstance(parsed, dict):
                                    logger.warning("[parse_json] Recovered via brace balance. context=%s", context or "unknown")
                                    return parsed
                            except (json.JSONDecodeError, TypeError):
                                continue
                        break
        logger.warning("[parse_json] Failed to parse LLM output as JSON. context=%s, text=%r", context or "unknown", text[:500])
        return None

    def vision_detect(self, camera_id: str, image_b64: str, llm_system_prompt: str | None,
                      target_classes: list[str] | None) -> dict | None:
        self._ensure_client()
        if self.client is None:
            return None

        target_label = target_classes[0] if target_classes else "target"
        language_name = _LANG_NAMES.get(self.config.log_language, self.config.log_language)
        combined_system = DETECT_SYSTEM_PROMPT.format(
            target_label=target_label,
            language_name=language_name,
        )
        if llm_system_prompt:
            combined_system += "\n\n" + llm_system_prompt
        messages = [{"role": "system", "content": combined_system}]
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Look at this image carefully.\n\nReturn JSON only.\n\n{{\n  \"target_present\": true/false,\n  \"visual_evidence\": [\n    \"...\"\n  ]\n}}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        })
        try:
            kwargs = dict(
                model=self.config.model_name,
                messages=messages,
            )
            if self.config.json_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            r = self.client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content.strip() if r.choices[0].message.content else ""
        except Exception as e:
            logger.error("Vision detect failed for %s: %s", camera_id, e)
            return None

        logger.debug("[detect:%s] raw (%d chars): %s", camera_id, len(text), text[:500])
        parsed = self._parse_json_response(text, f"detect_{camera_id}")
        if not parsed:
            logger.debug("[detect:%s] parse failed, raw: %s", camera_id, text[:200])
            return None

        target_present = parsed.get("target_present", False)
        evidence_list = parsed.get("visual_evidence", [])
        if isinstance(evidence_list, list):
            reasoning_str = "\n".join(str(e) for e in evidence_list)
        else:
            reasoning_str = str(evidence_list)
        parsed["reasoning"] = reasoning_str
        if target_present:
            logger.info("[detect:%s] target_present=True reason=%s", camera_id, reasoning_str)
        else:
            logger.info("[detect:%s] target_present=False reason=%s", camera_id, reasoning_str)
        return parsed

    def space_snapshot(self, space_id: str, space_name: str,
                       snapshots: Dict[str, CameraSnapshot],
                       vision_enabled: bool,
                       target_classes: List[str] | None = None,
                       llm_system_prompt: str | None = None,
                       detect_context=None) -> Optional[str]:
        timestamp = datetime.now(timezone.utc).isoformat()
        batch_id = uuid4().hex
        self._ensure_client()

        if self.client is None:
            return self._snapshot_fallback(space_id, snapshots, timestamp, space_name, batch_id)

        user_messages = []
        if vision_enabled:
            user_messages.append({"type": "text", "text": f"Timestamp: {timestamp}\nSpace: {space_name}"})
            if detect_context:
                context_str = (f"\nDETECTION CONTEXT: Camera '{detect_context['camera']}' detected the target. "
                               f"Reasoning: {detect_context['reasoning']}. Examine that camera's sequence first.")
                user_messages.append({"type": "text", "text": context_str})
            for cam_id in sorted(snapshots.keys()):
                snap = snapshots[cam_id]
                user_messages.append({"type": "text", "text": f"\n--- [{cam_id}] ---"})
                if snap.images:
                    for img_b64 in snap.images:
                        user_messages.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                elif snap.image_b64:
                    user_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{snap.image_b64}"}
                    })

            if target_classes:
                unique_classes = list(dict.fromkeys(target_classes))
                user_messages.append({"type": "text", "text": f"\nTarget objects: {', '.join(unique_classes)}"})
            parts = [SNAPSHOT_VISION_PROMPT]
            if llm_system_prompt:
                parts.append(f"Additional instructions:\n{llm_system_prompt}")
            lang_name = _LANG_NAMES.get(self.config.log_language, self.config.log_language)
            parts.append(f"IMPORTANT: All description and reasoning fields MUST be written in {lang_name}. JSON keys must remain in English.")
            system_prompt = "\n".join(parts)
            user_messages.append({"type": "text", "text": f"Respond in {lang_name}."})
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
                tracking_str = "target detected" if snap.target_present else "no target"
                coord_str = f" | bbox {snap.target_coordinate}" if snap.target_coordinate else ""
                prompt_lines.append(f"- {cam_id}: {tracking_str}{coord_str}")
            lang_name = _LANG_NAMES.get(self.config.log_language, self.config.log_language)
            system_prompt = SNAPSHOT_TRACKING_PROMPT + f"\n\nIMPORTANT: All description and reasoning fields MUST be written in {lang_name}. JSON keys must remain in English."
            user_messages = [{"type": "text", "text": "\n".join(prompt_lines) + f"\n\nRespond in {lang_name}."}]

        if not self.debouncer.should_call(f"{space_id}_snapshot"):
            logger.debug("[space:%s] snapshot debounce suppress", space_id)
            return None

        text = None
        try:
            _msgs = []
            if system_prompt:
                _msgs.append({"role": "system", "content": system_prompt})
            _msgs.append({"role": "user", "content": user_messages})
            response_kwargs = {
                "model": self.config.model_name,
                "messages": _msgs,
            }
            if self.config.json_response_format:
                response_kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**response_kwargs)
            text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Snapshot LLM API call failed for %s: %s", space_id, e)
            return self._snapshot_fallback(space_id, snapshots, timestamp, space_name, batch_id, target_classes)

        parsed = self._parse_json_response(text, f"snapshot_{space_id}")
        if not parsed:
            logger.warning("[snapshot:%s] parse failed, raw: %s", space_id, text[:400])
            return self._snapshot_fallback(space_id, snapshots, timestamp, space_name, batch_id, target_classes)

        cameras_resp = parsed.get("cameras", {})
        reasoning = parsed.get("reasoning", "")
        if not reasoning and isinstance(cameras_resp, dict):
            for v in cameras_resp.values():
                if isinstance(v, str):
                    reasoning = v
                    break

        top_level_present = parsed.get("target_present", False)
        if not top_level_present:
            logger.info("[snapshot:%s] top-level false, suppressed", space_id)
            return None

        if cameras_resp:
            has_any_true = any(
                isinstance(v, dict) and v.get("target_present", False)
                for v in cameras_resp.values()
            )
            if not has_any_true:
                logger.info("[snapshot:%s] all per-camera false, overridden to false", space_id)
                return None

        per_camera_present: List[bool] = []
        for cam_id, snap in snapshots.items():
            class_name: str = ""
            visual_evidence: list | None = None
            if cam_id in cameras_resp:
                cam_resp = cameras_resp[cam_id]
                if isinstance(cam_resp, str):
                    desc = cam_resp
                    coord = None
                    merged_present = False
                elif isinstance(cam_resp, dict):
                    desc = cam_resp.get("description", "") or cam_resp.get("reasoning", "")
                    class_name = cam_resp.get("class_name", "") or ""
                    visual_evidence = cam_resp.get("visual_evidence", None)
                    raw_coord = cam_resp.get("target_coordinate")
                    if raw_coord and len(raw_coord) == 4:
                        coord = [raw_coord[1] / 1000.0, raw_coord[0] / 1000.0, raw_coord[3] / 1000.0, raw_coord[2] / 1000.0]
                    else:
                        coord = snap.target_coordinate
                    merged_present = cam_resp.get("target_present", False)
                else:
                    desc = f"target={snap.target_present}"
                    coord = snap.target_coordinate
                    merged_present = False
            else:
                desc = None
                coord = snap.target_coordinate
                merged_present = False

            per_camera_present.append(merged_present)

            img_path = None
            if (snap.image_b64 or snap.images) and snap.target_present:
                img_to_save = snap.images[-1] if snap.images else snap.image_b64
                img_path = self._save_snapshot_image(img_to_save, space_name, cam_id, timestamp, coord, class_name)

            self._db_insert(
                log_type="detect",
                timestamp=timestamp,
                batch_id=batch_id,
                subject_id=cam_id,
                target_present=merged_present,
                description=desc,
                target_coordinate=coord,
                image_path=img_path,
                visual_evidence=visual_evidence,
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

        logger.debug("[snapshot:%s] parsed per-camera: %s", space_id, json.dumps(cameras_resp))
        logger.info("[snapshot:%s] cameras=%d reasoning=%s", space_id, len(snapshots), reasoning)
        return log_text


    def _save_snapshot_image(self, image_b64: str, space_name: str,
                              cam_id: str, timestamp: str, coord: Optional[List[float]] = None,
                              target_label: str = "") -> Optional[str]:
        try:
            capture_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S")
            filename = f"{space_name}_{cam_id}_{capture_ts}.jpg"
            img_data = image_b64
            if coord:
                img_data = draw_normalized_bbox(image_b64, coord, label=target_label or space_name)
            img_bytes = base64.b64decode(img_data)

            if self._minio:
                from io import BytesIO
                self._minio.put_object(
                    self._minio_config.bucket, filename, BytesIO(img_bytes), len(img_bytes), "image/jpeg"
                )
            else:
                output_dir = Path("output")
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / filename).write_bytes(img_bytes)
            return filename
        except Exception as e:
            logger.error("Failed to save snapshot image %s: %s", filename, e)
            return None

    def _snapshot_fallback(self, space_id: str, snapshots: Dict[str, CameraSnapshot],
                            timestamp: str, space_name: str = "",
                            batch_id: str = "",
                            target_classes: List[str] | None = None) -> str:
        if not batch_id:
            batch_id = uuid4().hex
        reasoning_parts = []
        for cam_id, snap in snapshots.items():
            desc = "no target detected"
            coord = None
            fallback_label = target_classes[0] if target_classes else ""
            if snap.target_present:
                desc = "target detected"
                coord = snap.target_coordinate
            reasoning_parts.append(f"{cam_id}: {desc}")

            img_path = None
            if (snap.image_b64 or snap.images) and snap.target_present:
                img_to_save = snap.images[-1] if snap.images else snap.image_b64
                img_path = self._save_snapshot_image(img_to_save, space_name or space_id, cam_id, timestamp, coord, fallback_label)

            self._db_insert(
                log_type="detect",
                timestamp=timestamp,
                batch_id=batch_id,
                subject_id=cam_id,
                target_present=snap.target_present,
                description=desc,
                target_coordinate=coord,
                image_path=img_path,
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
