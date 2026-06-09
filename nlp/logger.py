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

from config.config import LLMConfig
from modules.interaction_detector import InteractionResult
from modules.tracker import MovementState, TrackedBBox
from utils.image import annotate_image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an object behavior observation specialist. "
    "Describe the movement of tracked objects in one concise, objective sentence. "
    "Use natural expressions like 'moving left/right/up/down', 'rotating', "
    "'moving quickly', 'moving slowly', 'stopped' — never use pixel values or numerical measurements. "
    "If nearby objects are listed in the input, you MUST include them in your description. "
    "Do not omit any nearby objects that are provided. "
    "Never invent objects or relationships that are not in the input. "
    "No emotions, no speculation. Output exactly ONE sentence."
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
    def __init__(self, config: LLMConfig, log_dir: str = "logs", output_dir: str = "output"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
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
    ) -> Optional[str]:
        if not tracked_list:
            return None
        self._ensure_client()
        if self.client is None:
            return None
        debounce_key = f"{camera_id}_batch"
        if not self.debouncer.should_call(debounce_key):
            logger.debug("[logger] debounce suppress: %s", debounce_key)
            return None
        image_b64 = None
        if self.config.vision_enabled:
            image_b64 = annotate_image(frame, tracked_list, quality=self.config.vision_quality, max_width=self.config.vision_max_width)
            self._save_image(image_b64, camera_id)
        self._queue.put(_LogTask(tracked_list, camera_id, interaction_results, image_b64, space_logger, space_id))
        logger.debug("[logger] enqueue: %s (qsize=%d, vision=%s)", debounce_key, self._queue.qsize(), "on" if image_b64 else "off")
        return None

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

        log_file = self.log_dir / f"vision_{task.camera_id}_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")
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

        for cam_id, cam_val in cameras.items():
            clean_cam_id = cam_id.strip("[]")
            if valid_camera_ids and clean_cam_id not in valid_camera_ids:
                logger.warning("[vision:%s] unknown camera_id %r (not in %r), skipping", space_name, cam_id, valid_camera_ids)
                continue
            if isinstance(cam_val, str):
                self._save_log(cam_val.strip(), timestamp, f"vision_{clean_cam_id}")
            else:
                logger.warning("[vision:%s] camera %s value is not a string (%r), skipping", space_name, cam_id, cam_val)
                continue

        if target_present:
            last_per_cam: dict[str, str] = {}
            for cam_id, img_b64 in images:
                last_per_cam[cam_id] = img_b64
            for cam_id, img_b64 in last_per_cam.items():
                filename = f"{space_name}_{cam_id}_{capture_ts}.jpg"
                try:
                    (self.output_dir / filename).write_bytes(base64.b64decode(img_b64))
                except Exception as e:
                    logger.error("Failed to save target capture %s: %s", filename, e)

            log_file = self.log_dir / f"vision_space_{space_name}_{datetime.now().strftime('%Y%m%d')}.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {reasoning}\n")

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
        logger.info("[detect:%s] target_present=%s reason=%s", camera_id, target_present, reasoning)
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
        prompt = self._build_prompt(changes, timestamp, task.camera_id, task.interaction_results)
        logger.debug("[logger] LLM call: camera=%s prompt=%d chars", task.camera_id, len(prompt))
        try:
            if task.image_b64:
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{task.image_b64}"}},
                ]
            else:
                content = prompt
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=150,
            )
            text = response.choices[0].message.content.strip()
            self._save_log(text, timestamp, task.camera_id)
            if task.space_logger and task.space_id:
                task.space_logger.collect(task.space_id, task.camera_id, text)
                task.space_logger.try_flush(task.space_id, task.space_id)
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
        return text

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

    def _build_prompt(self, changes: List[Dict], timestamp: str, camera_id: str, interaction_results: List[InteractionResult] | None = None) -> str:
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

        lines.append("")
        if interaction_results:
            lines.append("Describe the state changes AND the object's relationship with nearby objects in one sentence.")
        else:
            lines.append("Describe the current state changes in one sentence.")
        return "\n".join(lines)

    def _save_log(self, text: str, timestamp: str, camera_id: str):
        log_file = self.log_dir / f"{camera_id}_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")


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
    "No emotions, no speculation. Output exactly ONE sentence."
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
    "If a target is not visible from a camera, write EXACTLY: 'No target detected.'\n"
    "3) ONE SENTENCE PER CAMERA: Each description must be one short sentence (under 20 words).\n"

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

    "Example when target IS present:\n"
    '{\n'
    '  "target_present": true,\n'
    '  "cameras": {\n'
    '    "hallway": "No target detected.",\n'
    '    "livingfront": "Cat sitting on the sofa, facing the camera, grooming its paw."\n'
    '  },\n'
    '  "reasoning": "Cat sitting on the sofa, grooming its left front paw with its tongue."\n'
    "}\n"

    "Example when target IS NOT present:\n"
    '{\n'
    '  "target_present": false,\n'
    '  "cameras": {\n'
    '    "hallway": "No target detected.",\n'
    '    "livingfront": "No target detected."\n'
    '  },\n'
    '  "reasoning": "No living cat visible from any camera; only inanimate objects present."\n'
    "}\n"
)


class SpaceLogger:
    def __init__(self, config: LLMConfig, log_dir: str = "logs", flush_threshold: int = 0):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
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

    def set_camera_count(self, space_id: str, count: int):
        self._camera_counts[space_id] = count

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
                        override_images: List[tuple[str, str]] | None = None):
        """Submit all camera images for space-level LLM analysis.

        Two call paths:
        - Periodic timer (legacy): reads from self._vision_buffer
        - Scheduler (new): caller provides override_images + camera_health directly
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
                for cam_id in sorted(entries.keys()):
                    entry = entries[cam_id]
                    for img in entry["images"]:
                        all_images.append((cam_id, img))
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
        self._ensure_client()
        if self.client is None:
            return None
        if not self.debouncer.should_call(space_id):
            return None
        timestamp = datetime.now(timezone.utc).isoformat()
        prompt_lines = [f"Timestamp: {timestamp}", f"Space: {space_name}", ""]
        for cam_id, texts in sorted(entries.items()):
            for t in texts:
                prompt_lines.append(f"- {cam_id}: {t}")
        prompt_lines.append("")
        prompt_lines.append("Synthesize these camera observations into one sentence.")
        prompt = "\n".join(prompt_lines)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SPACE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Space LLM API call failed: %s", e)
            return None
        log_file = self.log_dir / f"{space_id}_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")
        return text

    def try_flush(self, space_id: str, space_name: str) -> Optional[str]:
        if self._flush_threshold <= 0:
            return None
        with self._lock:
            entries = self._buffer.get(space_id, {})
            if len(entries) < self._flush_threshold:
                logger.debug("[space:%s] try_flush skipped: %d < %d", space_id, len(entries), self._flush_threshold)
                return None
        return self.flush(space_id, space_name)
