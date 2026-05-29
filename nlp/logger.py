import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from config.config import LLMConfig
from modules.interaction_detector import InteractionResult
from modules.tracker import MovementState, TrackedBBox

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
    space_logger: Optional['SpaceLogger'] = None
    space_id: Optional[str] = None


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
    def __init__(self, config: LLMConfig, log_dir: str = "logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debouncer = LLMCallDebouncer(config.cooldown_seconds)
        self.client: Optional[OpenAI] = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="nlp-worker")
        self._worker_thread.start()

    def _ensure_client(self):
        if self.client is None and self.config.api_key:
            self.client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
            )

    def log(self, tracked_list: List[TrackedBBox], camera_id: str = "cam_01", interaction_results: List[InteractionResult] | None = None, space_logger: Optional['SpaceLogger'] = None, space_id: Optional[str] = None) -> Optional[str]:
        if not tracked_list:
            return None
        self._ensure_client()
        if self.client is None:
            return None
        debounce_key = f"{camera_id}_batch"
        if not self.debouncer.should_call(debounce_key):
            return None
        self._queue.put(_LogTask(tracked_list, camera_id, interaction_results, space_logger, space_id))
        return None

    def stop(self):
        self._stop_event.set()
        self._worker_thread.join(timeout=5)

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_task(task)

    def _process_task(self, task: _LogTask):
        changes = self._build_state_changes(task.tracked_list)
        if not changes:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        prompt = self._build_prompt(changes, timestamp, task.camera_id, task.interaction_results)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.3,
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

    def log_appearance(self, tracked: TrackedBBox, camera_id: str = "cam_01", interaction_results: List[InteractionResult] | None = None) -> Optional[str]:
        self._ensure_client()
        if self.client is None:
            return None
        debounce_key = f"{camera_id}_{tracked.track_id}"
        if not self.debouncer.should_call(debounce_key):
            return None
        self._queue.put(_LogTask([tracked], camera_id, interaction_results))
        return None

    def log_disappearance(self, track_id: int, class_name: str, camera_id: str = "cam_01") -> Optional[str]:
        self._ensure_client()
        if self.client is None:
            return None
        fake_tracked = TrackedBBox(
            track_id=track_id, frame_id=0, x1=0, y1=0, x2=0, y2=0,
            confidence=0.0, class_id=0, class_name=class_name,
        )
        self._queue.put(_LogTask([fake_tracked], camera_id, None))
        return None


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


def _format_direction(direction: str) -> str:
    """Simplify direction for natural language."""
    if direction in ("stationary", "unknown"):
        return ""
    return direction


SPACE_SYSTEM_PROMPT = (
    "You are an object behavior observation specialist. "
    "Given observations from multiple cameras in the same space, "
    "synthesize them into one concise, objective sentence describing the overall situation. "
    "No emotions, no speculation. Output exactly ONE sentence."
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

    def _ensure_client(self):
        if self.client is None and self.config.api_key:
            self.client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
            )

    def set_camera_count(self, space_id: str, count: int):
        self._camera_counts[space_id] = count

    def collect(self, space_id: str, camera_id: str, text: str):
        with self._lock:
            if space_id not in self._buffer:
                self._buffer[space_id] = {}
            if camera_id not in self._buffer[space_id]:
                self._buffer[space_id][camera_id] = []
            self._buffer[space_id][camera_id].append(text)

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
                temperature=0.3,
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
                return None
        return self.flush(space_id, space_name)
