import logging
import time
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
    "Describe the movement and interactions of tracked objects in one concise, objective sentence. "
    "No emotions, no speculation. Output exactly ONE sentence."
)

DIRECTION_MAP = {
    (337.5, 360): "north",
    (337.5, 22.5): "north",
    (22.5, 67.5): "northeast",
    (67.5, 112.5): "east",
    (112.5, 157.5): "southeast",
    (157.5, 202.5): "south",
    (202.5, 247.5): "southwest",
    (247.5, 292.5): "west",
    (292.5, 337.5): "northwest",
}


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

    def _ensure_client(self):
        if self.client is None and self.config.api_key:
            self.client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
            )

    def log(self, tracked_list: List[TrackedBBox], camera_id: str = "cam_01", interaction_results: List[InteractionResult] | None = None) -> Optional[str]:
        if not tracked_list:
            return None

        self._ensure_client()
        if self.client is None:
            return None

        timestamp = datetime.now(timezone.utc).isoformat()
        changes = self._build_state_changes(tracked_list)

        if not changes:
            return None

        debounce_key = f"{camera_id}_batch"
        if not self.debouncer.should_call(debounce_key):
            return None

        prompt = self._build_prompt(changes, timestamp, camera_id, interaction_results)

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
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return None

        self._save_log(text, timestamp, camera_id)
        return text

    def _build_state_changes(self, tracked_list: List[TrackedBBox]) -> List[Dict]:
        changes = []
        for t in tracked_list:
            change = {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "current_state": t.state,
                "prev_state": None,
                "speed": t.speed,
                "direction": _angle_to_direction(t.speed, 0.0),
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
            if c.get("is_new"):
                lines.append(
                    f"- ID:{c['track_id']}, Class:{c['class_name']}: APPEARED, State={state_str}"
                )
            else:
                lines.append(
                    f"- ID:{c['track_id']}, Class:{c['class_name']}: "
                    f"State={state_str}, Speed={c['speed']:.0f}px"
                )

        if interaction_results:
            lines.append("")
            lines.append("Interactions:")
            for ir in interaction_results:
                lines.append(f"- ID:{ir.track_id}, Class:{ir.class_name}: {ir.relation_type} (distance={ir.distance})")

        lines.append("")
        lines.append("Describe the current state changes in one sentence.")
        return "\n".join(lines)

    def _save_log(self, text: str, timestamp: str, camera_id: str):
        log_file = self.log_dir / f"{camera_id}_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")

    def log_appearance(self, tracked: TrackedBBox, camera_id: str = "cam_01") -> Optional[str]:
        self._ensure_client()
        if self.client is None:
            return None

        debounce_key = f"{camera_id}_{tracked.track_id}"
        if not self.debouncer.should_call(debounce_key):
            return None

        timestamp = datetime.now(timezone.utc).isoformat()
        prompt = (
            f"Timestamp: {timestamp}\n"
            f"Camera: {camera_id}\n\n"
            f"A {tracked.class_name} (ID:{tracked.track_id}) has appeared in the frame.\n\n"
            "Describe this event in one sentence."
        )

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
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return None

        self._save_log(text, timestamp, camera_id)
        return text

    def log_disappearance(self, track_id: int, class_name: str, camera_id: str = "cam_01") -> Optional[str]:
        self._ensure_client()
        if self.client is None:
            return None

        timestamp = datetime.now(timezone.utc).isoformat()
        prompt = (
            f"Timestamp: {timestamp}\n"
            f"Camera: {camera_id}\n\n"
            f"The {class_name} (ID:{track_id}) has disappeared from the frame.\n\n"
            "Describe this event in one sentence."
        )

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
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return None

        self._save_log(text, timestamp, camera_id)
        return text


def _angle_to_direction(speed: float, angle: float) -> str:
    if speed < 5:
        return "stationary"
    for (lo, hi), direction in DIRECTION_MAP.items():
        if lo <= angle < hi:
            return direction
    return "unknown"


SPACE_SYSTEM_PROMPT = (
    "You are an object behavior observation specialist. "
    "Given observations from multiple cameras in the same space, "
    "synthesize them into one concise, objective sentence describing the overall situation. "
    "No emotions, no speculation. Output exactly ONE sentence."
)


class SpaceLogger:
    def __init__(self, config: LLMConfig, log_dir: str = "logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debouncer = LLMCallDebouncer(cooldown_seconds=5.0)
        self.client: Optional[OpenAI] = None
        self._buffer: Dict[str, Dict[str, str]] = {}

    def _ensure_client(self):
        if self.client is None and self.config.api_key:
            self.client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
            )

    def collect(self, space_id: str, camera_id: str, text: str):
        if space_id not in self._buffer:
            self._buffer[space_id] = {}
        self._buffer[space_id][camera_id] = text

    def flush(self, space_id: str, space_name: str) -> Optional[str]:
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
        for cam_id, text in sorted(entries.items()):
            prompt_lines.append(f"- {cam_id}: {text}")
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
