import logging
from typing import List, Optional, Set

from config.config import PipelineConfig
from modules.analyzer import classify_movement
from modules.interaction_detector import InteractionDetector, InteractionResult
from modules.tracker import MovementState, Tracker, TrackedBBox
from nlp.logger import NLPLogger, SpaceLogger

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: PipelineConfig, camera_id: str = "cam_01", space_logger: Optional[SpaceLogger] = None, space_id: Optional[str] = None):
        self.config = config
        self.camera_id = camera_id
        self.space_logger = space_logger
        self.space_id = space_id
        self.tracker = Tracker(config.yolo)
        self.nlp_logger = NLPLogger(config.llm)
        self.interaction_detector = InteractionDetector(
            overlap_threshold=config.thresholds.overlap,
            distance_threshold=config.thresholds.distance,
        )
        self._prev_states: dict[int, MovementState] = {}
        self._prev_frame_ids: dict[int, int] = {}
        self._state_hold: dict[int, int] = {}

    def process_frame(self, frame, frame_id: int) -> Optional[str]:
        tracked_list, interaction_list = self.tracker.update(
            frame, self.config.target_classes, frame_id, self.config.interaction_classes
        )

        if not tracked_list:
            disappeared = self._check_disappeared(set())
            for t_id, cls_name in disappeared:
                text = self.nlp_logger.log_disappearance(t_id, cls_name, self.camera_id)
                if text:
                    self._collect(text)
                return text
            return None

        current_ids: Set[int] = {t.track_id for t in tracked_list}
        disappeared = self._check_disappeared(current_ids)
        for t_id, cls_name in disappeared:
                text = self.nlp_logger.log_disappearance(t_id, cls_name, self.camera_id)
                if text:
                    self._collect(text)

        results: List[str] = []
        for t in tracked_list:
            state, meta = classify_movement(t, self.config.thresholds)
            t.state = state
            t.speed = meta["speed"]

            interactions = self.interaction_detector.detect(t, interaction_list)

            is_new = t.prev_bbox is None
            if is_new:
                text = self.nlp_logger.log_appearance(t, self.camera_id)
                if text:
                    results.append(text)
                    self._collect(text)
                self._prev_states[t.track_id] = state
                self._prev_frame_ids[t.track_id] = frame_id
                self._state_hold[t.track_id] = 1
                continue

            hold = self._state_hold.get(t.track_id, 0) + 1
            self._state_hold[t.track_id] = hold

            prev_state = self._prev_states.get(t.track_id)
            if prev_state is None:
                self._prev_states[t.track_id] = state
                self._prev_frame_ids[t.track_id] = frame_id
                continue

            if prev_state != state:
                if hold >= self.config.thresholds.min_frames:
                    text = self.nlp_logger.log([t], self.camera_id, interactions)
                    if text:
                        results.append(text)
                        self._collect(text)
                    self._prev_states[t.track_id] = state
                    self._prev_frame_ids[t.track_id] = frame_id
                    self._state_hold[t.track_id] = 0
            else:
                if hold >= self.config.thresholds.min_frames:
                    self._state_hold[t.track_id] = 0

        return " | ".join(results) if results else None

    def _collect(self, text: str):
        if self.space_logger and self.space_id:
            self.space_logger.collect(self.space_id, self.camera_id, text)
            self.space_logger.try_flush(self.space_id, self.space_id)

    def _check_disappeared(self, current_ids: Set[int]) -> List[tuple]:
        disappeared = []
        for t_id in list(self._prev_states.keys()):
            if t_id not in current_ids:
                cls_name = "unknown"
                for t in self.tracker._history.values():
                    if t.track_id == t_id:
                        cls_name = t.class_name
                        break
                disappeared.append((t_id, cls_name))
                del self._prev_states[t_id]
                self._prev_frame_ids.pop(t_id, None)
                self._state_hold.pop(t_id, None)
        return disappeared
