import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np

from config.config import PipelineConfig
from modules.analyzer import classify_movement
from modules.interaction_detector import InteractionDetector, InteractionResult
from modules.tracker import MovementState, Tracker, TrackedBBox
from nlp.logger import NLPLogger, SpaceLogger

logger = logging.getLogger(__name__)


@dataclass
class DetectResult:
    target_present: bool
    class_name: Optional[str] = None
    target_coordinate: Optional[List[float]] = None
    tracked_ids: Optional[List[int]] = None


@dataclass
class LogEvent:
    tracked_list: List[TrackedBBox] = field(default_factory=list)
    frame: Optional[np.ndarray] = None
    interactions: Optional[List[InteractionResult]] = None
    target_coordinate: Optional[List[float]] = None
    target_classes: Optional[List[str]] = None


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
        self._prev_interactions: dict[int, List[InteractionResult]] = {}
        self._class_names: dict[int, str] = {}

    def _normalize_bbox(self, t: TrackedBBox, frame: np.ndarray) -> List[float]:
        h, w = frame.shape[:2]
        return [t.x1 / w, t.y1 / h, t.x2 / w, t.y2 / h]

    def process_frame(self, frame: np.ndarray, frame_id: int) -> tuple[DetectResult, Optional[LogEvent]]:
        tracked_list, interaction_list = self.tracker.update(
            frame, self.config.target_classes, frame_id, self.config.interaction_classes
        )

        if not tracked_list:
            self._check_disappeared(set())
            logger.debug("[%s] frame=%d: no targets, %d interactions", self.camera_id, frame_id, len(interaction_list) if interaction_list else 0)
            return DetectResult(target_present=False), None

        current_ids: Set[int] = {t.track_id for t in tracked_list}
        self._check_disappeared(current_ids)

        primary = tracked_list[0]
        target_coord = self._normalize_bbox(primary, frame)
        detect = DetectResult(
            target_present=True,
            class_name=primary.class_name,
            target_coordinate=target_coord,
            tracked_ids=[t.track_id for t in tracked_list],
        )

        log_event: Optional[LogEvent] = None
        for t in tracked_list:
            state, meta = classify_movement(t, self.config.thresholds)
            t.state = state
            t.speed = meta["speed"]
            t.direction_angle = meta["direction_angle"]

            interactions = self.interaction_detector.detect(t, interaction_list)

            is_new = t.prev_bbox is None
            if is_new:
                self._prev_states[t.track_id] = state
                self._prev_frame_ids[t.track_id] = frame_id
                self._state_hold[t.track_id] = 1
                self._prev_interactions[t.track_id] = interactions
                self._class_names[t.track_id] = t.class_name
                continue

            hold = self._state_hold.get(t.track_id, 0) + 1
            self._state_hold[t.track_id] = hold

            prev_state = self._prev_states.get(t.track_id)
            if prev_state is None:
                self._prev_states[t.track_id] = state
                self._prev_frame_ids[t.track_id] = frame_id
                self._prev_interactions[t.track_id] = interactions
                continue

            if prev_state != state:
                logger.debug("[%s] target %d state %s->%s hold=%d/%d", self.camera_id, t.track_id, prev_state.name, state.name, hold, self.config.thresholds.min_frames)
                if hold >= self.config.thresholds.min_frames:
                    self.nlp_logger.log([t], frame, self.camera_id, interactions, self.space_logger, self.space_id, target_classes=self.config.target_classes)
                    log_event = LogEvent(
                        tracked_list=[t],
                        frame=frame,
                        interactions=interactions,
                        target_coordinate=target_coord,
                        target_classes=self.config.target_classes,
                    )
                    self._prev_states[t.track_id] = state
                    self._prev_frame_ids[t.track_id] = frame_id
                    self._state_hold[t.track_id] = 0
                    self._prev_interactions[t.track_id] = interactions
            else:
                if hold >= self.config.thresholds.min_frames:
                    self._state_hold[t.track_id] = 0

                prev_interactions = self._prev_interactions.get(t.track_id)
                if self._interactions_changed(prev_interactions, interactions):
                    logger.debug("[%s] target %d interactions changed", self.camera_id, t.track_id)
                    self.nlp_logger.log([t], frame, self.camera_id, interactions, self.space_logger, self.space_id, target_classes=self.config.target_classes)
                    log_event = LogEvent(
                        tracked_list=[t],
                        frame=frame,
                        interactions=interactions,
                        target_coordinate=target_coord,
                        target_classes=self.config.target_classes,
                    )
                    self._prev_interactions[t.track_id] = interactions

        return detect, log_event

    def stop(self):
        self.nlp_logger.stop()

    def _check_disappeared(self, current_ids: Set[int]) -> List[tuple]:
        disappeared = []
        for t_id in list(self._prev_states.keys()):
            if t_id not in current_ids:
                cls_name = self._class_names.get(t_id, "unknown")
                disappeared.append((t_id, cls_name))
                del self._prev_states[t_id]
                self._prev_frame_ids.pop(t_id, None)
                self._state_hold.pop(t_id, None)
                self._prev_interactions.pop(t_id, None)
                logger.info("[%s] disappeared: target %d (%s)", self.camera_id, t_id, cls_name)
        return disappeared

    def _interactions_changed(self, prev: List[InteractionResult] | None, curr: List[InteractionResult]) -> bool:
        if prev is None:
            return len(curr) > 0
        if len(prev) != len(curr):
            return True
        prev_ids = {ir.track_id for ir in prev}
        curr_ids = {ir.track_id for ir in curr}
        return prev_ids != curr_ids
