from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import logging
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)
from config.config import YOLOConfig, Thresholds
from modules.tile_detector import HybridDetector


class MovementState(Enum):
    STOPPED = auto()
    SLOW_MOVE = auto()
    FAST_MOVE = auto()
    DASHING = auto()
    ROTATING = auto()


@dataclass
class TrackedBBox:
    track_id: int
    frame_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    prev_bbox: Optional[Tuple[int, int, int, int]] = None
    state: Optional[MovementState] = None
    speed: float = 0.0
    acceleration: float = 0.0
    prev_speed: float = 0.0


class Tracker:
    _CLASS_NAME_MAP = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
        19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
        24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
        29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball",
        33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard",
        37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
        41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
        50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
        55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
        60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
        65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
        69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
        73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
        78: "hair drier", 79: "toothbrush",
    }

    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        self._history: dict[int, TrackedBBox] = {}

    def _ensure_loaded(self):
        if self.model is None:
            self.model = YOLO(self.config.model_path)
            if self.config.quantize:
                self.model = self.model.quantize()
            self.detector = HybridDetector(
                self.model,
                grid_x=self.config.tile_grid_x,
                grid_y=self.config.tile_grid_y,
                overlap=self.config.tile_overlap,
                enabled=self.config.tile_enabled,
                yolo_classes=self.config.yolo_classes,
            )
            logger.info("Model loaded: %s (quantize=%s)", self.config.model_path, self.config.quantize)

    def update(self, frame: np.ndarray, target_classes: List[str], frame_id: int, interaction_classes: List[str] | None = None) -> tuple[List[TrackedBBox], List[TrackedBBox]]:
        if interaction_classes is None:
            all_classes = None  # YOLO 전체 80클래스 감지
        else:
            all_classes = list(dict.fromkeys(target_classes + (interaction_classes or [])))
        self._ensure_loaded()
        try:
            result = self.detector.detect(
                frame,
                conf=self.config.conf_threshold,
                target_classes=all_classes or target_classes,
                iou=self.config.iou_threshold,
            )
        except Exception as e:
            logger.error("Detection error: %s", e)
            return [], []

        if not result or result.boxes is None:
            logger.debug("No detections")
            return [], []

        boxes = result.boxes
        if not hasattr(boxes, "id") or boxes.id is None or len(boxes.id) == 0:
            logger.debug("No tracking IDs from ByteTrack")
            return [], []

        class_name_map = self._CLASS_NAME_MAP
        target_id_set = {k for k, v in class_name_map.items() if v in target_classes} if target_classes else set()
        if interaction_classes is None:
            interaction_id_set = set(class_name_map.keys())  # COCO 80 전체
        elif interaction_classes:
            interaction_id_set = {k for k, v in class_name_map.items() if v in interaction_classes}
        else:
            interaction_id_set = set()  # [] → interaction 없음

        tracked: List[TrackedBBox] = []
        interactions: List[TrackedBBox] = []
        current_ids: set[int] = set()

        for i in range(len(boxes.xyxy)):
            track_id = int(boxes.id[i].item())
            xyxy = boxes.xyxy[i].cpu().tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])

            is_target = cls_id in target_id_set
            is_interaction = cls_id in interaction_id_set
            if not is_target and not is_interaction:
                continue

            tb = TrackedBBox(
                track_id=track_id,
                frame_id=frame_id,
                x1=int(xyxy[0]),
                y1=int(xyxy[1]),
                x2=int(xyxy[2]),
                y2=int(xyxy[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=class_name_map.get(cls_id, f"class_{cls_id}"),
            )

            if track_id in self._history:
                prev = self._history[track_id]
                tb.prev_bbox = (prev.x1, prev.y1, prev.x2, prev.y2)
                tb.prev_speed = prev.speed

                prev_center = self._center(prev)
                curr_center = self._center(tb)
                tb.speed = float(np.sqrt(
                    (curr_center[0] - prev_center[0]) ** 2
                    + (curr_center[1] - prev_center[1]) ** 2
                ))
                tb.acceleration = tb.speed - tb.prev_speed

            current_ids.add(track_id)
            if is_target:
                tracked.append(tb)
            if is_interaction:
                interactions.append(tb)

        self._history = {t.track_id: t for t in tracked + interactions}
        logger.debug("Detect: %d boxes, %d tracked, %d interaction, %d unique IDs", len(boxes), len(tracked), len(interactions), len(current_ids))
        return tracked, interactions

    @staticmethod
    def _center(bbox: TrackedBBox) -> Tuple[float, float]:
        return ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)
