from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from config.config import YOLOConfig, Thresholds


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
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        self._history: dict[int, TrackedBBox] = {}

    def _ensure_loaded(self):
        if self.model is None:
            self.model = YOLO(self.config.model_path)

    def update(self, frame: np.ndarray, target_classes: List[str], frame_id: int) -> List[TrackedBBox]:
        self._ensure_loaded()
        try:
            results = self.model.track(
                frame,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                persist=True,
                verbose=False,
            )
        except Exception:
            return []

        if not results or not results[0].boxes:
            return []

        boxes = results[0].boxes
        if not hasattr(boxes, "id") or boxes.id is None:
            return []

        class_name_map = {
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

        class_id_set = (
            {k for k, v in class_name_map.items() if v in target_classes}
            if target_classes
            else set(class_name_map.keys())
        )

        tracked: List[TrackedBBox] = []
        current_ids: set[int] = set()

        for i in range(len(boxes.xyxy)):
            track_id = int(boxes.id[i].item())
            xyxy = boxes.xyxy[i].cpu().tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])

            if cls_id not in class_id_set:
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
            tracked.append(tb)

        self._history = {t.track_id: t for t in tracked}
        return tracked

    @staticmethod
    def _center(bbox: TrackedBBox) -> Tuple[float, float]:
        return ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)
