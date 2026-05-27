from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ultralytics import YOLO

from config.config import YOLOConfig

_COCO_CLASSES = {
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


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


class Detector:
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.model: Optional[YOLO] = None

    def _ensure_loaded(self):
        if self.model is None:
            self.model = YOLO(self.config.model_path)

    def detect(self, frame: np.ndarray, target_classes: List[str]) -> List[BBox]:
        self._ensure_loaded()
        try:
            results = self.model(
                frame,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )
        except Exception:
            return []

        if not results or not results[0].boxes:
            return []

        class_id_set = (
            {k for k, v in _COCO_CLASSES.items() if v in target_classes}
            if target_classes
            else set(_COCO_CLASSES.keys())
        )

        boxes = results[0].boxes
        bboxes: List[BBox] = []
        for i in range(len(boxes.xyxy)):
            xyxy = boxes.xyxy[i].cpu().tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])

            if cls_id not in class_id_set:
                continue

            bboxes.append(BBox(
                x1=int(xyxy[0]),
                y1=int(xyxy[1]),
                x2=int(xyxy[2]),
                y2=int(xyxy[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=_COCO_CLASSES.get(cls_id, f"class_{cls_id}"),
            ))

        return bboxes
