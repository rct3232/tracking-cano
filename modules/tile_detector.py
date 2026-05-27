import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

class HybridDetector:
    def __init__(self, model: YOLO, grid_x: int = 2, grid_y: int = 2, overlap: int = 20, enabled: bool = True):
        self.model = model
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.overlap = overlap
        self.enabled = enabled

    def detect(self, frame: np.ndarray, conf: float, target_classes: List[str], iou: float = 0.70) -> object:
        target_class_ids = self._resolve_class_ids(target_classes)
        results = self.model.track(frame, conf=conf, iou=iou, persist=False, verbose=False)
        has_tracks = results and results[0].boxes is not None and results[0].boxes.id is not None and len(results[0].boxes.id) > 0
        if not has_tracks:
            if self.enabled:
                return self._tile_detect(frame, conf, target_class_ids, iou)
            return results[0] if results else None
        if self._has_target(results[0].boxes, target_class_ids):
            return results[0]
        if self.enabled:
            return self._tile_detect(frame, conf, target_class_ids, iou)
        return results[0]

    def _tile_detect(self, frame: np.ndarray, conf: float, target_class_ids: set, iou: float) -> object:
        h, w = frame.shape[:2]
        tile_w = int(w / self.grid_x)
        tile_h = int(h / self.grid_y)
        overlap_x = int(tile_w * self.overlap / 100)
        overlap_y = int(tile_h * self.overlap / 100)

        all_boxes = []
        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                x1 = max(0, gx * tile_w - overlap_x)
                y1 = max(0, gy * tile_h - overlap_y)
                x2 = min(w, (gx + 1) * tile_w + overlap_x)
                y2 = min(h, (gy + 1) * tile_h + overlap_y)
                tile = frame[y1:y2, x1:x2]
                results = self.model(tile, conf=conf, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1b, y1b, x2b, y2b = box.xyxy[0].cpu().tolist()
                        global_x1 = x1 + x1b
                        global_y1 = y1 + y1b
                        global_x2 = x1 + x2b
                        global_y2 = y1 + y2b
                        cls_id = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        all_boxes.append({
                            "x1": global_x1, "y1": global_y1,
                            "x2": global_x2, "y2": global_y2,
                            "cls": cls_id, "conf": conf_val,
                        })
        all_boxes = self._nms(all_boxes, iou)
        return self._build_result(all_boxes, frame)

    def _has_target(self, boxes: object, target_class_ids: set) -> bool:
        if not hasattr(boxes, "cls") or boxes.cls is None or len(boxes.cls) == 0:
            return False
        for cls_id in boxes.cls:
            if int(cls_id.item()) in target_class_ids:
                return True
        return False

    def _resolve_class_ids(self, target_classes: List[str]) -> set:
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
        return {k for k, v in class_name_map.items() if v in target_classes}

    def _nms(self, boxes: List[dict], iou_threshold: float) -> List[dict]:
        if not boxes:
            return []
        boxes.sort(key=lambda b: b["conf"], reverse=True)
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [
                b for b in boxes
                if self._iou(best, b) < iou_threshold
            ]
        return keep

    def _iou(self, a: dict, b: dict) -> float:
        x1 = max(a["x1"], b["x1"])
        y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"])
        y2 = min(a["y2"], b["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _build_result(self, boxes: List[dict], frame: np.ndarray) -> object:
        class FakeBoxes:
            pass
        class FakeResults:
            pass
        fake = FakeResults()
        fake.boxes = FakeBoxes()
        fake.boxes.xyxy = torch_tensor_from_boxes(boxes)
        fake.boxes.conf = torch_tensor_from_list([b["conf"] for b in boxes])
        fake.boxes.cls = torch_tensor_from_list([b["cls"] for b in boxes])
        fake.boxes.id = None
        return fake

def torch_tensor_from_boxes(boxes: List[dict]):
    import torch
    data = []
    for b in boxes:
        data.append([b["x1"], b["y1"], b["x2"], b["y2"]])
    return torch.tensor(data, dtype=torch.float32)

def torch_tensor_from_list(lst: list):
    import torch
    return torch.tensor(lst, dtype=torch.float32)
