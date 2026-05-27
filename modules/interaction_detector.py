from dataclasses import dataclass
from typing import List, Optional

from modules.tracker import TrackedBBox


@dataclass
class InteractionResult:
    track_id: int
    class_name: str
    relation_type: str  # "contact" | "nearby" | "interacting"
    distance: float


class InteractionDetector:
    def __init__(self, overlap_threshold: float = 0.3, distance_threshold: float = 50.0):
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold

    def detect(self, target: TrackedBBox, interactions: List[TrackedBBox]) -> List[InteractionResult]:
        if not interactions:
            return []
        results: List[InteractionResult] = []
        for obj in interactions:
            iou = self._iou(target, obj)
            dist = self._distance(target, obj)
            has_overlap = iou >= self.overlap_threshold
            has_proximity = dist <= self.distance_threshold
            is_contained = dist == 0.0
            if has_overlap and has_proximity:
                relation = "interacting"
            elif is_contained:
                relation = "interacting"
            elif has_overlap:
                relation = "contact"
            elif has_proximity:
                relation = "nearby"
            else:
                continue
            results.append(InteractionResult(
                track_id=obj.track_id,
                class_name=obj.class_name,
                relation_type=relation,
                distance=round(dist, 1),
            ))
        return results

    @staticmethod
    def _iou(a: TrackedBBox, b: TrackedBBox) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _distance(a: TrackedBBox, b: TrackedBBox) -> float:
        dx = max(0, max(b.x1 - a.x2, a.x1 - b.x2))
        dy = max(0, max(b.y1 - a.y2, a.y1 - b.y2))
        return (dx ** 2 + dy ** 2) ** 0.5
