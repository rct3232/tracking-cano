from typing import Any, Dict, Tuple
import numpy as np

from config.config import Thresholds
from modules.tracker import MovementState, TrackedBBox


def classify_movement(tracked: TrackedBBox, thresholds: Thresholds) -> Tuple[MovementState, Dict[str, Any]]:
    if tracked.prev_bbox is None:
        return MovementState.STOPPED, {
            "speed": 0.0,
            "acceleration": 0.0,
            "direction_angle": 0.0,
        }

    speed = tracked.speed
    acceleration = tracked.acceleration
    direction_angle = _compute_direction(tracked)

    slow_thresh = thresholds.speed_slow
    if tracked.state is not MovementState.STOPPED:
        slow_thresh = thresholds.speed_slow - thresholds.hysteresis

    state = _classify(speed, acceleration, direction_angle, slow_thresh, thresholds)

    return state, {
        "speed": round(speed, 2),
        "acceleration": round(acceleration, 2),
        "direction_angle": round(direction_angle, 2),
    }


def _classify(
    speed: float,
    acceleration: float,
    direction_angle: float,
    slow_thresh: float,
    thresholds: Thresholds,
) -> MovementState:
    if speed < slow_thresh:
        if direction_angle > thresholds.rotation_threshold:
            return MovementState.ROTATING
        return MovementState.STOPPED

    if acceleration >= thresholds.dash_threshold:
        return MovementState.DASHING

    if speed >= thresholds.speed_fast:
        if direction_angle > thresholds.rotation_threshold:
            return MovementState.ROTATING
        return MovementState.FAST_MOVE

    return MovementState.SLOW_MOVE


def _compute_direction(tracked: TrackedBBox) -> float:
    if tracked.prev_bbox is None:
        return 0.0

    prev_cx = (tracked.prev_bbox[0] + tracked.prev_bbox[2]) / 2
    prev_cy = (tracked.prev_bbox[1] + tracked.prev_bbox[3]) / 2
    curr_cx = (tracked.x1 + tracked.x2) / 2
    curr_cy = (tracked.y1 + tracked.y2) / 2

    dx = curr_cx - prev_cx
    dy = curr_cy - prev_cy

    if dx == 0 and dy == 0:
        return 0.0

    angle = np.degrees(np.arctan2(dx, -dy))
    if angle < 0:
        angle += 360
    return angle
