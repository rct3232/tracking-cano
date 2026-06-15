import base64
from typing import List

import cv2
import numpy as np

from modules.tracker import TrackedBBox


_STATE_COLORS = {
    "STOPPED": (0, 0, 255),
    "SLOW_MOVE": (0, 165, 255),
    "FAST_MOVE": (0, 255, 0),
    "DASHING": (255, 0, 0),
    "ROTATING": (160, 32, 240),
}


def annotate_image(
    frame: np.ndarray,
    tracked_list: List[TrackedBBox],
    quality: int = 60,
    max_width: int = 1024,
) -> str:
    vis = frame.copy()

    if max_width > 0 and vis.shape[1] > max_width:
        scale = max_width / vis.shape[1]
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        scale_bboxes = scale
    else:
        scale_bboxes = 1.0

    for t in tracked_list:
        x1 = int(t.x1 * scale_bboxes)
        y1 = int(t.y1 * scale_bboxes)
        x2 = int(t.x2 * scale_bboxes)
        y2 = int(t.y2 * scale_bboxes)

        color = _STATE_COLORS.get(t.state.name if t.state else "STOPPED", (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label_parts = [f"id:{t.track_id}", t.class_name]
        if t.state:
            label_parts.append(t.state.name)
        label = " ".join(label_parts)
        font_scale = max(0.4, min(0.7, (x2 - x1) / 300))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw_normalized_bbox(
    image_b64: str,
    coords: list[float],
    color: tuple[int, int, int] = (0, 255, 0),
    label: str = "",
    quality: int = 60,
) -> str:
    img_bytes = base64.b64decode(image_b64)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return image_b64
    h, w = img.shape[:2]
    # Support both pixel coords and normalized coords
    if any(v > 1.0 for v in coords):
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    else:
        x1 = int(coords[0] * w)
        y1 = int(coords[1] * h)
        x2 = int(coords[2] * w)
        y2 = int(coords[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")
