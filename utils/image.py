import base64
from typing import List

import cv2
import numpy as np


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
    # Support pixel coords, 0-1000 scale, and normalized (0-1) coords
    if any(v > 1.0 for v in coords):
        if all(0 <= v <= 1000 for v in coords):
            x1 = int(coords[0] * w / 1000)
            y1 = int(coords[1] * h / 1000)
            x2 = int(coords[2] * w / 1000)
            y2 = int(coords[3] * h / 1000)
        else:
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
