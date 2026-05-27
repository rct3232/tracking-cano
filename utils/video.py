from typing import Optional

import cv2


def create_capture(source) -> Optional[cv2.VideoCapture]:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    elif isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://")):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    elif isinstance(source, str):
        cap = cv2.VideoCapture(source)
    else:
        return None

    if not cap.isOpened():
        return None
    return cap
