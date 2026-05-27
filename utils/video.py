import os
from typing import Optional

import cv2


def create_capture(source) -> Optional[cv2.VideoCapture]:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    elif isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://")):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "reconnect;1"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 32)
    elif isinstance(source, str):
        cap = cv2.VideoCapture(source)
    else:
        return None

    if not cap.isOpened():
        return None
    return cap
