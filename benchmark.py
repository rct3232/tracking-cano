#!/usr/bin/env python3
import time
import logging
from modules.tracker import Tracker
from config.config import YOLOConfig

logging.basicConfig(level=logging.WARNING)

VIDEO_PATHS = [
    "data/cam_rec_hallway_20260527-102400.mp4.mp4",
    "data/cam_rec_livingfront_20260527-102300.mp4.mp4",
    "data/cam_rec_livingroom_20260527-113900.mp4.mp4",
]

TARGET_CLASSES = ["cat"]
INTERACTION_CLASSES = [
    "person", "backpack", "umbrella", "handbag", "suitcase", "sports ball",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "chair", "couch", "potted plant", "bed", "dining table", "tv",
    "laptop", "mouse", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "teddy bear",
]

RUNTIME = 30  # seconds

def benchmark(video_path: str, label: str, yolo_classes: list | None):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {video_path}, FPS={fps:.1f}, Total={total_frames}")

    config = YOLOConfig()
    config.yolo_classes = yolo_classes

    tracker = Tracker(config)
    frame_id = 0
    start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= RUNTIME:
            break

        ret, frame = cap.read()
        if not ret:
            break

        tracker.update(frame, TARGET_CLASSES, frame_id, INTERACTION_CLASSES)
        frame_id += 1

    cap.release()
    elapsed = time.perf_counter() - start
    print(f"  [{label}] {frame_id} frames in {elapsed:.1f}s ({frame_id / elapsed:.1f} fps)")
    return frame_id, elapsed

def main():
    for video_path in VIDEO_PATHS:
        print(f"\n=== {video_path} ===")
        benchmark(video_path, "80 classes", None)
        benchmark(video_path, "38 classes", TARGET_CLASSES + INTERACTION_CLASSES)

if __name__ == "__main__":
    main()
