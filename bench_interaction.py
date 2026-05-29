#!/usr/bin/env python3
"""Benchmark: interaction_classes (33) vs all 80 COCO classes on livingfront video."""
import time
import logging
import cv2

from config.config import YOLOConfig
from modules.tracker import Tracker

logging.basicConfig(level=logging.WARNING)

VIDEO_PATH = "data/cam_rec_livingfront_20260527-102300.mp4.mp4"
TARGET_CLASSES = ["cat"]
INTERACTION_CLASSES = [
    "person", "backpack", "umbrella", "handbag", "suitcase", "sports ball",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "chair", "couch", "potted plant", "bed", "dining table", "tv",
    "laptop", "mouse", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "teddy bear",
]
RUNTIME = 30  # seconds

def benchmark(label: str, yolo_classes: list | None):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  [ERR] Cannot open {VIDEO_PATH}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video FPS={fps:.1f}, Total frames={total_frames}")

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
    fps_actual = frame_id / elapsed if elapsed > 0 else 0
    print(f"  [{label}] {frame_id} frames in {elapsed:.1f}s ({fps_actual:.1f} fps)")
    return frame_id, elapsed

def main():
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Runtime: {RUNTIME}s per run")
    print(f"Target classes: {TARGET_CLASSES}")
    print(f"Interaction classes: {len(INTERACTION_CLASSES)} items")
    print("=" * 60)

    print(f"\n--- Run 1: Selected interaction classes (33 total) ---")
    r1 = benchmark("33 classes", TARGET_CLASSES + INTERACTION_CLASSES)

    print(f"\n--- Run 2: All 80 COCO classes ---")
    r2 = benchmark("80 classes", None)

    if r1 and r2:
        f1, t1 = r1
        f2, t2 = r2
        ratio = f1 / f2 if f2 > 0 else 0
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"  33 classes: {f1} frames ({f1/t1:.1f} fps)")
        print(f"  80 classes: {f2} frames ({f2/t2:.1f} fps)")
        print(f"  Ratio (33/80): {ratio:.2%}")
        print(f"  Difference: {f1 - f2} frames")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
