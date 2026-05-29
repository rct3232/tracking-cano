#!/usr/bin/env python3
"""Benchmark: memory usage during inference for 33 classes vs 80 classes."""
import time
import logging
import tracemalloc
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
    print(f"  Video FPS={fps:.1f}")

    config = YOLOConfig()
    config.yolo_classes = yolo_classes

    tracemalloc.start()
    tracker = Tracker(config)
    frame_id = 0
    start = time.perf_counter()

    peak_mem = 0
    snapshots = []

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= RUNTIME:
            break

        ret, frame = cap.read()
        if not ret:
            break

        tracker.update(frame, TARGET_CLASSES, frame_id, INTERACTION_CLASSES)
        frame_id += 1

        cur, peak = tracemalloc.get_traced_memory()
        peak_mem = max(peak_mem, peak)

        if frame_id % 5 == 0:
            snapshots.append((frame_id, cur / 1024 / 1024, peak / 1024 / 1024))

    cap.release()
    elapsed = time.perf_counter() - start
    final_cur = tracemalloc.get_traced_memory()[0] / 1024 / 1024
    peak_mb = peak_mem / 1024 / 1024

    print(f"  [{label}] {frame_id} frames in {elapsed:.1f}s")
    print(f"    Peak memory: {peak_mb:.1f} MB")
    print(f"    Final memory: {final_cur:.1f} MB")
    print(f"    Snapshots (frame, current_MB, peak_MB):")
    for s in snapshots:
        print(f"      frame {s[0]}: {s[1]:.1f} / {s[2]:.1f}")
    tracemalloc.stop()

    return frame_id, peak_mb

def main():
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Runtime: {RUNTIME}s per run")
    print("=" * 60)

    print(f"\n--- Run 1: Selected interaction classes (33 total) ---")
    r1 = benchmark("33 classes", TARGET_CLASSES + INTERACTION_CLASSES)

    print(f"\n--- Run 2: All 80 COCO classes ---")
    r2 = benchmark("80 classes", None)

    if r1 and r2:
        f1, p1 = r1
        f2, p2 = r2
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"  33 classes: peak {p1:.1f} MB")
        print(f"  80 classes: peak {p2:.1f} MB")
        diff = p2 - p1
        print(f"  Difference: {diff:+.1f} MB")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
