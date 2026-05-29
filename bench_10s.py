#!/usr/bin/env python3
import argparse
import glob
import os
import threading
import time
import logging
from pathlib import Path
from dataclasses import dataclass

import cv2

from config.config import PipelineConfig, YOLOConfig
from core.pipeline import Pipeline
from modules.tracker import Tracker

logging.basicConfig(level=logging.WARNING)

TARGET_CLASSES = ["cat"]
INTERACTION_CLASSES = [
    "person", "backpack", "umbrella", "handbag", "suitcase", "sports ball",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "chair", "couch", "potted plant", "bed", "dining table", "tv",
    "laptop", "mouse", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "teddy bear",
]
SELECTED_CLASSES = TARGET_CLASSES + INTERACTION_CLASSES

DATA_DIR = Path("data")
DEFAULT_RUNTIME = 10


@dataclass
class BenchResult:
    video_path: str
    label: str
    total_frames: int
    processed: int
    elapsed: float
    fps: float


def discover_videos(data_dir: str = "data") -> list[str]:
    patterns = [os.path.join(data_dir, "*.mp4"), os.path.join(data_dir, "*.mp4.mp4")]
    videos = []
    for p in patterns:
        videos.extend(glob.glob(p))
    seen = set()
    result = []
    for v in sorted(videos):
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def video_info(video_path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0, 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (fps, total)


def _run_detect(video_path: str, runtime: int, yolo_classes: list | None, frame_skip: int, full_video: bool, label: str, results: list, lock: threading.Lock):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    config = YOLOConfig()
    config.model_path = config.model_path or "yolo26s.pt"
    config.yolo_classes = yolo_classes
    tracker = Tracker(config)

    skip_interval = frame_skip + 1 if frame_skip > 0 else 1
    frame_id = 0
    processed = 0
    start = time.perf_counter()

    while True:
        if not full_video and time.perf_counter() - start >= runtime:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % skip_interval == 0:
            tracker.update(frame, TARGET_CLASSES, frame_id, INTERACTION_CLASSES)
            processed += 1
        frame_id += 1

    cap.release()
    elapsed = time.perf_counter() - start
    fps_val = processed / elapsed if elapsed > 0 else 0

    with lock:
        results.append(BenchResult(video_path, label, frame_id, processed, elapsed, fps_val))

    return


def _run_full(video_path: str, runtime: int, frame_skip: int, full_video: bool, results: list, lock: threading.Lock):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    yolo_config = YOLOConfig()
    yolo_config.model_path = yolo_config.model_path or "yolo26s.pt"
    config = PipelineConfig(
        target_classes=TARGET_CLASSES,
        interaction_classes=INTERACTION_CLASSES,
        yolo=yolo_config,
    )
    pipeline = Pipeline(config, camera_id="bench")

    skip_interval = frame_skip + 1 if frame_skip > 0 else 1
    frame_id = 0
    processed = 0
    start = time.perf_counter()

    while True:
        if not full_video and time.perf_counter() - start >= runtime:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % skip_interval == 0:
            pipeline.process_frame(frame, frame_id)
            processed += 1
        frame_id += 1

    cap.release()
    elapsed = time.perf_counter() - start
    fps_val = processed / elapsed if elapsed > 0 else 0

    with lock:
        results.append(BenchResult(video_path, "full", frame_id, processed, elapsed, fps_val))

    return


def bench_concurrent(videos: list[str], mode: str, runtime: int, frame_skip: int, full_video: bool, count: int):
    results: list[BenchResult] = []
    lock = threading.Lock()

    def _select_classes(video_path: str, mode: str) -> list[tuple[str, list | None]]:
        if mode == "detect":
            return [("80c", None), ("sel", SELECTED_CLASSES)]
        if mode == "full":
            return []
        return [("80c", None), ("sel", SELECTED_CLASSES)]

    threads: list[threading.Thread] = []
    batch = videos[:count]

    for video_path in batch:
        # Detect threads
        if mode in ("detect", "both"):
            for label, yolo_classes in _select_classes(video_path, mode):
                t = threading.Thread(
                    target=_run_detect,
                    args=(video_path, runtime, yolo_classes, frame_skip, full_video, label, results, lock),
                    name=f"bench-{label}-{os.path.basename(video_path)}",
                )
                threads.append(t)

        # Full pipeline thread
        if mode in ("full", "both"):
            t = threading.Thread(
                target=_run_full,
                args=(video_path, runtime, frame_skip, full_video, results, lock),
                name=f"bench-full-{os.path.basename(video_path)}",
            )
            threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results, batch


def bench_sequential(videos: list[str], mode: str, runtime: int, frame_skip: int, full_video: bool):
    results: list[BenchResult] = []

    for video_path in videos:
        if mode in ("detect", "both"):
            f1, p1, _, t1 = _bench_detect_sequential(video_path, runtime, None, frame_skip, full_video)
            fps1 = p1 / t1 if t1 > 0 else 0
            results.append(BenchResult(video_path, "80c", f1, p1, t1, fps1))

            f2, p2, _, t2 = _bench_detect_sequential(video_path, runtime, SELECTED_CLASSES, frame_skip, full_video)
            fps2 = p2 / t2 if t2 > 0 else 0
            results.append(BenchResult(video_path, "sel", f2, p2, t2, fps2))

        if mode in ("full", "both"):
            f3, p3, _, t3 = _bench_full_sequential(video_path, runtime, frame_skip, full_video)
            fps3 = p3 / t3 if t3 > 0 else 0
            results.append(BenchResult(video_path, "full", f3, p3, t3, fps3))

    return results, videos


def _bench_detect_sequential(video_path: str, runtime: int, yolo_classes: list | None, frame_skip: int, full_video: bool) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0, 0, 0, 0)

    config = YOLOConfig()
    config.model_path = config.model_path or "yolo26s.pt"
    config.yolo_classes = yolo_classes
    tracker = Tracker(config)

    skip_interval = frame_skip + 1 if frame_skip > 0 else 1
    frame_id = 0
    processed = 0
    start = time.perf_counter()

    while True:
        if not full_video and time.perf_counter() - start >= runtime:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % skip_interval == 0:
            tracker.update(frame, TARGET_CLASSES, frame_id, INTERACTION_CLASSES)
            processed += 1
        frame_id += 1

    cap.release()
    elapsed = time.perf_counter() - start
    return (frame_id, processed, skip_interval, elapsed)


def _bench_full_sequential(video_path: str, runtime: int, frame_skip: int, full_video: bool) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0, 0, 0, 0)

    yolo_config = YOLOConfig()
    yolo_config.model_path = yolo_config.model_path or "yolo26s.pt"
    config = PipelineConfig(
        target_classes=TARGET_CLASSES,
        interaction_classes=INTERACTION_CLASSES,
        yolo=yolo_config,
    )
    pipeline = Pipeline(config, camera_id="bench")

    skip_interval = frame_skip + 1 if frame_skip > 0 else 1
    frame_id = 0
    processed = 0
    start = time.perf_counter()

    while True:
        if not full_video and time.perf_counter() - start >= runtime:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % skip_interval == 0:
            pipeline.process_frame(frame, frame_id)
            processed += 1
        frame_id += 1

    cap.release()
    elapsed = time.perf_counter() - start
    return (frame_id, processed, skip_interval, elapsed)


def _print_results(results: list[BenchResult], videos: list[str], full_video: bool):
    for video_path in videos:
        fps_val, total = video_info(video_path)
        print(f"\n  {os.path.basename(video_path)} (FPS={fps_val:.1f}, Total={total})")

        for r in results:
            if r.video_path != video_path:
                continue
            if full_video:
                print(f"  [{r.label:4s}]  {r.processed}/{r.total_frames} frames in {r.elapsed:.1f}s ({r.fps:.1f} fps)")
            else:
                print(f"  [{r.label:4s}]  {r.processed} frames in {r.elapsed:.1f}s ({r.fps:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="10-second frame benchmark")
    parser.add_argument("--mode", choices=["detect", "full", "both"], default="detect",
                        help="Benchmark mode: detect (default), full, both")
    parser.add_argument("--video", type=str, default=None,
                        help="Specific video path (default: scan data/)")
    parser.add_argument("--runtime", type=int, default=DEFAULT_RUNTIME,
                        help=f"Runtime in seconds (default: {DEFAULT_RUNTIME})")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory to scan for videos (default: data)")
    parser.add_argument("--frame-skip", type=int, default=None,
                        help="Frame skip (N frames skipped between each processed frame, 0=none)")
    parser.add_argument("--full-video", action="store_true",
                        help="Process entire video instead of time-limited run")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of videos to process concurrently (default: 1)")
    args = parser.parse_args()

    if args.video:
        videos = [args.video]
    else:
        videos = discover_videos(args.data_dir)

    if not videos:
        print("No videos found.")
        return

    frame_skip = args.frame_skip if args.frame_skip is not None else YOLOConfig().frame_skip

    print("=" * 60)
    mode_str = f"Mode: {args.mode}"
    skip_str = f"Skip: {frame_skip}"
    count_str = f"Concurrent: {args.count}"
    if args.full_video:
        mode_str += " | Full video"
    else:
        mode_str += f" | Runtime: {args.runtime}s"
    print(f"{mode_str} | {skip_str} | {count_str} | Videos: {len(videos)}")
    print("=" * 60)

    if args.count > 1:
        results, batch = bench_concurrent(videos, args.mode, args.runtime, frame_skip, args.full_video, args.count)
    else:
        results, batch = bench_sequential(videos, args.mode, args.runtime, frame_skip, args.full_video)

    _print_results(results, batch, args.full_video)

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
