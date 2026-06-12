#!/usr/bin/env python3
"""Unified benchmark: detect, full pipeline, memory, RTSP, config-based multi-camera."""
import argparse
import glob
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2

from settings import LLMConfig, PipelineConfig, Thresholds, YOLOConfig
from core.config_manager import load_config
from core.pipeline import Pipeline
from modules.tracker import Tracker
from utils.video import create_capture

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
_MODEL_PATH: str | None = None


@dataclass
class BenchResult:
    video_path: str
    label: str
    total_frames: int
    processed: int
    elapsed: float
    fps: float


@dataclass
class MemoryResult:
    label: str
    peak_mb: float
    final_mb: float
    snapshots: list = field(default_factory=list)

    def add_snapshot(self, frame_id: int, cur_mb: float, peak_mb: float):
        self.snapshots.append((frame_id, cur_mb, peak_mb))


@dataclass
class CamStats:
    camera_id: str
    frames_read: int = 0
    frames_inferred: int = 0
    infer_times: list = field(default_factory=list)


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


# --- Detect-only runs ---

def _run_detect(video_path: str, runtime: int, yolo_classes: list | None, frame_skip: int, full_video: bool, label: str, results: list, lock: threading.Lock):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    config = YOLOConfig()
    config.model_path = _MODEL_PATH or "yolo26s.pt"
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


def _run_detect_sequential(video_path: str, runtime: int, yolo_classes: list | None, frame_skip: int, full_video: bool) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0, 0, 0, 0)

    config = YOLOConfig()
    config.model_path = _MODEL_PATH or "yolo26s.pt"
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


# --- Full pipeline runs ---

def _run_full(video_path: str, runtime: int, frame_skip: int, full_video: bool, results: list, lock: threading.Lock):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    yolo_config = YOLOConfig()
    yolo_config.model_path = _MODEL_PATH or "yolo26s.pt"
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


def _run_full_sequential(video_path: str, runtime: int, frame_skip: int, full_video: bool) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0, 0, 0, 0)

    yolo_config = YOLOConfig()
    yolo_config.model_path = _MODEL_PATH or "yolo26s.pt"
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


# --- Memory benchmark ---

def _run_memory(video_path: str, runtime: int, yolo_classes: list | None, label: str, results: list, lock: threading.Lock):
    import tracemalloc
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    config = YOLOConfig()
    config.model_path = _MODEL_PATH or "yolo26s.pt"
    config.yolo_classes = yolo_classes
    tracemalloc.start()
    tracker = Tracker(config)

    frame_id = 0
    start = time.perf_counter()
    peak_mem = 0
    mem_result = MemoryResult(label=label, peak_mb=0, final_mb=0)

    # GPU memory tracking
    cuda_peak_mb = 0
    has_cuda = False
    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= runtime:
            break
        ret, frame = cap.read()
        if not ret:
            break

        tracker.update(frame, TARGET_CLASSES, frame_id, INTERACTION_CLASSES)
        frame_id += 1

        cur, peak = tracemalloc.get_traced_memory()
        peak_mem = max(peak_mem, peak)

        if has_cuda:
            cuda_peak_mb = max(cuda_peak_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)

        if frame_id % 5 == 0:
            mem_result.add_snapshot(frame_id, cur / 1024 / 1024, peak_mem / 1024 / 1024)

    cap.release()
    elapsed = time.perf_counter() - start
    final_cur = tracemalloc.get_traced_memory()[0] / 1024 / 1024
    peak_mb = peak_mem / 1024 / 1024
    mem_result.peak_mb = peak_mb
    mem_result.final_mb = final_cur
    tracemalloc.stop()

    with lock:
        results.append((mem_result, has_cuda, cuda_peak_mb))


# --- RTSP / config-based benchmark worker ---

class _BenchWorker:
    def __init__(self, camera_id: str, source: str, pipeline: Pipeline, stop_event: threading.Event, frame_skip: int, stats: CamStats):
        self.camera_id = camera_id
        self.source = source
        self.pipeline = pipeline
        self.stop_event = stop_event
        self.frame_skip = frame_skip
        self.stats = stats
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"bench-{camera_id}")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.pipeline.stop()

    def _run(self):
        cap = create_capture(self.source)
        if cap is None:
            logging.error("Cannot open %s from %s", self.camera_id, self.source)
            return

        skip_interval = self.frame_skip + 1 if self.frame_skip > 0 else 1
        frame_id = 0
        consecutive_failures = 0
        max_failures = 5

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logging.warning("Reconnecting %s...", self.camera_id)
                        cap.release()
                        cap = create_capture(self.source)
                        if cap is None:
                            time.sleep(2)
                            continue
                        consecutive_failures = 0
                    else:
                        time.sleep(0.5)
                    continue

                consecutive_failures = 0
                self.stats.frames_read += 1

                if frame_id % skip_interval == 0:
                    t0 = time.perf_counter()
                    self.pipeline.process_frame(frame, frame_id)
                    dt = time.perf_counter() - t0
                    self.stats.frames_inferred += 1
                    self.stats.infer_times.append(dt)
                frame_id += 1
        finally:
            cap.release()


def _make_pipeline_config(target_classes, interaction_classes, model_size="n", quantize=False) -> PipelineConfig:
    thresholds = Thresholds()
    llm = LLMConfig()
    target_classes = target_classes or []
    interaction_classes = interaction_classes or []
    all_classes = list(dict.fromkeys(target_classes + interaction_classes))
    model_path = f"yolo26{model_size}.pt"
    yolo = YOLOConfig(
        model_size=model_size,
        model_path=model_path,
        quantize=quantize,
        yolo_classes=all_classes if all_classes else None,
    )
    return PipelineConfig(
        target_classes=target_classes,
        interaction_classes=interaction_classes,
        thresholds=thresholds,
        yolo=yolo,
        llm=llm,
    )


# --- Config-based (YAML) benchmark ---

def _run_config_bench(app_config, runtime: float, frame_skip: int):
    cameras = [c for c in app_config.cameras if c.status == "active"]
    if not cameras:
        print("No active cameras found.")
        return

    stats_list: list[CamStats] = []
    workers: list[_BenchWorker] = []
    stop_event = threading.Event()

    for cam in cameras:
        config = _make_pipeline_config(
            cam.target_classes, cam.interaction_classes,
            model_size=cam.model_size, quantize=cam.quantize,
        )
        pipeline = Pipeline(config, cam.id)
        stats = CamStats(camera_id=cam.id)
        stats_list.append(stats)
        worker = _BenchWorker(cam.id, cam.source, pipeline, stop_event, frame_skip, stats)
        workers.append(worker)

    print(f"\n=== Config Benchmark (runtime={runtime:.1f}s, frame_skip={frame_skip}) ===")
    print(f"Cameras: {', '.join(c.id for c in cameras)}\n")

    for w in workers:
        w.start()

    time.sleep(runtime)
    stop_event.set()

    for w in workers:
        w.stop()

    total_frames_read = 0
    total_frames_inferred = 0
    all_infer_times: list[float] = []

    for s in stats_list:
        avg_infer = (sum(s.infer_times) / len(s.infer_times) * 1000 if s.infer_times else 0)
        infer_fps = s.frames_inferred / runtime if runtime > 0 else 0
        print(f"  {s.camera_id:15s}: {s.frames_read:5d} frames read, {s.frames_inferred:3d} inferred, {infer_fps:5.1f} infer/s, avg {avg_infer:6.1f}ms/infer")
        total_frames_read += s.frames_read
        total_frames_inferred += s.frames_inferred
        all_infer_times.extend(s.infer_times)

    overall_avg = (sum(all_infer_times) / len(all_infer_times) * 1000 if all_infer_times else 0)
    total_infer_fps = total_frames_inferred / runtime if runtime > 0 else 0
    print(f"\n  Total: {total_frames_read} frames read, {total_frames_inferred} inferred across {len(cameras)} cameras, {total_infer_fps:.1f} infer/s, avg {overall_avg:.1f}ms/infer\n")


# --- Concurrent file benchmark ---

def bench_concurrent(videos: list[str], mode: str, runtime: int, frame_skip: int, full_video: bool, count: int, compare_classes: list[str] | None = None):
    results: list[BenchResult] = []
    lock = threading.Lock()

    threads: list[threading.Thread] = []
    batch = videos[:count]

    for video_path in batch:
        if mode in ("detect", "both"):
            t1 = threading.Thread(
                target=_run_detect,
                args=(video_path, runtime, None, frame_skip, full_video, "80c", results, lock),
                name=f"bench-80c-{os.path.basename(video_path)}",
            )
            threads.append(t1)

            if compare_classes is not None:
                t2 = threading.Thread(
                    target=_run_detect,
                    args=(video_path, runtime, compare_classes, frame_skip, full_video, "sel", results, lock),
                    name=f"bench-sel-{os.path.basename(video_path)}",
                )
                threads.append(t2)

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


# --- Sequential file benchmark ---

def bench_sequential(videos: list[str], mode: str, runtime: int, frame_skip: int, full_video: bool, compare_classes: list[str] | None = None):
    results: list[BenchResult] = []

    for video_path in videos:
        if mode in ("detect", "both"):
            f1, p1, _, t1 = _run_detect_sequential(video_path, runtime, None, frame_skip, full_video)
            fps1 = p1 / t1 if t1 > 0 else 0
            results.append(BenchResult(video_path, "80c", f1, p1, t1, fps1))

            if compare_classes is not None:
                f2, p2, _, t2 = _run_detect_sequential(video_path, runtime, compare_classes, frame_skip, full_video)
                fps2 = p2 / t2 if t2 > 0 else 0
                results.append(BenchResult(video_path, "sel", f2, p2, t2, fps2))

        if mode in ("full", "both"):
            f3, p3, _, t3 = _run_full_sequential(video_path, runtime, frame_skip, full_video)
            fps3 = p3 / t3 if t3 > 0 else 0
            results.append(BenchResult(video_path, "full", f3, p3, t3, fps3))

    return results, videos


# --- Print helpers ---

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


def _print_compare_results(results: list[BenchResult], video_path: str):
    r80 = [r for r in results if r.video_path == video_path and r.label == "80c"][0]
    rsel = [r for r in results if r.video_path == video_path and r.label == "sel"][0]

    ratio_frames = rsel.processed / r80.processed if r80.processed > 0 else 0
    ratio_fps = rsel.fps / r80.fps if r80.fps > 0 else 0

    print(f"\n  Compare for {os.path.basename(video_path)}:")
    print(f"    80c: {r80.processed} frames ({r80.fps:.1f} fps)")
    print(f"    sel: {rsel.processed} frames ({rsel.fps:.1f} fps)")
    print(f"    Ratio (sel/80c): {ratio_frames:.2%} frames, {ratio_fps:.2%} fps")
    print(f"    Difference: {rsel.processed - r80.processed} frames")


def _print_memory_results(results: list, video_paths: list[str]):
    for mem_result, has_cuda, cuda_peak_mb in results:
        label = mem_result.label
        print(f"\n  [{label}]")
        if has_cuda:
            print(f"    GPU peak: {cuda_peak_mb:.1f} MB")
        print(f"    Peak memory (tracemalloc): {mem_result.peak_mb:.1f} MB")
        print(f"    Final memory: {mem_result.final_mb:.1f} MB")
        if mem_result.snapshots:
            print(f"    Snapshots (frame, current_MB, peak_MB):")
            for s in mem_result.snapshots:
                print(f"      frame {s[0]}: {s[1]:.1f} / {s[2]:.1f}")

    # Compare summary if both 80c and sel exist
    r80 = next((r for r in results if r[0].label == "80c"), None)
    rsel = next((r for r in results if r[0].label == "sel"), None)
    if r80 and rsel:
        p80, _, _ = r80
        ps, _, _ = rsel
        diff = ps.peak_mb - p80.peak_mb
        print(f"\n  Compare:")
        print(f"    80c peak: {p80.peak_mb:.1f} MB")
        print(f"    sel peak: {ps.peak_mb:.1f} MB")
        print(f"    Difference: {diff:+.1f} MB")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Unified benchmark")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path (e.g. configuration.yaml)")
    parser.add_argument("--video", type=str, default=None, help="Specific video path")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Directory to scan for videos")
    parser.add_argument("--mode", choices=["detect", "full", "both"], default="detect", help="Benchmark mode (default: detect)")
    parser.add_argument("--runtime", type=int, default=DEFAULT_RUNTIME, help=f"Runtime in seconds (default: {DEFAULT_RUNTIME})")
    parser.add_argument("--compare-classes", nargs="*", default=None, help="Compare classes. Use 'sel' for interaction classes. Omit for 80c only.")
    parser.add_argument("--frame-skip", type=int, default=None, help="Frame skip interval (default: from configuration.yaml / settings.py)")
    parser.add_argument("--count", type=int, default=None, help="Concurrent video count (default: auto)")
    parser.add_argument("--full-video", action="store_true", help="Process entire video")
    parser.add_argument("--model", type=str, default=None, help="YOLO model path")
    parser.add_argument("--memory", action="store_true", help="Memory benchmark (detect mode only)")

    args = parser.parse_args()

    frame_skip = args.frame_skip if args.frame_skip is not None else YOLOConfig().frame_skip
    compare_classes = SELECTED_CLASSES if args.compare_classes == ["sel"] else args.compare_classes

    if args.model:
        _MODEL_PATH = args.model

    # --- Config-based mode ---
    if args.config:
        app_config = load_config(args.config)
        cameras = [c for c in app_config.cameras if c.status == "active"]
        if not cameras:
            print("No active cameras found.")
            return

        is_rtsp = any(c.source.startswith("rtsp://") for c in cameras)

        if is_rtsp or args.memory:
            # RTSP or memory: use config-based full pipeline
            _run_config_bench(app_config, float(args.runtime), frame_skip)
        else:
            # Video file config: extract video paths and run file benchmark
            video_paths = [c.source for c in cameras]
            count = args.count if args.count is not None else len(video_paths)

            print("=" * 60)
            mode_str = f"Mode: {args.mode} | Config: {args.config}"
            skip_str = f"Skip: {frame_skip}"
            count_str = f"Concurrent: {count}"
            compare_str = f"Compare: {len(compare_classes)} classes" if compare_classes else "Compare: none"
            runtime_str = f"Full video" if args.full_video else f"Runtime: {args.runtime}s"
            print(f"{mode_str} | {skip_str} | {count_str} | {compare_str} | {runtime_str}")
            print("=" * 60)

            if count > 1:
                results, batch = bench_concurrent(video_paths, args.mode, args.runtime, frame_skip, args.full_video, count, compare_classes)
            else:
                results, batch = bench_sequential(video_paths, args.mode, args.runtime, frame_skip, args.full_video, compare_classes)

            _print_results(results, video_paths, args.full_video)

            if compare_classes and len(batch) == 1:
                _print_compare_results(results, video_paths[0])

    # --- File-based mode (video or data-dir) ---
    elif args.video or not args.config:
        videos = [args.video] if args.video else discover_videos(args.data_dir)
        if not videos:
            print("No videos found.")
            return

        count = args.count if args.count is not None else 1

        # Memory benchmark
        if args.memory:
            print("=" * 60)
            print(f"Memory Benchmark | Skip: {frame_skip} | Videos: {len(videos)}")
            print("=" * 60)

            for video_path in videos:
                mem_lock = threading.Lock()

                mem_results_80: list = []
                _run_memory(video_path, args.runtime, None, "80c", mem_results_80, mem_lock)

                if compare_classes is not None:
                    mem_results_sel: list = []
                    _run_memory(video_path, args.runtime, compare_classes, "sel", mem_results_sel, mem_lock)
                    all_mem = mem_results_80 + mem_results_sel
                else:
                    all_mem = mem_results_80

                _print_memory_results(all_mem, [video_path])

        # Standard benchmark
        else:
            print("=" * 60)
            mode_str = f"Mode: {args.mode}"
            skip_str = f"Skip: {frame_skip}"
            count_str = f"Concurrent: {count}"
            compare_str = f"Compare: {len(compare_classes)} classes" if compare_classes else "Compare: none"
            runtime_str = f"Full video" if args.full_video else f"Runtime: {args.runtime}s"
            print(f"{mode_str} | {skip_str} | {count_str} | {compare_str} | {runtime_str} | Videos: {len(videos)}")
            print("=" * 60)

            if count > 1:
                results, batch = bench_concurrent(videos, args.mode, args.runtime, frame_skip, args.full_video, count, compare_classes)
            else:
                results, batch = bench_sequential(videos, args.mode, args.runtime, frame_skip, args.full_video, compare_classes)

            _print_results(results, batch, args.full_video)

            if compare_classes and len(batch) == 1:
                _print_compare_results(results, batch[0])

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
