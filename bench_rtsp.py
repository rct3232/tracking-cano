#!/usr/bin/env python3
import argparse
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2

from config.config import LLMConfig, PipelineConfig, Thresholds, YOLOConfig
from core.config_manager import load_config
from core.pipeline import Pipeline
from nlp.logger import SpaceLogger
from utils.video import create_capture

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class _CamStats:
    camera_id: str
    frames_read: int = 0
    frames_inferred: int = 0
    infer_times: List[float] = field(default_factory=list)


class _BenchWorker:
    def __init__(
        self,
        camera_id: str,
        source: str,
        pipeline: Pipeline,
        stop_event: threading.Event,
        frame_skip: int,
        stats: _CamStats,
    ):
        self.camera_id = camera_id
        self.source = source
        self.pipeline = pipeline
        self.stop_event = stop_event
        self.frame_skip = frame_skip
        self.stats = stats
        self.thread = threading.Thread(
            target=self._run, daemon=True, name=f"bench-{camera_id}"
        )

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.pipeline.stop()

    def _run(self):
        cap = create_capture(self.source)
        if cap is None:
            logger.error("Cannot open %s from %s", self.camera_id, self.source)
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
                        logger.warning("Reconnecting %s...", self.camera_id)
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


def _make_pipeline_config(
    target_classes: List[str],
    interaction_classes: List[str],
    model_size: str = "n",
    quantize: bool = False,
) -> PipelineConfig:
    thresholds = Thresholds()
    llm = LLMConfig()
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


def main():
    parser = argparse.ArgumentParser(description="RTSP stream processing benchmark")
    parser.add_argument(
        "--runtime", type=float, default=10.0, help="Runtime in seconds (default: 10)"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=15,
        help="Frame skip interval (default: 15)",
    )
    parser.add_argument(
        "--config",
        default="config/spaces.yaml",
        help="Config file path (default: config/spaces.yaml)",
    )
    args = parser.parse_args()

    app_config = load_config(args.config)
    cameras = [c for c in app_config.cameras if c.status == "active"]

    if not cameras:
        print("No active cameras found.")
        return

    max_cameras = max((len(s.camera_ids) for s in app_config.spaces), default=1)
    space_logger = SpaceLogger(PipelineConfig().llm, flush_threshold=max_cameras)

    stats_list: List[_CamStats] = []
    workers: List[_BenchWorker] = []

    for cam in cameras:
        config = _make_pipeline_config(
            cam.target_classes, cam.interaction_classes,
            model_size=cam.model_size, quantize=cam.quantize,
        )
        space_id = next(
            (s.id for s in app_config.spaces if cam.id in s.camera_ids), None
        )
        pipeline = Pipeline(config, cam.id, space_logger, space_id)
        stats = _CamStats(camera_id=cam.id)
        stats_list.append(stats)
        stop_event = threading.Event()
        worker = _BenchWorker(
            cam.id, cam.source, pipeline, stop_event, args.frame_skip, stats
        )
        workers.append(worker)

    print(f"\n=== RTSP Benchmark (runtime={args.runtime}s, frame_skip={args.frame_skip}) ===")
    print(f"Cameras: {', '.join(c.id for c in cameras)}\n")

    stop_event = threading.Event()
    for w in workers:
        w.start()

    time.sleep(args.runtime)
    stop_event.set()

    for w in workers:
        w.stop()

    total_frames_read = 0
    total_frames_inferred = 0
    all_infer_times: List[float] = []

    for s in stats_list:
        avg_infer = (
            sum(s.infer_times) / len(s.infer_times) * 1000
            if s.infer_times
            else 0
        )
        fps = s.frames_read / args.runtime if args.runtime > 0 else 0
        infer_fps = s.frames_inferred / args.runtime if args.runtime > 0 else 0

        print(
            f"  {s.camera_id:15s}: "
            f"{s.frames_read:5d} frames read, "
            f"{s.frames_inferred:3d} inferred, "
            f"{infer_fps:5.1f} infer/s, "
            f"avg {avg_infer:6.1f}ms/infer"
        )
        total_frames_read += s.frames_read
        total_frames_inferred += s.frames_inferred
        all_infer_times.extend(s.infer_times)

    overall_avg = (
        sum(all_infer_times) / len(all_infer_times) * 1000
        if all_infer_times
        else 0
    )
    total_infer_fps = total_frames_inferred / args.runtime if args.runtime > 0 else 0

    print(
        f"\n  Total: {total_frames_read} frames read, "
        f"{total_frames_inferred} inferred across {len(cameras)} cameras, "
        f"{total_infer_fps:.1f} infer/s, "
        f"avg {overall_avg:.1f}ms/infer"
    )
    print()


if __name__ == "__main__":
    main()
