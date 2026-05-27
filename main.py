from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import signal
import sys
import time

import cv2

from config.config import PipelineConfig
from core.pipeline import Pipeline
from utils.video import create_capture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Shutdown signal received (%s)", signum)
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def run_live(config: PipelineConfig, camera_source: str):
    cap = create_capture(camera_source)
    if cap is None:
        logger.error("Cannot open camera %s", camera_source)
        return

    cam_id = camera_source.split("/")[-1] if "/" in camera_source else camera_source
    logger.info("Live mode started: %s", camera_source)
    pipeline = Pipeline(config, f"cam_{cam_id}")
    frame_id = 0
    consecutive_failures = 0
    max_failures = 5

    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                logger.warning("Failed to read frame from %s (attempt %d/%d)", camera_source, consecutive_failures, max_failures)
                if consecutive_failures >= max_failures:
                    logger.warning("Reconnecting to %s...", camera_source)
                    cap.release()
                    cap = create_capture(camera_source)
                    if cap is None:
                        logger.error("Cannot reconnect to %s, retrying in 5s...", camera_source)
                        time.sleep(5)
                        cap = create_capture(camera_source)
                        if cap is None:
                            logger.error("Still cannot reconnect to %s", camera_source)
                            time.sleep(5)
                            consecutive_failures = 0
                            continue
                    consecutive_failures = 0
                else:
                    time.sleep(0.5)
                continue

            consecutive_failures = 0
            result = pipeline.process_frame(frame, frame_id)
            if result:
                logger.info("[cam_%s] %s", cam_id, result)

            frame_id += 1
    finally:
        cap.release()
        logger.info("Camera %s released", camera_source)


def run_video(config: PipelineConfig, video_path: str):
    cap = create_capture(video_path)
    if cap is None:
        logger.error("Cannot open video %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    logger.info("Video mode: %s (fps=%.1f)", video_path, fps)
    pipeline = Pipeline(config, "video")
    frame_id = 0

    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video ended")
                break

            result = pipeline.process_frame(frame, frame_id)
            if result:
                logger.info("[video] %s", result)

            frame_id += 1
            time.sleep(1.0 / fps)
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Tracking-Cano")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", type=str, help="Camera source (RTSP URL, e.g. rtsp://admin:pass@192.168.0.100:554/stream)")
    group.add_argument("--video", type=str, help="Offline video file path")
    parser.add_argument("--target-classes", nargs="+", default=["cat"], help="Target COCO classes")
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    config = PipelineConfig(
        target_classes=args.target_classes,
    )
    config.yolo.model_path = args.model
    config.yolo.conf_threshold = args.conf

    if args.live:
        run_live(config, args.live)
    else:
        run_video(config, args.video)


if __name__ == "__main__":
    main()
