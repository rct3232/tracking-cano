from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import os
import signal
import sys
import time

import cv2

from config.config import PipelineConfig
from core.config_manager import ConfigWatcher
from core.pipeline import Pipeline
from core.orchestrator import Orchestrator
from nlp.logger import SpaceLogger
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
    mode = os.environ.get("MODE", "cv_pipeline")
    logger.info("Live mode started: %s (mode=%s)", camera_source, mode)
    pipeline = Pipeline(config, f"cam_{cam_id}")
    frame_id = 0
    consecutive_failures = 0
    max_failures = 5

    if mode == "llm_vision":
        from collections import deque
        import base64 as b64mod
        buffer = deque(maxlen=config.llm.snapshot_count)
        last_batch_time = time.monotonic()
        snapshot_interval = config.llm.snapshot_interval
    else:
        buffer = None
        snapshot_interval = 0

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

            if mode == "llm_vision":
                import cv2 as cv2_lib
                now = time.monotonic()
                if now - last_batch_time >= snapshot_interval:
                    _, buf = cv2_lib.imencode(".jpg", frame, [cv2_lib.IMWRITE_JPEG_QUALITY, config.llm.vision_quality])
                    image_b64 = b64mod.b64encode(buf).decode("utf-8")
                    buffer.append(image_b64)
                    logger.debug("[vision:%s] buffer_size=%d", cam_id, len(buffer))
                    if len(buffer) >= config.llm.snapshot_count:
                        batch = list(buffer)
                        pipeline.nlp_logger.vision_log(
                            batch, f"cam_{cam_id}",
                            config.llm_system_prompt, config.target_classes,
                        )
                        last_batch_time = time.monotonic()
                        buffer.popleft()
                elif len(buffer) > 0:
                    result = pipeline.process_frame(frame, frame_id)
                    if result:
                        logger.info("[cam_%s] %s", cam_id, result)

            frame_id += 1
    finally:
        pipeline.stop()
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
        pipeline.stop()
        cap.release()


def run_multi(config_path: str, model_path: str | None = None):
    from core.config_manager import load_config
    app_config = load_config(config_path)
    max_cameras = max((len(s.camera_ids) for s in app_config.spaces), default=1)
    space_logger = SpaceLogger(PipelineConfig().llm, flush_threshold=max_cameras)
    orchestrator = Orchestrator(app_config, space_logger, default_model_path=model_path)
    orchestrator.start()
    watcher = ConfigWatcher(config_path, lambda new_cfg, diff: _on_config_change(orchestrator, space_logger, new_cfg, diff))
    watcher.start()

    flush_interval = 10.0
    last_flush = 0.0
    try:
        while _running and not orchestrator.all_finished:
            now = time.time()
            if now - last_flush >= flush_interval:
                orchestrator.flush_spaces()
                last_flush = now
            time.sleep(1)
        if orchestrator.all_finished:
            logger.info("All cameras finished processing")
    finally:
        orchestrator.flush_spaces()
        watcher.stop()
        orchestrator.stop()


def _on_config_change(orchestrator: Orchestrator, space_logger: SpaceLogger, new_config, diff):
    for cam_id in diff.reassigned_cameras:
        old_space, new_space = diff.reassigned_cameras[cam_id]
        logger.info("Camera %s reassigned: %s → %s", cam_id, old_space, new_space)
        orchestrator.reassign_camera(cam_id, old_space, new_space)
    for cam_id in diff.added_cameras:
        cam = next((c for c in new_config.cameras if c.id == cam_id), None)
        if cam:
            orchestrator.add_camera(cam)
    for cam_id in diff.removed_cameras:
        orchestrator.remove_camera(cam_id)
    for space_id in diff.added_spaces:
        space = next((s for s in new_config.spaces if s.id == space_id), None)
        if space:
            logger.info("Space added: %s (cameras: %d)", space_id, len(space.camera_ids))
            space_logger.set_camera_count(space_id, len(space.camera_ids))
    for space_id in diff.removed_spaces:
        text = space_logger.flush(space_id, space_id)
        if text:
            logger.info("[%s] (removed) %s", space_id, text)


def main():
    parser = argparse.ArgumentParser(description="Tracking-Cano")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", nargs="?", const="", default=None,
                       help="Camera source. No arg = multi-camera from config file. With arg = single camera (RTSP URL)")
    group.add_argument("--video", type=str, help="Offline video file path")
    parser.add_argument("--target-classes", nargs="+", default=["cat"], help="Target COCO classes")
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--config", default="config/spaces.yaml", help="Config file path (multi-camera mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")
    args = parser.parse_args()

    if args.verbose or os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
        logging.getLogger().setLevel(logging.DEBUG)

    mode = os.environ.get("MODE", "cv_pipeline")
    logger.info("Running in mode: %s", mode)

    config = PipelineConfig(
        target_classes=args.target_classes,
        interaction_classes=["couch", "chair", "dining table", "tv", "bed"],
    )
    config.yolo.model_path = args.model
    config.yolo.conf_threshold = args.conf

    if args.live == "":
        run_multi(args.config, model_path=args.model)
    elif args.live:
        run_live(config, args.live)
    else:
        run_video(config, args.video)


if __name__ == "__main__":
    main()
