from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from sqlalchemy import text

from settings import MinIOConfig, PipelineConfig, ReconnectConfig, YOLOConfig
from core.config_applier import apply_config_changes
from core.config_manager import AppConfig, ConfigWatcher, load_config, load_from_db
from core.pipeline import Pipeline
from core.orchestrator import Orchestrator
from nlp.logger import SpaceLogger
from storage.database import init_db
from storage.repository import LogRepository
from utils.video import create_capture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def setup_logging(log_dir: str):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    console_log = log_path / f"console_{datetime.now().strftime('%Y%m%d')}.log"
    handler = logging.handlers.RotatingFileHandler(
        console_log, maxBytes=10 * 1024 * 1024, backupCount=7,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(handler)
    logger.info("Console log: %s", console_log)

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Shutdown signal received (%s)", signum)
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def run_live(config: PipelineConfig, camera_source: str):
    rc = ReconnectConfig()
    backoff_delay = rc.base_delay

    cap = create_capture(camera_source)
    is_stream = isinstance(camera_source, str) and camera_source.startswith(("rtsp://", "http://", "https://"))
    while cap is None and is_stream:
        logger.warning("Cannot open %s, retrying in %.1fs...", camera_source, backoff_delay)
        time.sleep(backoff_delay)
        try:
            cap = create_capture(camera_source)
        except Exception:
            logger.exception("create_capture failed during initial connect")
            cap = None
        if cap is None:
            backoff_delay = min(backoff_delay * 2, rc.max_delay)

    if cap is None:
        logger.error("Cannot open camera %s", camera_source)
        return

    cam_id = camera_source.split("/")[-1] if "/" in camera_source else camera_source
    logger.info("Live mode started: %s (mode=%s)", camera_source, config.mode if hasattr(config, 'mode') else "cv_pipeline")
    pipeline = Pipeline(config, f"cam_{cam_id}")
    frame_id = 0
    consecutive_failures = 0
    max_failures = rc.max_failures

    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                logger.warning("Failed to read frame from %s (attempt %d/%d)", camera_source, consecutive_failures, max_failures)
                if consecutive_failures >= max_failures:
                    logger.warning("Reconnecting to %s... (backoff=%.1fs)", camera_source, backoff_delay)
                    cap.release()
                    time.sleep(backoff_delay)
                    try:
                        cap = create_capture(camera_source)
                    except Exception:
                        logger.exception("create_capture failed during reconnect")
                        cap = None
                    if cap is None:
                        logger.error("Cannot reconnect to %s", camera_source)
                        time.sleep(rc.reconnect_backoff)
                        backoff_delay = min(backoff_delay * 2, rc.max_delay)
                        consecutive_failures = 0
                        continue
                    consecutive_failures = 0
                    backoff_delay = rc.base_delay
                else:
                    time.sleep(rc.read_backoff)
                continue

            consecutive_failures = 0

            detect, _ = pipeline.process_frame(frame, frame_id)
            if detect.target_present:
                logger.info("[cam_%s] target_present=true class=%s bbox=%s", cam_id, detect.class_name, detect.target_coordinate)
            else:
                logger.debug("[cam_%s] target_present=false", cam_id)

            frame_id += 1
    finally:
        pipeline.stop()
        cap.release()
        logger.info("Camera %s released", camera_source)


def run_video(config: PipelineConfig, video_path: str, repo: LogRepository | None = None):
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

            detect, _ = pipeline.process_frame(frame, frame_id)
            if detect.target_present:
                logger.info("[video] target_present=true class=%s bbox=%s", detect.class_name, detect.target_coordinate)
            else:
                logger.debug("[video] target_present=false")

            frame_id += 1
            time.sleep(1.0 / fps)
    finally:
        pipeline.stop()
        cap.release()


def run_multi(config_path: str, model_path: str | None = None, repo: LogRepository | None = None, app_config: AppConfig | None = None, config_repo=None):
    from api.event_bus import EventBus
    from api.server import start_api

    event_bus = EventBus()
    minio_cfg = MinIOConfig.from_env()
    space_logger = SpaceLogger(app_config.llm, repo=repo, event_bus=event_bus, minio_config=minio_cfg)
    orchestrator = Orchestrator(
        app_config, space_logger, default_model_path=model_path, repo=repo, event_bus=event_bus
    )
    orchestrator._config_repo = config_repo

    start_api(orchestrator=orchestrator, space_logger=space_logger, repo=repo, event_bus=event_bus)
    orchestrator.start()

    if config_repo is not None:
        from core.config_listener import ConfigListener
        listener = ConfigListener(config_repo, orchestrator, space_logger, os.environ.get("DATABASE_URL", ""))
        listener.start()
        config_watcher_stop = lambda: listener.stop()
    else:
        watcher = ConfigWatcher(config_path, lambda new_cfg, diff: _on_config_change(orchestrator, space_logger, new_cfg, diff))
        watcher.start()
        config_watcher_stop = lambda: watcher.stop()

    all_finished_since: float | None = None
    try:
        while _running:
            if orchestrator.all_finished:
                if all_finished_since is None:
                    all_finished_since = time.time()
                elif time.time() - all_finished_since > 3.0:
                    logger.info("All workers finished, exiting")
                    break
                time.sleep(1)
                continue
            all_finished_since = None
            time.sleep(1)
    finally:
        config_watcher_stop()
        orchestrator.stop()


def _on_config_change(orchestrator: Orchestrator, space_logger: SpaceLogger, new_config, diff):
    apply_config_changes(orchestrator, space_logger, new_config)


def main():
    parser = argparse.ArgumentParser(description="Tracking-Cano")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", nargs="?", const="", default=None,
                       help="Camera source. No arg = multi-camera from config file. With arg = single camera (RTSP URL)")
    group.add_argument("--video", type=str, help="Offline video file path")
    parser.add_argument("--target-classes", nargs="+", help="Target COCO classes (default: from config)")
    parser.add_argument("--model", help="YOLO model path (override)")
    parser.add_argument("--config", default="configuration.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")
    args = parser.parse_args()

    if args.verbose or os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
        logging.getLogger().setLevel(logging.DEBUG)

    db_url = os.environ.get("DATABASE_URL", "")
    is_postgres = db_url.startswith("postgresql://")

    if is_postgres:
        engine, Session = init_db(db_url)
        if not engine:
            logger.error("DB connection failed, aborting")
            sys.exit(1)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error("DB connection check failed: %s", e)
            sys.exit(1)

        from storage.config_repository import ConfigRepository
        config_repo = ConfigRepository(Session, db_url)
        app_config = load_from_db(config_repo, llm_key=os.environ.get("LLM_KEY", ""))
        repo = LogRepository(Session) if Session else None
    else:
        app_config = load_config(args.config)
        config_repo = None
        engine, Session = init_db(app_config.log.db_url)
        repo = LogRepository(Session) if Session else None

    log_config = app_config.log
    setup_logging(log_config.log_dir)
    logger.info("Running in mode: %s", app_config.mode)

    if repo:
        logger.info("Log DB: %s", log_config.db_url)
    else:
        logger.warning("No DB available — log entries will not be persisted")

    if args.live == "":
        run_multi(args.config, model_path=args.model, repo=repo, app_config=app_config, config_repo=config_repo)
    elif args.live:
        target_classes = args.target_classes or ["cat"]
        config = PipelineConfig(
            target_classes=target_classes,
            interaction_classes=["couch", "chair", "dining table", "tv", "bed"],
            thresholds=app_config.thresholds,
            yolo=YOLOConfig(
                conf_threshold=app_config.yolo.conf_threshold,
                iou_threshold=app_config.yolo.iou_threshold,
                tile_enabled=app_config.yolo.tile_enabled,
                tile_grid_x=app_config.yolo.tile_grid_x,
                tile_grid_y=app_config.yolo.tile_grid_y,
                tile_overlap=app_config.yolo.tile_overlap,
                model_path=args.model or app_config.yolo.model_path or "yolo26s.pt",
            ),
            llm=app_config.llm,
        )
        run_live(config, args.live)
    else:
        target_classes = args.target_classes or ["cat"]
        config = PipelineConfig(
            target_classes=target_classes,
            interaction_classes=["couch", "chair", "dining table", "tv", "bed"],
            thresholds=app_config.thresholds,
            yolo=YOLOConfig(
                conf_threshold=app_config.yolo.conf_threshold,
                iou_threshold=app_config.yolo.iou_threshold,
                tile_enabled=app_config.yolo.tile_enabled,
                tile_grid_x=app_config.yolo.tile_grid_x,
                tile_grid_y=app_config.yolo.tile_grid_y,
                tile_overlap=app_config.yolo.tile_overlap,
                model_path=args.model or app_config.yolo.model_path or "yolo26s.pt",
            ),
            llm=app_config.llm,
        )
        run_video(config, args.video, repo=repo)


if __name__ == "__main__":
    main()
