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

from sqlalchemy import text

from settings import MinIOConfig
from core.config_applier import apply_config_changes
from core.config_manager import AppConfig, ConfigWatcher, load_config, load_from_db
from core.orchestrator import Orchestrator
from nlp.logger import SpaceLogger
from storage.database import init_db
from storage.repository import LogRepository

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


def run_multi(config_path: str, repo: LogRepository | None = None, app_config: AppConfig | None = None, config_repo=None):
    from api.event_bus import EventBus
    from api.server import start_api

    event_bus = EventBus()
    minio_cfg = MinIOConfig.from_env()
    space_logger = SpaceLogger(app_config.llm, repo=repo, event_bus=event_bus, minio_config=minio_cfg, log_config=app_config.log)
    orchestrator = Orchestrator(
        app_config, space_logger, repo=repo, event_bus=event_bus
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
        space_logger.stop()


def _on_config_change(orchestrator: Orchestrator, space_logger: SpaceLogger, new_config, diff):
    apply_config_changes(orchestrator, space_logger, new_config)


def main():
    parser = argparse.ArgumentParser(description="Tracking-Cano")
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

    if repo:
        logger.info("Log DB: %s", log_config.db_url)
    else:
        logger.warning("No DB available — log entries will not be persisted")

    if not app_config.cameras:
        logger.error("No cameras configured — define at least one camera in the config")
        sys.exit(1)

    run_multi(args.config, repo=repo, app_config=app_config, config_repo=config_repo)


if __name__ == "__main__":
    main()
