import logging
import os
import threading
import time

from core.config_applier import apply_config_changes, camera_values_differ
from core.config_manager import diff_configs, load_from_db

logger = logging.getLogger(__name__)


class ConfigListener:
    """PostgreSQL LISTEN/NOTIFY 기반 설정 변경 감지 + polling 폴백."""

    def __init__(self, repo, orchestrator, space_logger, db_url: str):
        self._repo = repo
        self._orchestrator = orchestrator
        self._space_logger = space_logger
        self._db_url = db_url
        self._last_version = repo.get_version() or 0
        self._stop_event = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="config-listener")
        self._thread.start()
        logger.info("ConfigListener started (LISTEN/NOTIFY + polling)")

    def stop(self):
        self._stop_event.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)
        logger.info("ConfigListener stopped")

    def _run(self):
        import select
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        conn = None
        poll_interval = 30.0
        next_poll = time.monotonic() + poll_interval

        while not self._stop_event.is_set():
            try:
                if conn is None or conn.closed:
                    conn = psycopg2.connect(self._db_url)
                    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                    cur = conn.cursor()
                    cur.execute("LISTEN config_changed")
                    logger.info("ConfigListener connected, LISTENing on 'config_changed'")

                now = time.monotonic()
                remaining = max(0.1, next_poll - now)

                # psycopg2 표준 LISTEN/NOTIFY: select + poll 패턴
                try:
                    ready, _, _ = select.select([conn], [], [], remaining)
                except (ValueError, OSError):
                    ready = []

                if ready:
                    conn.poll()
                    for notif in conn.notifies:
                        logger.debug("Received NOTIFY: %s", notif.payload)
                        self._handle_change()

                # polling 폴백 체크
                if time.monotonic() >= next_poll:
                    current_version = self._repo.get_version()
                    if current_version is not None and current_version != self._last_version:
                        logger.info(
                            "Config version changed (polling): %d → %d",
                            self._last_version, current_version,
                        )
                        self._handle_change()
                    next_poll = time.monotonic() + poll_interval

            except Exception as e:
                logger.error("ConfigListener error: %s — reconnecting in 5s", e)
                if conn and not conn.closed:
                    try:
                        conn.close()
                    except Exception:
                        pass
                conn = None
                self._stop_event.wait(5)

        # Cleanup
        if conn and not conn.closed:
            conn.close()

    def _handle_change(self):
        """DB에서 새 config 로드 → diff 계산 → 적용."""
        try:
            new_cfg = load_from_db(self._repo, llm_key=os.environ.get("LLM_KEY", ""))
            old_cfg = self._orchestrator.app_config
            diff = diff_configs(old_cfg, new_cfg)

            # 구조 변경 (camera/space 추가·삭제) 감지
            if not diff.is_empty:
                logger.info("Structural config change detected: %s", diff)
                apply_config_changes(self._orchestrator, self._space_logger, new_cfg)
            else:
                # 값 변경 (thresholds/yolo/llm/camera 내부) 감지
                if old_cfg.thresholds != new_cfg.thresholds or \
                   old_cfg.yolo != new_cfg.yolo or \
                   old_cfg.llm != new_cfg.llm or \
                   old_cfg.mode != new_cfg.mode:
                    logger.info("Global config value change detected")
                    apply_config_changes(self._orchestrator, self._space_logger, new_cfg)

                # camera 개별 값 변경 감지
                for cam in new_cfg.cameras:
                    old_cam = next((c for c in old_cfg.cameras if c.id == cam.id), None)
                    if old_cam and camera_values_differ(old_cam, cam):
                        logger.info("Camera %s value change detected", cam.id)
                        apply_config_changes(self._orchestrator, self._space_logger, new_cfg)
                        break

            self._last_version = self._repo.get_version() or self._last_version

        except Exception:
            logger.exception("Failed to apply config changes")
