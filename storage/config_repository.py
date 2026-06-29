import json
import logging

from sqlalchemy import text

from storage.database import AppSetting, ConfigVersion

logger = logging.getLogger(__name__)


class ConfigRepository:
    """DB ↔ config 양방향 관리 (PostgreSQL 전용)."""

    def __init__(self, session_factory, db_url: str):
        self._session_factory = session_factory
        self._is_postgres = db_url.startswith("postgresql://")

    # ─────────────────────── 읽기 ────────────────────────

    def get_version(self) -> int | None:
        with self._session_factory() as session:
            row = session.get(ConfigVersion, 1)
            return row.version if row else None

    def get_full_config(self) -> dict:
        """prefix별 배치 조회 → flat config dict 반환."""
        result = {
            "mode": "",
            "llm": {},
            "cameras": [],
            "spaces": [],
        }

        with self._session_factory() as session:
            result["reconnect"] = {}

            # prefix별 조회 (INDEX SCAN)
            for prefix in ("mode", "llm", "reconnect"):
                rows = session.query(AppSetting).filter(
                    AppSetting.key_prefix == prefix
                ).all()

                if prefix == "mode":
                    row = rows[0] if rows else None
                    result["mode"] = row.value_text or "" if row else ""
                elif prefix == "reconnect":
                    # reconnect — 모두 numeric
                    for r in rows:
                        result[prefix][r.key] = float(r.value_number) if r.value_number is not None else None
                else:
                    # llm — mixed type
                    for r in rows:
                        val = self._resolve_value(r)
                        result[prefix][r.key] = val

            # cameras (JSON)
            cam_rows = session.query(AppSetting).filter(
                AppSetting.key_prefix == "cameras"
            ).all()
            for r in cam_rows:
                if r.value_text:
                    result["cameras"].append(json.loads(r.value_text))

            # spaces (JSON)
            sp_rows = session.query(AppSetting).filter(
                AppSetting.key_prefix == "spaces"
            ).all()
            for r in sp_rows:
                if r.value_text:
                    result["spaces"].append(json.loads(r.value_text))

        return result

    # ─────────────────────── 쓰기 ────────────────────────

    def patch_llm(self, updates: dict) -> None:
        """llm 개별 필드 업데이트 (mixed type)."""
        with self._session_factory() as session:
            for k, v in updates.items():
                row = session.query(AppSetting).filter(
                    AppSetting.key_prefix == "llm",
                    AppSetting.key == k,
                ).first()
                if row is None:
                    row = AppSetting(key=k, key_prefix="llm")
                    session.add(row)
                self._set_value(row, v)
            self._increment_version(session)

    def save_camera(self, camera_id: str, data: dict) -> None:
        """camera 전체 JSON 덮어쓰기."""
        with self._session_factory() as session:
            row = session.query(AppSetting).filter(
                AppSetting.key_prefix == "cameras",
                AppSetting.key == camera_id,
            ).first()
            if row is None:
                row = AppSetting(key=camera_id, key_prefix="cameras")
                session.add(row)
            row.value_text = json.dumps(data, ensure_ascii=False)
            self._increment_version(session)

    def remove_camera(self, camera_id: str) -> None:
        with self._session_factory() as session:
            session.query(AppSetting).filter(
                AppSetting.key_prefix == "cameras",
                AppSetting.key == camera_id,
            ).delete(synchronize_session="fetch")
            self._increment_version(session)

    def save_space(self, space_id: str, data: dict) -> None:
        """space 전체 JSON 덮어쓰기."""
        with self._session_factory() as session:
            row = session.query(AppSetting).filter(
                AppSetting.key_prefix == "spaces",
                AppSetting.key == space_id,
            ).first()
            if row is None:
                row = AppSetting(key=space_id, key_prefix="spaces")
                session.add(row)
            row.value_text = json.dumps(data, ensure_ascii=False)
            self._increment_version(session)

    def remove_space(self, space_id: str) -> None:
        with self._session_factory() as session:
            session.query(AppSetting).filter(
                AppSetting.key_prefix == "spaces",
                AppSetting.key == space_id,
            ).delete(synchronize_session="fetch")
            self._increment_version(session)

    def save_mode(self, mode: str) -> None:
        with self._session_factory() as session:
            row = session.query(AppSetting).filter(
                AppSetting.key_prefix == "mode",
                AppSetting.key == "mode",
            ).first()
            if row is None:
                row = AppSetting(key="mode", key_prefix="mode")
                session.add(row)
            row.value_text = mode
            self._increment_version(session)

    # ─────────────────────── 내부 ────────────────────────

    @staticmethod
    def _resolve_value(setting: AppSetting):
        if setting.value_bool is not None:
            return bool(setting.value_bool)
        if setting.value_number is not None:
            v = float(setting.value_number)
            return int(v) if v == int(v) else v
        return setting.value_text

    @staticmethod
    def _set_value(row, value):
        """타입에 따라 적절한 컬럼 설정."""
        if isinstance(value, bool):
            row.value_bool = value
            row.value_number = None
            row.value_text = None
        elif isinstance(value, (int, float)):
            row.value_number = float(value)
            row.value_bool = None
            row.value_text = None
        else:
            row.value_text = str(value)
            row.value_bool = None
            row.value_number = None

    def _increment_version(self, session) -> int:
        """버전 +1 + commit + NOTIFY."""
        ver_row = session.get(ConfigVersion, 1)
        if ver_row is None:
            ver_row = ConfigVersion(id=1, version=0)
            session.add(ver_row)
        ver_row.version += 1
        session.commit()

        # PostgreSQL 전용 NOTIFY
        if self._is_postgres:
            try:
                session.execute(text("SELECT pg_notify('config_changed', '')"))
                session.commit()
            except Exception as e:
                logger.warning("pg_notify failed: %s", e)

        return ver_row.version
