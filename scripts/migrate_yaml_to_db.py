"""YAML → DB 마이그레이션 (수동 실행). 운영 DB에 기존 YAML 설정을 이관."""

from dotenv import load_dotenv
load_dotenv()

import json
import os

from core.config_manager import CameraConfig, SpaceConfig, load_config
from storage.database import init_db
from storage.config_repository import ConfigRepository


def migrate():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL env var is required")
        return

    engine, Session = init_db(db_url)
    if not engine or not Session:
        print("ERROR: DB connection failed")
        return

    repo = ConfigRepository(Session, db_url)
    app_config = load_config("configuration.yaml")

    # llm (api_key 제외 — env var 전용)
    for k, v in app_config.llm.__dict__.items():
        if k == "api_key":
            continue
        elif isinstance(v, bool):
            repo.patch_llm({k: v})
        elif isinstance(v, (int, float)):
            repo.patch_llm({k: v})
        else:
            repo.patch_llm({k: str(v)})
    print(f"  llm: {len(app_config.llm.__dict__) - 1} fields (api_key excluded)")

    # cameras → JSON 전체 저장
    for cam in app_config.cameras:
        data = {
            "id": cam.id,
            "source": cam.source,
            "status": cam.status,
            "target_classes": cam.target_classes,
            "llm_system_prompt": cam.llm_system_prompt,
        }
        repo.save_camera(cam.id, data)
    print(f"  cameras: {len(app_config.cameras)} entries")

    # spaces → JSON 전체 저장
    for sp in app_config.spaces:
        data = {
            "id": sp.id,
            "name": sp.name,
            "cameras": sp.camera_ids,
            "llm_system_prompt": getattr(sp, "llm_system_prompt", None),
        }
        repo.save_space(sp.id, data)
    print(f"  spaces: {len(app_config.spaces)} entries")

    version = repo.get_version()
    print(f"\nMigration complete. config_version={version}")


if __name__ == "__main__":
    migrate()
