"""Config read/update endpoints — modifies configuration.yaml (dev) or DB (production)."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from api.auth import verify_token
from api.models import LLMUpdate
from core.yaml_writer import (
    read_yaml,
    update_yaml_config_section,
)

router = APIRouter()


def _get_orchestrator():
    from api.server import _orchestrator as o
    return o


def _is_db_mode():
    orch = _get_orchestrator()
    if orch is None:
        return False
    return getattr(orch, "_config_repo", None) is not None


@router.get("/")
async def get_config(_: str = Depends(verify_token)) -> Dict[str, Any]:
    if _is_db_mode():
        repo = _get_orchestrator()._config_repo
        raw = repo.get_full_config()
        # Flatten to YAML-like structure for API response
        return {
            "mode": raw["mode"],
            "llm": {**raw["llm"]},  # exclude api_key from response
            "cameras": raw["cameras"],
            "spaces": raw["spaces"],
        }
    return read_yaml()



@router.put("/llm")
async def update_llm(body: LLMUpdate, _: str = Depends(verify_token)) -> Dict[str, Any]:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    orch = _get_orchestrator()
    if _is_db_mode():
        repo = orch._config_repo
        repo.patch_llm(updates)
        from core.config_applier import apply_config_changes
        from core.config_manager import load_from_db
        new_cfg = load_from_db(repo, llm_key=orch.app_config.llm.api_key)
        apply_config_changes(orch, orch.space_logger, new_cfg)
    else:
        update_yaml_config_section("llm", updates)

    return {"status": "ok"}
