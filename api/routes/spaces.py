"""Space CRUD endpoints — modifies configuration.yaml (dev) or DB (production)."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from api.auth import verify_token
from api.models import SpaceCreate, SpaceResponse, SpaceUpdate
from core.config_applier import apply_config_changes
from core.yaml_writer import (
    read_yaml,
    remove_yaml_space,
    update_yaml_space,
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


@router.get("/", response_model=List[SpaceResponse])
async def list_spaces(_: str = Depends(verify_token)) -> List[SpaceResponse]:
    orch = _get_orchestrator()

    spaces_raw: List[Dict] = []
    if orch is not None:
        for space in orch.spaces:
            spaces_raw.append({
                "id": space.id,
                "name": space.name,
                "cameras": space.camera_ids,
                "state": "detecting" if orch._vision_detector else None,
            })
    else:
        data = read_yaml()
        for sp in data.get("spaces", []):
            spaces_raw.append({
                "id": sp["id"],
                "name": sp.get("name", sp["id"]),
                "cameras": sp.get("cameras", []),
                "state": None,
            })

    return [SpaceResponse(**s) for s in spaces_raw]


@router.get("/{space_id}", response_model=SpaceResponse)
async def get_space(space_id: str, _: str = Depends(verify_token)) -> SpaceResponse:
    orch = _get_orchestrator()
    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    space = next((s for s in orch.spaces if s.id == space_id), None)
    if not space:
        raise HTTPException(status_code=404, detail=f"Space {space_id} not found")

    return SpaceResponse(
        id=space.id,
        name=space.name,
        cameras=space.camera_ids,
        state="detecting" if orch._vision_detector else None,
    )


@router.post("/", response_model=SpaceResponse)
async def create_space(body: SpaceCreate, _: str = Depends(verify_token)) -> SpaceResponse:
    space_dict = body.model_dump(exclude_none=True)

    if _is_db_mode():
        orch = _get_orchestrator()
        repo = orch._config_repo
        data = {
            "id": body.id,
            "name": body.name or body.id,
            "cameras": body.cameras or [],
            "llm_system_prompt": getattr(body, "llm_system_prompt", None),
        }
        repo.save_space(body.id, data)
        from core.config_manager import load_from_db
        new_cfg = load_from_db(repo, llm_key=orch.app_config.llm.api_key)
        apply_config_changes(orch, orch.space_logger, new_cfg)
    else:
        update_yaml_space(space_dict)

    return SpaceResponse(**space_dict, state=None)


@router.put("/{space_id}", response_model=SpaceResponse)
async def update_space(
    space_id: str,
    body: SpaceUpdate,
    _: str = Depends(verify_token),
) -> SpaceResponse:
    orch = _get_orchestrator()

    if _is_db_mode():
        # Read current from orchestrator config
        space = next((s for s in orch.spaces if s.id == space_id), None)
        if not space:
            raise HTTPException(status_code=404, detail=f"Space {space_id} not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            return SpaceResponse(
                id=space.id, name=space.name, cameras=space.camera_ids, state=None,
            )

        sp_dict = {
            "id": space_id,
            "name": space.name,
            "cameras": space.camera_ids,
            "llm_system_prompt": getattr(space, "llm_system_prompt", None),
        }
        sp_dict.update(updates)

        repo = orch._config_repo
        repo.save_space(space_id, sp_dict)
        from core.config_manager import load_from_db
        new_cfg = load_from_db(repo, llm_key=orch.app_config.llm.api_key)
        apply_config_changes(orch, orch.space_logger, new_cfg)

        return SpaceResponse(**sp_dict, state=None)
    else:
        data = read_yaml()
        spaces_raw = data.get("spaces", [])
        sp_dict = next((s for s in spaces_raw if isinstance(s, dict) and s["id"] == space_id), None)
        if not sp_dict:
            raise HTTPException(status_code=404, detail=f"Space {space_id} not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            return SpaceResponse(**sp_dict, state=None)

        sp_dict.update(updates)
        update_yaml_space(sp_dict)
        return SpaceResponse(**sp_dict, state=None)


@router.delete("/{space_id}")
async def delete_space(space_id: str, _: str = Depends(verify_token)) -> Dict[str, str]:
    if _is_db_mode():
        orch = _get_orchestrator()
        repo = orch._config_repo
        repo.remove_space(space_id)
    else:
        remove_yaml_space(space_id)

    return {"status": "deleted", "space_id": space_id}


@router.post("/{space_id}/flush")
async def flush_space(space_id: str, _: str = Depends(verify_token)) -> Dict[str, Any]:
    from api.server import _space_logger
    orch = _get_orchestrator()
    if not orch or not _space_logger:
        raise HTTPException(status_code=503, detail="Space logger not available")

    space = next((s for s in orch.spaces if s.id == space_id), None)
    if not space:
        raise HTTPException(status_code=404, detail=f"Space {space_id} not found")

    orch.request_space_snapshot(space_id)
    return {"status": "snapshot_triggered", "space_id": space_id}
