"""Camera CRUD endpoints — modifies configuration.yaml (dev) or DB (production)."""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from api.auth import verify_token
from api.models import CameraCreate, CameraResponse, CameraUpdate
from core.config_applier import apply_config_changes
from core.yaml_writer import (
    read_yaml,
    remove_yaml_camera,
    update_yaml_camera,
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


@router.get("/", response_model=List[CameraResponse])
async def list_cameras(_: str = Depends(verify_token)) -> List[CameraResponse]:
    orch = _get_orchestrator()
    if orch is None:
        data = read_yaml()
        cameras_raw = data.get("cameras", [])
        return [CameraResponse(**c) for c in cameras_raw]

    resp = []
    for cam in orch.app_config.cameras:
        worker_state = "stopped"
        if cam.id in orch._workers:
            worker_state = "running"
        elif cam.id in orch._collectors:
            worker_state = "collector"

        resp.append(CameraResponse(
            id=cam.id,
            source=cam.source,
            status=cam.status,
            target_classes=cam.target_classes,
            interaction_classes=cam.interaction_classes,
            model_size=cam.model_size,
            frame_skip=cam.frame_skip,
            worker_state=worker_state,
        ))
    return resp


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, _: str = Depends(verify_token)) -> CameraResponse:
    orch = _get_orchestrator()
    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    cam = next((c for c in orch.app_config.cameras if c.id == camera_id), None)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    worker_state = "stopped"
    if camera_id in orch._workers:
        worker_state = "running"
    elif camera_id in orch._collectors:
        worker_state = "collector"

    return CameraResponse(
        id=cam.id,
        source=cam.source,
        status=cam.status,
        target_classes=cam.target_classes,
        interaction_classes=cam.interaction_classes,
        model_size=cam.model_size,
        frame_skip=cam.frame_skip,
        worker_state=worker_state,
    )


@router.post("/", response_model=CameraResponse)
async def create_camera(body: CameraCreate, _: str = Depends(verify_token)) -> CameraResponse:
    orch = _get_orchestrator()
    camera_dict = body.model_dump(exclude_none=True)

    if _is_db_mode():
        repo = orch._config_repo
        repo.save_camera(body.id, camera_dict)
        from core.config_manager import load_from_db
        new_cfg = load_from_db(repo, llm_key=orch.app_config.llm.api_key)
        apply_config_changes(orch, orch.space_logger, new_cfg)
    else:
        update_yaml_camera(camera_dict)

    return CameraResponse(**camera_dict, worker_state="pending")


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str,
    body: CameraUpdate,
    _: str = Depends(verify_token),
) -> CameraResponse:
    orch = _get_orchestrator()

    if _is_db_mode():
        # Read current from orchestrator config
        cam = next((c for c in orch.app_config.cameras if c.id == camera_id), None)
        if not cam:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            return CameraResponse(
                id=cam.id, source=cam.source, status=cam.status,
                target_classes=cam.target_classes, interaction_classes=cam.interaction_classes,
                model_size=cam.model_size, frame_skip=cam.frame_skip, worker_state="unknown",
            )

        # Merge updates into camera dict
        cam_dict = {
            "id": cam.id, "source": cam.source, "status": cam.status,
            "target_classes": cam.target_classes, "interaction_classes": cam.interaction_classes,
            "model_size": cam.model_size, "frame_skip": cam.frame_skip,
            "quantize": cam.quantize, "llm_system_prompt": cam.llm_system_prompt,
        }
        cam_dict.update(updates)

        repo = orch._config_repo
        repo.save_camera(camera_id, cam_dict)
        from core.config_manager import load_from_db
        new_cfg = load_from_db(repo, llm_key=orch.app_config.llm.api_key)
        apply_config_changes(orch, orch.space_logger, new_cfg)

        return CameraResponse(**cam_dict, worker_state="restarting")
    else:
        data = read_yaml()
        cameras_raw = data.get("cameras", [])
        cam_dict = next((c for c in cameras_raw if isinstance(c, dict) and c["id"] == camera_id), None)
        if not cam_dict:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            return CameraResponse(**cam_dict, worker_state="unknown")

        cam_dict.update(updates)
        update_yaml_camera(cam_dict)

        from core.config_manager import load_config
        new_cfg = load_config()
        apply_config_changes(orch, orch.space_logger, new_cfg)

        return CameraResponse(**cam_dict, worker_state="restarting")


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str, _: str = Depends(verify_token)) -> Dict[str, str]:
    orch = _get_orchestrator()

    if orch:
        orch.remove_camera(camera_id)

    if _is_db_mode():
        repo = orch._config_repo
        repo.remove_camera(camera_id)
    else:
        remove_yaml_camera(camera_id)

    return {"status": "deleted", "camera_id": camera_id}


@router.post("/{camera_id}/restart")
async def restart_camera(camera_id: str, _: str = Depends(verify_token)) -> Dict[str, str]:
    orch = _get_orchestrator()
    if not orch:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    cam_obj = next((c for c in orch.app_config.cameras if c.id == camera_id), None)
    if not cam_obj:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    orch.remove_camera(camera_id)
    orch.add_camera(cam_obj)
    return {"status": "restarted", "camera_id": camera_id}
