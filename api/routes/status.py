"""Health and status endpoints."""

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

from api.auth import verify_token
from api.models import CameraResponse, SpaceResponse, SystemStatus

router = APIRouter()


@router.get("/health")
async def health(_: str = Depends(verify_token)) -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/status", response_model=SystemStatus)
async def status(_: str = Depends(verify_token)) -> SystemStatus:
    from api.server import _orchestrator

    cameras_resp: List[CameraResponse] = []
    spaces_resp: List[SpaceResponse] = []

    if _orchestrator is not None:
        # Cameras
        for cam in _orchestrator.app_config.cameras:
            worker_state = "collector" if cam.id in _orchestrator._collectors else "stopped"

            cameras_resp.append(CameraResponse(
                id=cam.id,
                source=cam.source,
                status=cam.status,
                target_classes=cam.target_classes,
                worker_state=worker_state,
            ))

        # Spaces
        for space in _orchestrator.spaces:
            state = None
            if hasattr(_orchestrator, "_vision_scheduler") and _orchestrator._vision_scheduler:
                state_obj = _orchestrator._vision_scheduler._states.get(space.id)
                if state_obj:
                    state = state_obj.state

            spaces_resp.append(SpaceResponse(
                id=space.id,
                name=space.name,
                cameras=space.camera_ids,
                state=state,
            ))

    return SystemStatus(
        cameras=cameras_resp,
        spaces=spaces_resp,
        uptime_seconds=time.monotonic(),  # Approximate; main.py could inject real start time
    )
