"""Config read/update endpoints — modifies configuration.yaml sections."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from api.auth import verify_token
from api.models import LLMUpdate, ThresholdsUpdate, YOLOUpdate
from core.yaml_writer import (
    read_yaml,
    update_yaml_config_section,
)

router = APIRouter()


@router.get("/")
async def get_config(_: str = Depends(verify_token)) -> Dict[str, Any]:
    return read_yaml()


@router.put("/thresholds")
async def update_thresholds(body: ThresholdsUpdate, _: str = Depends(verify_token)) -> Dict[str, Any]:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    return update_yaml_config_section("thresholds", updates)


@router.put("/yolo")
async def update_yolo(body: YOLOUpdate, _: str = Depends(verify_token)) -> Dict[str, Any]:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    return update_yaml_config_section("yolo", updates)


@router.put("/llm")
async def update_llm(body: LLMUpdate, _: str = Depends(verify_token)) -> Dict[str, Any]:
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    return update_yaml_config_section("llm", updates)
