"""Log query and SSE streaming endpoints."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse

from api.auth import verify_token
from api.models import LogEntryResponse, LogLabelUpdate

router = APIRouter()


def _get_repo():
    from api.server import _repo
    return _repo


@router.get("/", response_model=List[LogEntryResponse])
async def list_logs(
    log_type: Optional[str] = Query(None),
    subject_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: str = Depends(verify_token),
) -> List[LogEntryResponse]:
    repo = _get_repo()
    if repo is None:
        return []

    from storage.database import LogEntry
    with repo._session_factory() as session:
        q = session.query(LogEntry)
        if log_type:
            q = q.filter(LogEntry.log_type == log_type)
        if subject_id:
            q = q.filter(LogEntry.subject_id == subject_id)
        q = q.order_by(LogEntry.id.desc()).limit(limit).offset(offset)
        entries = q.all()

    return [_to_response(e) for e in entries]


@router.get("/recent", response_model=List[LogEntryResponse])
async def recent_logs(
    n: int = Query(10, ge=1, le=100),
    _: str = Depends(verify_token),
) -> List[LogEntryResponse]:
    repo = _get_repo()
    if repo is None:
        return []

    from storage.database import LogEntry
    with repo._session_factory() as session:
        entries = session.query(LogEntry).order_by(LogEntry.id.desc()).limit(n).all()

    return [_to_response(e) for e in entries]


@router.get("/stream")
async def stream_logs(
    event_type: Optional[str] = Query(None),
    _: str = Depends(verify_token),
) -> EventSourceResponse:
    """SSE endpoint for real-time log events."""
    from api.server import _event_bus

    if not _event_bus:
        raise HTTPException(status_code=503, detail="Event bus not available")

    queue, sid = _event_bus.subscribe(event_type)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield {"event": "ping", "data": ""}
                    continue

                if event_type and event.get("type") != event_type:
                    continue

                data = event.get("data", event)
                yield {"event": event.get("type", "message"), "data": _json_dumps(data)}
        finally:
            _event_bus.unsubscribe(sid)

    return EventSourceResponse(event_generator())


@router.get("/{log_id}", response_model=LogEntryResponse)
async def get_log(log_id: int, _: str = Depends(verify_token)) -> LogEntryResponse:
    repo = _get_repo()
    if repo is None:
        raise HTTPException(status_code=503, detail="Database not available")

    from storage.database import LogEntry
    with repo._session_factory() as session:
        entry = session.query(LogEntry).filter(LogEntry.id == log_id).first()

    if not entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    return _to_response(entry)


@router.get("/{log_id}/image")
async def get_log_image(log_id: int, _: str = Depends(verify_token)):
    from api.server import _space_logger

    repo = _get_repo()
    if repo is None:
        raise HTTPException(status_code=503, detail="Database not available")

    from storage.database import LogEntry
    with repo._session_factory() as session:
        entry = session.query(LogEntry).filter(LogEntry.id == log_id).first()

    if not entry or not entry.image_path:
        raise HTTPException(status_code=404, detail="Image not found")

    path = entry.image_path

    local_path = Path("output") / path
    if local_path.exists():
        return Response(content=local_path.read_bytes(), media_type="image/jpeg")

    if _space_logger and _space_logger._minio:
        try:
            response = _space_logger._minio.get_object(_space_logger._minio_config.bucket, path)
            data = response.read()
            response.close()
            response.release_conn()
            return Response(content=data, media_type="image/jpeg")
        except Exception:
            pass

    raise HTTPException(status_code=404, detail="Image not found")


@router.patch("/{log_id}/label", response_model=LogEntryResponse)
async def patch_log_label(
    log_id: int,
    body: LogLabelUpdate,
    _: str = Depends(verify_token),
) -> LogEntryResponse:
    repo = _get_repo()
    if repo is None:
        raise HTTPException(status_code=503, detail="Database not available")

    from storage.database import LogEntry
    with repo._session_factory() as session:
        entry = session.query(LogEntry).filter(LogEntry.id == log_id).first()
        if not entry:
            raise HTTPException(status_code=404, detail="Log entry not found")

        if body.is_false_positive is not None:
            entry.is_false_positive = body.is_false_positive
        if body.is_false_negative is not None:
            entry.is_false_negative = body.is_false_negative

        session.add(entry)
        session.commit()

    return _to_response(entry)


def _to_response(entry) -> LogEntryResponse:
    return LogEntryResponse(
        id=entry.id,
        timestamp=entry.timestamp if entry.timestamp else datetime.utcnow(),
        log_type=entry.log_type,
        subject_id=entry.subject_id,
        target_present=entry.target_present,
        description=entry.description,
        target_coordinate=entry.target_coordinate,
        visual_evidence=getattr(entry, 'visual_evidence', None),
        image_url=f"/api/logs/{entry.id}/image" if entry.image_path else None,
        is_false_positive=getattr(entry, 'is_false_positive', False),
        is_false_negative=getattr(entry, 'is_false_negative', False),
    )


def _json_dumps(obj: Any) -> str:
    import json
    def default_handler(o):
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=default_handler, ensure_ascii=False)
