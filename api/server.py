"""FastAPI server with daemon uvicorn thread."""

import logging
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from api.event_bus import EventBus

logger = logging.getLogger(__name__)

# Module-level references set by start_api()
_app: Optional[FastAPI] = None
_event_bus: Optional["EventBus"] = None
_orchestrator = None
_space_logger = None
_repo = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("REST API starting")
    yield
    logger.info("REST API shutting down")


def _build_app() -> FastAPI:
    app = FastAPI(title="tracking-cano", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    from api.routes.status import router as status_router
    from api.routes.logs import router as logs_router
    from api.routes.cameras import router as cameras_router
    from api.routes.spaces import router as spaces_router
    from api.routes.config import router as config_router

    app.include_router(status_router, prefix="/api")
    app.include_router(logs_router, prefix="/api/logs")
    app.include_router(cameras_router, prefix="/api/cameras")
    app.include_router(spaces_router, prefix="/api/spaces")
    app.include_router(config_router, prefix="/api/config")

    return app


def start_api(
    port: int = 8000,
    orchestrator=None,
    space_logger=None,
    repo=None,
    event_bus: Optional["EventBus"] = None,
) -> threading.Thread:
    """Start the FastAPI server in a daemon thread.

    Returns the thread handle for lifecycle management.
    """
    global _app, _event_bus, _orchestrator, _space_logger, _repo
    _event_bus = event_bus
    _orchestrator = orchestrator
    _space_logger = space_logger
    _repo = repo

    if _app is None:
        _app = _build_app()

    import uvicorn

    config = uvicorn.Config(_app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    def _run():
        server.run()

    thread = threading.Thread(target=_run, daemon=True, name="uvicorn-api")
    thread.start()
    logger.info("REST API started on port %d", port)
    return thread


def get_app() -> FastAPI:
    if _app is None:
        _app = _build_app()
    return _app
