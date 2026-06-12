"""Bearer token authentication for REST API."""

import os
from typing import Optional

from fastapi import Depends, Header, HTTPException


def _get_api_key() -> Optional[str]:
    return os.environ.get("API_KEY") or None


async def verify_token(
    authorization: Optional[str] = Header(None),
) -> str:
    """FastAPI dependency — optional Bearer auth.

    If API_KEY env var is set, requires valid token. Otherwise passes through.
    Returns the authenticated identity string (or 'anonymous').
    """
    expected = _get_api_key()
    if not expected:
        return "anonymous"

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization[len("Bearer "):]
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return "authenticated"
