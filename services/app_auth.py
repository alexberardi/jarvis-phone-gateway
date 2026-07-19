"""Inbound app-to-app auth for /internal/* endpoints.

Same pattern as the other services: the caller sends X-Jarvis-App-Id/Key
and we round-trip them to jarvis-auth /internal/app-ping. Fail-closed: no
auth URL, auth unreachable, or bad creds → reject. Validated pairs are
cached briefly so per-turn escalation traffic doesn't hammer auth.
"""

from __future__ import annotations

import logging
import time

import httpx
from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 60.0
_cache: dict[tuple[str, str], float] = {}


def _cached(app_id: str, app_key: str) -> bool:
    expiry = _cache.get((app_id, app_key))
    return expiry is not None and expiry > time.monotonic()


def _remember(app_id: str, app_key: str) -> None:
    _cache[(app_id, app_key)] = time.monotonic() + _CACHE_TTL_S


def clear_cache() -> None:
    _cache.clear()


async def require_app_auth(
    request: Request,
    x_jarvis_app_id: str | None = Header(default=None),
    x_jarvis_app_key: str | None = Header(default=None),
) -> str:
    """Validate app credentials against jarvis-auth; returns the app id."""
    if not x_jarvis_app_id or not x_jarvis_app_key:
        raise HTTPException(status_code=401, detail="Missing app credentials")

    if _cached(x_jarvis_app_id, x_jarvis_app_key):
        return x_jarvis_app_id

    auth_url = getattr(request.app.state.config, "auth_url", "")
    if not auth_url:
        raise HTTPException(
            status_code=503, detail="Auth URL not configured (fail-closed)"
        )
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.post(
                f"{auth_url.rstrip('/')}/internal/app-ping",
                headers={
                    "X-Jarvis-App-Id": x_jarvis_app_id,
                    "X-Jarvis-App-Key": x_jarvis_app_key,
                },
            )
    except httpx.HTTPError as e:
        logger.error("App-auth round-trip failed: %s", e)
        raise HTTPException(status_code=503, detail="Auth service unreachable")
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid app credentials")
    _remember(x_jarvis_app_id, x_jarvis_app_key)
    return x_jarvis_app_id
