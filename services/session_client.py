"""CC phone-session client (gateway -> command-center).

Encodes the PRD's CC↔gateway interface: the two services never share a
database — CC owns ``phone_call_sessions`` and the gateway reports
everything through ``POST /internal/phone/sessions/{id}/events`` with
app-to-app credentials. Event kinds:

- ``claim_dial``   — atomic confirmed→dialing compare-and-set, records this
                     worker's base URL (session affinity). 200 → proceed;
                     409 → someone else claimed it / not confirmed → DROP.
- ``state``        — lifecycle transitions (in_call, wrapup, done, failed).
- ``turn``         — per-turn transcript append + stage timings. Doubles as
                     a heartbeat.
- ``heartbeat``    — bare liveness for long silences (CC reaper: stale 60s
                     in_call sessions get terminated).
- ``outcome``      — final structured facts + MinIO audio key.

NOTE: the CC endpoints do not exist yet (CC-side P1 work). This client is
the contract's gateway half; contract tests on both sides per the PRD test
strategy.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SessionEventError(RuntimeError):
    """CC rejected an event with an unexpected status."""


class SessionClient:
    def __init__(self, cc_base_url: str, app_id: str, app_key: str):
        self.base_url = cc_base_url.rstrip("/")
        self._headers = {"X-Jarvis-App-Id": app_id, "X-Jarvis-App-Key": app_key}

    def _events_url(self, session_id: str) -> str:
        return f"{self.base_url}/internal/phone/sessions/{session_id}/events"

    async def get_session(self, session_id: str, *, http: httpx.AsyncClient) -> dict:
        r = await http.get(
            f"{self.base_url}/internal/phone/sessions/{session_id}",
            headers=self._headers,
        )
        r.raise_for_status()
        return r.json()

    async def _post_event(
        self,
        session_id: str,
        payload: dict[str, Any],
        *,
        http: httpx.AsyncClient,
        allow_conflict: bool = False,
    ) -> bool:
        r = await http.post(
            self._events_url(session_id), json=payload, headers=self._headers
        )
        if r.status_code == 200:
            return True
        if allow_conflict and r.status_code == 409:
            return False
        raise SessionEventError(
            f"CC rejected {payload.get('type')} event for {session_id}: "
            f"HTTP {r.status_code}"
        )

    async def claim_for_dial(
        self, session_id: str, worker_url: str, *, http: httpx.AsyncClient
    ) -> bool:
        """confirmed→dialing compare-and-set. False = do NOT dial.

        Security requirement 1: the Redis job is transport only — this call
        against CC's session row is the authorization. Unknown, duplicate,
        or not-confirmed sessions come back 409 and the job is dropped.
        """
        return await self._post_event(
            session_id,
            {"type": "claim_dial", "worker_url": worker_url},
            http=http,
            allow_conflict=True,
        )

    async def state_event(
        self, session_id: str, state: str, *, http: httpx.AsyncClient, **extra: Any
    ) -> None:
        await self._post_event(
            session_id, {"type": "state", "state": state, **extra}, http=http
        )

    async def turn_event(
        self, session_id: str, turn: dict[str, Any], *, http: httpx.AsyncClient
    ) -> None:
        """Per-turn transcript + timings; CC treats it as a heartbeat too."""
        await self._post_event(session_id, {"type": "turn", "turn": turn}, http=http)

    async def heartbeat(self, session_id: str, *, http: httpx.AsyncClient) -> None:
        await self._post_event(session_id, {"type": "heartbeat"}, http=http)

    async def escalation_event(
        self, session_id: str, question: str, *, http: httpx.AsyncClient
    ) -> None:
        """Mid-call escalation: CC turns this into the push + inbox card whose
        answer comes back over POST /internal/call/{id}/escalation-answer."""
        await self._post_event(
            session_id, {"type": "escalation", "question": question}, http=http
        )

    async def outcome_event(
        self,
        session_id: str,
        outcome: dict[str, Any],
        *,
        http: httpx.AsyncClient,
        audio_key: str | None = None,
    ) -> None:
        await self._post_event(
            session_id,
            {"type": "outcome", "outcome": outcome, "audio_key": audio_key},
            http=http,
        )
