"""Redis dial-queue consumer (PRD decision 4).

A dedicated queue — NOT llm-proxy's one-shot job queue — because call
sessions need liveness/heartbeat semantics. Job shape is deliberately
minimal: ``{"session_id": ..., "household_id": ...}`` and NOTHING more.

Security requirement 1 — the queue is transport, never authorization: stack
Redis is unauthenticated on jarvis-net and anything that can reach it can
LPUSH. A worker that pops a job must load the session from CC and dial only
after the atomic confirmed→dialing claim (SessionClient.claim_for_dial); a
forged entry with no matching confirmed session does nothing. Malformed
payloads are logged and dropped, never retried.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

DIAL_QUEUE_KEY = "phone:dial"


class RedisLike(Protocol):
    """The two operations we use — lets tests stub without a Redis server."""

    def blpop(self, keys, timeout: int = 0): ...
    def rpush(self, key, *values): ...


@dataclass
class DialJob:
    session_id: str
    household_id: str


def parse_job(raw: bytes | str) -> DialJob | None:
    """Strict parse of a queue payload; anything else is dropped with a log."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Dropping unparseable dial job: %r", raw[:200])
        return None
    if not isinstance(data, dict):
        logger.warning("Dropping non-object dial job: %r", raw[:200])
        return None
    session_id = data.get("session_id")
    household_id = data.get("household_id")
    if not isinstance(session_id, str) or not session_id:
        logger.warning("Dropping dial job without session_id")
        return None
    if not isinstance(household_id, str) or not household_id:
        logger.warning("Dropping dial job without household_id")
        return None
    return DialJob(session_id=session_id, household_id=household_id)


class DialQueue:
    def __init__(self, redis_client: RedisLike, queue_key: str = DIAL_QUEUE_KEY):
        self._redis = redis_client
        self._key = queue_key

    def pop(self, timeout_s: int = 5) -> DialJob | None:
        """Blocking pop of one job; None on timeout or malformed payload."""
        item = self._redis.blpop([self._key], timeout=timeout_s)
        if item is None:
            return None
        _key, raw = item
        return parse_job(raw)

    def push(self, job: DialJob) -> None:
        """Producer helper (CC is the real producer; used by tests/tools)."""
        self._redis.rpush(
            self._key,
            json.dumps({"session_id": job.session_id, "household_id": job.household_id}),
        )
