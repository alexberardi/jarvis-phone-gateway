"""llm-proxy streaming client for call turns.

Contract facts (P0 + llm-proxy PR #47):
- ``/v1/chat/completions`` with ``stream=true`` is NOT OpenAI chunk format:
  frames are ``{"delta": "tok"}`` … ``{"done": true, content, usage, ...}``,
  plus ``{"error": "..."}`` and ``{"cancelled": true}``. No ``data: [DONE]``.
- Every streamed request carries an ``X-Request-Id`` (ours), echoed back;
  ``POST /v1/chat/completions/cancel/{request_id}`` aborts the generation at
  the next token boundary.
- The turn cap MUST cancel, never silently abandon: an abandoned stream is
  exactly the model-service wedge the P0 spike hit (failure ladder item 3).
  ``stream_deltas`` fires the cancel endpoint on timeout before raising.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TURN_TIMEOUT_S = 20.0

SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


class LlmStreamError(RuntimeError):
    """Upstream reported an error frame or a non-200 response."""


class TurnTimeout(TimeoutError):
    """The turn cap fired; the upstream generation was cancelled."""

    def __init__(self, request_id: str, timeout_s: float):
        super().__init__(f"LLM turn exceeded {timeout_s}s (request {request_id})")
        self.request_id = request_id


class LlmProxyStreamClient:
    def __init__(self, base_url: str, app_id: str, app_key: str):
        self.base_url = base_url.rstrip("/")
        self._headers = {"X-Jarvis-App-Id": app_id, "X-Jarvis-App-Key": app_key}

    async def stream_deltas(
        self,
        messages: list[dict],
        *,
        http: httpx.AsyncClient,
        model: str = "live",
        max_tokens: int = 400,
        turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_S,
    ) -> AsyncIterator[str]:
        """Yield token deltas; enforce the turn cap WITH upstream cancel."""
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        deadline = loop.time() + turn_timeout_s
        try:
            async with http.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": max_tokens,
                },
                headers={**self._headers, "X-Request-Id": request_id},
                timeout=httpx.Timeout(turn_timeout_s, connect=5),
            ) as r:
                if r.status_code != 200:
                    raise LlmStreamError(
                        f"llm-proxy returned {r.status_code} for streamed chat"
                    )
                lines = r.aiter_lines()
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    line = await asyncio.wait_for(anext(lines, None), timeout=remaining)
                    if line is None:
                        return
                    if not line.startswith("data: "):
                        continue
                    try:
                        frame = json.loads(line[6:])
                    except json.JSONDecodeError:
                        logger.warning("Unparseable SSE frame skipped: %r", line[:80])
                        continue
                    if frame.get("done"):
                        return
                    if frame.get("cancelled"):
                        return
                    if "error" in frame:
                        raise LlmStreamError(str(frame["error"]))
                    delta = frame.get("delta") or ""
                    if delta:
                        yield delta
        except (asyncio.TimeoutError, httpx.TimeoutException):
            # Cancel upstream so the generation stops NOW — abandoning the
            # stream is the wedge trigger the turn cap exists to avoid.
            await self._cancel(http, request_id)
            raise TurnTimeout(request_id, turn_timeout_s) from None

    async def complete(
        self,
        messages: list[dict],
        *,
        http: httpx.AsyncClient,
        model: str = "live",
        max_tokens: int = 64,
        timeout_s: float = 10.0,
    ) -> str:
        """One short non-streaming completion, for classification side-calls.

        Non-streaming IS OpenAI-shaped on llm-proxy (only the streaming path
        diverges), so this reads choices[0] normally. Raises on anything
        unexpected — callers of this path are guards, and a guard that
        cannot tell "no" from "the server broke" is not a guard.
        """
        r = await http.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
            },
            headers=self._headers,
            timeout=httpx.Timeout(timeout_s, connect=5),
        )
        if r.status_code != 200:
            raise LlmStreamError(f"llm-proxy returned {r.status_code} for completion")
        body = r.json()
        return (body["choices"][0]["message"]["content"] or "").strip()

    async def _cancel(self, http: httpx.AsyncClient, request_id: str) -> None:
        try:
            await http.post(
                f"{self.base_url}/v1/chat/completions/cancel/{request_id}",
                headers=self._headers,
                timeout=5.0,
            )
        except httpx.HTTPError as e:
            # Best-effort: the model service's own disconnect-abort is the
            # backstop; log so a broken cancel path is visible.
            logger.warning("Cancel of request %s failed: %s", request_id, e)


async def sentences(deltas: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup a delta stream into complete sentences (spike-proven split)."""
    buf = ""
    async for delta in deltas:
        buf += delta
        parts = SENTENCE_RE.split(buf)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                if sentence.strip():
                    yield sentence.strip()
            buf = parts[-1]
    if buf.strip():
        yield buf.strip()
