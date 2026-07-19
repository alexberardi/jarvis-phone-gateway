"""Per-call media-stream session: WS frames in, VAD, turn pipeline, frames out.

Composes the foundation pieces around one provider WebSocket:

    frames in -> provider.parse -> mu-law decode -> VAD endpointing
        -> [turn pipeline] -> PCM reply -> provider media frames out

The turn pipeline is injected (``async (utterance_pcm, session) -> pcm|None``)
— in production it is the whisper→LLM→TTS chain (live-wiring phase); in
tests it is a stub. Half-duplex v1: while the session is speaking, inbound
audio is suppressed at the VAD so the agent doesn't endpoint on its own
playback echo.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from audio.vad import RmsVad
from telephony.provider import (
    InboundAudio,
    MarkReceived,
    StreamStart,
    StreamStop,
    TelephonyProvider,
)
from telephony.twilio_provider import PendingSession, validate_stream_start

logger = logging.getLogger(__name__)

TurnPipeline = Callable[[np.ndarray, "MediaStreamSession"], Awaitable[np.ndarray | None]]

# WS close codes (4xxx = application-defined).
CLOSE_BAD_BINDING = 4403


class MediaStreamSession:
    def __init__(
        self,
        ws: WebSocket,
        provider: TelephonyProvider,
        vad: RmsVad,
        turn_pipeline: TurnPipeline,
        pending: PendingSession,
    ):
        self.ws = ws
        self.provider = provider
        self.vad = vad
        self.turn_pipeline = turn_pipeline
        self.pending = pending
        self.session_id = pending.session_id
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self.speaking = False
        self.hangup_requested = False
        self.turn_no = 0
        self.marks_received: list[str] = []

    async def run(self) -> None:
        """Drive the stream until stop/disconnect. Caller has already
        accepted the WS and claimed the session token."""
        try:
            while True:
                raw = await self.ws.receive_text()
                event = self.provider.parse_ws_message(raw)
                if isinstance(event, StreamStart):
                    if not validate_stream_start(event, self.pending):
                        logger.warning(
                            "Stream-start binding failed for session %s "
                            "(callSid=%s, params=%s) — closing",
                            self.session_id, event.call_sid, event.custom_parameters,
                        )
                        await self.ws.close(code=CLOSE_BAD_BINDING)
                        return
                    self.stream_sid = event.stream_sid
                    self.call_sid = event.call_sid
                    logger.info(
                        "Media stream started session=%s callSid=%s",
                        self.session_id, self.call_sid,
                    )
                elif isinstance(event, InboundAudio):
                    if self.stream_sid is None:
                        continue  # media before start — ignore
                    utterance = self.vad.feed(event.pcm, suppress=self.speaking)
                    if utterance is not None:
                        await self._run_turn(utterance)
                        if self.hangup_requested:
                            return
                elif isinstance(event, MarkReceived):
                    self.marks_received.append(event.name)
                elif isinstance(event, StreamStop):
                    logger.info("Media stream stopped session=%s", self.session_id)
                    return
        except WebSocketDisconnect:
            logger.info("Media stream disconnected session=%s", self.session_id)

    async def _run_turn(self, utterance: np.ndarray) -> None:
        self.turn_no += 1
        try:
            reply = await self.turn_pipeline(utterance, self)
        except Exception:  # noqa: BLE001 — a failed turn must not kill the call
            logger.exception(
                "Turn %d failed for session %s — staying on the line",
                self.turn_no, self.session_id,
            )
            return
        if reply is not None and len(reply):
            await self.speak(reply)

    async def speak(self, pcm_8k: np.ndarray) -> None:
        """Send one PCM buffer to the call, half-duplex guarded, marked."""
        if self.stream_sid is None:
            return
        self.speaking = True
        try:
            for msg in self.provider.media_messages(self.stream_sid, pcm_8k):
                await self.ws.send_json(msg)
            await self.ws.send_json(
                self.provider.mark_message(self.stream_sid, f"t{self.turn_no}")
            )
        finally:
            self.speaking = False

    def request_hangup(self) -> None:
        """Turn pipeline signals the call should end after this turn."""
        self.hangup_requested = True
