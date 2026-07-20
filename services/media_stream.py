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

import asyncio
import logging
import time
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
    # Upper bound on waiting for the provider to confirm the final buffer
    # played. Generous enough for a long goodbye, short enough that a lost
    # mark can't wedge the call open.
    FINAL_PLAYBACK_TIMEOUT_S = 10.0
    # After a deferred hangup we stay on the line to let the other party
    # confirm. If they say nothing, the call must still end — without this the
    # only backstop is DEFAULT_MAX_CALL_SECONDS (600s) of billed silence.
    #
    # Kept SHORT. This window is pure dead air on the wire, and the agent has
    # usually just said goodbye. At 8s (2026-07-20) the shop hung up on us
    # first, which reads as the agent failing to end the call. A couple of
    # seconds is a natural end-of-call pause; longer is a silence.
    IDLE_HANGUP_SECONDS = 2.5

    def __init__(
        self,
        ws: WebSocket,
        provider: TelephonyProvider,
        vad: RmsVad,
        turn_pipeline: TurnPipeline,
        pending: PendingSession,
        *,
        recorder: Any | None = None,
        on_stream_start: Callable[["MediaStreamSession"], Awaitable[None]] | None = None,
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
        # Mark name for the most recent spoken buffer, awaited before hangup.
        self._pending_mark: str | None = None
        # Absolute deadline after which an idle call ends itself; armed only
        # when a hangup was deferred waiting for the other party to confirm.
        self._idle_deadline: float | None = None
        # Optional local-recording tap (PRD decision 9): inbound frames and
        # outbound bursts both flow through this session, so it is the one
        # place a complete two-direction recording can be captured.
        self.recorder = recorder
        # Fired once after a VALIDATED stream-start — where the runtime
        # speaks the (pre-synthesized) disclosure and marks the call in_call.
        self.on_stream_start = on_stream_start

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
                    if self.on_stream_start is not None:
                        await self.on_stream_start(self)
                        if self.hangup_requested:
                            await self._await_playback()
                            return
                elif isinstance(event, InboundAudio):
                    if self.stream_sid is None:
                        continue  # media before start — ignore
                    if self.recorder is not None:
                        self.recorder.add_inbound(event.pcm)
                    if (
                        self._idle_deadline is not None
                        and time.monotonic() > self._idle_deadline
                        and not self.speaking
                    ):
                        logger.warning(
                            "Idle %.1fs after deferred hangup on session %s "
                            "— ending the call",
                            self.IDLE_HANGUP_SECONDS, self.session_id,
                        )
                        self.hangup_requested = True
                        return
                    utterance = self.vad.feed(event.pcm, suppress=self.speaking)
                    if utterance is not None:
                        self._idle_deadline = None  # they spoke; let the turn run
                        await self._run_turn(utterance)
                        if self.hangup_requested:
                            await self._await_playback()
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
            if self.recorder is not None:
                self.recorder.add_outbound(pcm_8k)
            for msg in self.provider.media_messages(self.stream_sid, pcm_8k):
                await self.ws.send_json(msg)
            mark = f"t{self.turn_no}"
            self._pending_mark = mark
            await self.ws.send_json(
                self.provider.mark_message(self.stream_sid, mark)
            )
        finally:
            self.speaking = False

    def arm_idle_hangup(self, seconds: float | None = None) -> None:
        """End the call if the other party stays silent for `seconds`.

        Used after a deferred hangup: the agent has said its goodbye but is
        holding the line for a confirmation that may never come.
        """
        self._idle_deadline = time.monotonic() + (
            self.IDLE_HANGUP_SECONDS if seconds is None else seconds
        )

    async def _await_playback(self) -> None:
        """Block until the provider confirms the last buffer finished playing.

        ``speak`` only QUEUES frames onto the websocket; Twilio plays them
        asynchronously and echoes the mark when the audio has actually been
        heard. Returning straight from a hangup turn therefore closed the
        socket mid-sentence and the caller heard a click (live 2026-07-20:
        the goodbye was cut off on every completed call). Marks arrive on the
        same event stream as everything else, so this keeps draining events
        rather than waiting on a future nothing would resolve.
        """
        mark = self._pending_mark
        if mark is None or mark in self.marks_received:
            return
        deadline = time.monotonic() + self.FINAL_PLAYBACK_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "Timed out waiting for playback of %s on session %s — "
                    "closing anyway", mark, self.session_id,
                )
                return
            try:
                raw = await asyncio.wait_for(
                    self.ws.receive_text(), timeout=remaining
                )
            except (asyncio.TimeoutError, WebSocketDisconnect):
                return
            event = self.provider.parse_ws_message(raw)
            if isinstance(event, MarkReceived):
                self.marks_received.append(event.name)
                if event.name == mark:
                    return
            elif isinstance(event, InboundAudio):
                # Keep the recording complete through the goodbye.
                if self.recorder is not None:
                    self.recorder.add_inbound(event.pcm)
            elif isinstance(event, StreamStop):
                return

    def request_hangup(self) -> None:
        """Turn pipeline signals the call should end after this turn."""
        self.hangup_requested = True
