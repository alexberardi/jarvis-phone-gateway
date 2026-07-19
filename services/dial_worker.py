"""Dial worker: Redis job → CC claim → Twilio dial → call lifecycle → outcome.

The worker owns everything around the media stream (PRD decisions 4 + 10,
call-lifecycle section):

    pop {session_id, household_id}            (queue = transport, never authz)
      → GET session from CC                   (unknown → drop)
      → claim_dial CAS confirmed→dialing      (409 → drop; this IS the authz)
      → pre-synthesize the disclosure         (prewarm during ring; TTS down
                                               → abort BEFORE dialing: the
                                               disclosure is never skippable)
      → single-use wss token → TwiML → calls.create
      → stream lands (media WS) → disclosure spoken → in_call
      → heartbeats every ≤30 s + max_call_seconds watchdog (belt — CC's
        reaper is the suspenders)
      → wrapup: background-model summary, recording → MinIO, outcome event

Notice-off rule (PRD decision 9): session["record_enabled"] False → no
recorder is ever constructed, turn events carry timings only (no
transcript), and the summary is generated from in-memory history that dies
with this object.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from config import GatewayConfig
from llm.client import LlmProxyStreamClient
from queues.dial_queue import DialJob, DialQueue
from services.escalation import EscalationWindow
from services.prompt import build_disclosure
from services.recording import CallRecorder, upload_recording
from services.session_client import SessionClient, SessionEventError
from services.turn_pipeline import LiveTurnPipeline, synthesize
from telephony.provider import TelephonyProvider
from telephony.twilio_provider import SessionTokenRegistry

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 25.0
STREAM_START_TIMEOUT_S = 60.0
DEFAULT_MAX_CALL_SECONDS = 600
_HANGUP_GRACE_S = 10.0

_SUMMARY_INSTRUCTION = (
    "Summarize this phone call in two sentences for the person it was made "
    "on behalf of. State plainly whether the goal was achieved. Do not "
    "invent details."
)


@dataclass
class CallRuntime:
    """Everything the media WS needs, keyed by session_id in app.state."""

    session_id: str
    session: dict[str, Any]
    pipeline: LiveTurnPipeline
    escalation: EscalationWindow
    recorder: CallRecorder | None
    disclosure_pcm: np.ndarray | None = None
    call_sid: str | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    stream_done: asyncio.Event = field(default_factory=asyncio.Event)
    media_session: Any | None = None

    async def on_stream_start(self, media_session: Any) -> None:
        """Validated stream-start: speak the disclosure, mark in_call."""
        self.media_session = media_session
        self.started.set()
        if self.disclosure_pcm is not None and len(self.disclosure_pcm):
            await media_session.speak(self.disclosure_pcm)


class DialWorker:
    def __init__(
        self,
        cfg: GatewayConfig,
        *,
        provider: TelephonyProvider,
        token_registry: SessionTokenRegistry,
        dial_queue: DialQueue,
        call_runtimes: dict[str, CallRuntime],
        session_client: SessionClient | None = None,
        http: httpx.AsyncClient | None = None,
    ):
        self.cfg = cfg
        self.provider = provider
        self.token_registry = token_registry
        self.dial_queue = dial_queue
        self.call_runtimes = call_runtimes
        self.session_client = session_client or SessionClient(
            cfg.cc_base_url, cfg.app_id, cfg.app_key
        )
        self.llm = LlmProxyStreamClient(cfg.llm_url, cfg.app_id, cfg.app_key)
        self._http = http
        self._running = False

    def _client(self) -> httpx.AsyncClient:
        # Default app-auth headers so EVERY outbound hop (whisper /transcribe,
        # tts /speak/stream, …) authenticates — found live 2026-07-19: bare
        # calls got 401 from tts, which correctly blocked dialing at the
        # disclosure synth (stub-backed tests never enforced auth).
        return self._http or httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5),
            headers={
                "X-Jarvis-App-Id": self.cfg.app_id,
                "X-Jarvis-App-Key": self.cfg.app_key,
            },
        )

    # ------------------------------------------------------------ loop

    async def run_forever(self) -> None:
        """Consume dial jobs until cancelled. Never dies on a bad job."""
        self._running = True
        logger.info("Dial worker consuming %r", self.dial_queue._key)
        while self._running:
            try:
                job = await asyncio.to_thread(self.dial_queue.pop, 5)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — Redis down: back off, retry
                logger.error("Dial queue pop failed (%s) — retrying in 5s", e)
                await asyncio.sleep(5)
                continue
            if job is None:
                continue
            try:
                await self.handle_job(job)
            except Exception:  # noqa: BLE001 — one bad call must not stop the worker
                logger.exception("Dial job for session %s failed", job.session_id)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------ one call

    async def handle_job(self, job: DialJob) -> None:
        http = self._client()
        owns_http = self._http is None
        token: str | None = None
        runtime: CallRuntime | None = None
        try:
            session = await self._load_session(job, http)
            if session is None:
                return

            claimed = await self.session_client.claim_for_dial(
                job.session_id, self.cfg.public_url or "unknown", http=http
            )
            if not claimed:
                logger.info(
                    "Session %s not claimable (409) — dropping job", job.session_id
                )
                return

            dialed_number = str(session.get("dialed_number") or "")
            if not dialed_number:
                await self._fail(job.session_id, "session has no dialed_number", http)
                return

            # Prewarm during ring — and a hard gate: no disclosure, no dial.
            disclosure_pcm, _ = await synthesize(
                build_disclosure(session), self.cfg.tts_url, http
            )
            if not len(disclosure_pcm):
                await self._fail(
                    job.session_id, "disclosure synthesis failed (TTS down?)", http
                )
                return

            record_enabled = bool(session.get("record_enabled", True))
            escalation = EscalationWindow()
            pipeline = LiveTurnPipeline(
                session=session,
                whisper_url=self.cfg.whisper_url,
                tts_url=self.cfg.tts_url,
                llm=self.llm,
                http=http,
                session_client=self.session_client,
                escalation=escalation,
                redact_transcript=not record_enabled,
            )
            runtime = CallRuntime(
                session_id=job.session_id,
                session=session,
                pipeline=pipeline,
                escalation=escalation,
                recorder=CallRecorder() if record_enabled else None,
                disclosure_pcm=disclosure_pcm,
            )
            self.call_runtimes[job.session_id] = runtime

            token = self.token_registry.issue(job.session_id)
            wss_url = f"{self.cfg.public_wss_url}/media/{token}"
            twiml = self.provider.build_stream_instructions(
                wss_url, {"session_id": job.session_id}
            )

            try:
                call_sid = await self.provider.start_call(
                    to_number=dialed_number, instructions=twiml, http=http
                )
            except Exception as e:  # noqa: BLE001
                await self._fail(job.session_id, f"calls.create failed: {e}", http)
                return
            runtime.call_sid = call_sid
            self.token_registry.bind_call_sid(token, call_sid)
            logger.info(
                "Dialing %s for session %s (callSid=%s)",
                dialed_number, job.session_id, call_sid,
            )

            started = await self._wait_started(runtime)
            if not started:
                await self._end_call_quietly(call_sid, http)
                await self._fail(
                    job.session_id, "no media stream within 60s (no answer?)", http
                )
                return
            await self._state_quietly(job.session_id, "in_call", http)

            await self._supervise(job.session_id, runtime, session, http)
            await self._wrapup(job.session_id, runtime, session, http)
        finally:
            self.call_runtimes.pop(job.session_id, None)
            if token is not None:
                self.token_registry.revoke(token)  # no-op if claimed
            if owns_http:
                await http.aclose()

    # ------------------------------------------------------------ phases

    async def _load_session(
        self, job: DialJob, http: httpx.AsyncClient
    ) -> dict[str, Any] | None:
        try:
            return await self.session_client.get_session(job.session_id, http=http)
        except (httpx.HTTPError, SessionEventError) as e:
            logger.warning(
                "Dropping dial job %s: session fetch failed (%s)", job.session_id, e
            )
            return None

    async def _wait_started(self, runtime: CallRuntime) -> bool:
        try:
            await asyncio.wait_for(
                runtime.started.wait(), timeout=STREAM_START_TIMEOUT_S
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def _supervise(
        self,
        session_id: str,
        runtime: CallRuntime,
        session: dict[str, Any],
        http: httpx.AsyncClient,
    ) -> None:
        """Heartbeat + max_call_seconds watchdog until the stream ends."""
        max_seconds = float(session.get("max_call_seconds") or DEFAULT_MAX_CALL_SECONDS)
        started_at = time.monotonic()
        cap_fired = False
        while not runtime.stream_done.is_set():
            elapsed = time.monotonic() - started_at
            if elapsed >= max_seconds:
                if not cap_fired:
                    cap_fired = True
                    logger.warning(
                        "Session %s hit max_call_seconds=%s — ending call",
                        session_id, max_seconds,
                    )
                    if runtime.call_sid:
                        await self._end_call_quietly(runtime.call_sid, http)
                    # Grace: let the stream-stop arrive; then give up waiting.
                    try:
                        await asyncio.wait_for(
                            runtime.stream_done.wait(), timeout=_HANGUP_GRACE_S
                        )
                    except asyncio.TimeoutError:
                        pass
                break
            timeout = min(HEARTBEAT_INTERVAL_S, max_seconds - elapsed)
            try:
                await asyncio.wait_for(runtime.stream_done.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                try:
                    await self.session_client.heartbeat(session_id, http=http)
                except Exception as e:  # noqa: BLE001 — CC down ≠ dead call
                    logger.warning("Heartbeat failed for %s: %s", session_id, e)

    async def _wrapup(
        self,
        session_id: str,
        runtime: CallRuntime,
        session: dict[str, Any],
        http: httpx.AsyncClient,
    ) -> None:
        await self._state_quietly(session_id, "wrapup", http)
        pipeline = runtime.pipeline

        summary = await self._summarize(pipeline, http)

        audio_key: str | None = None
        if runtime.recorder is not None:
            audio_key = await upload_recording(
                str(session.get("household_id") or "unknown"),
                session_id,
                runtime.recorder.wav_bytes(),
            )

        outcome = {
            "summary": summary,
            "facts": pipeline.outcome_facts,
            "turns": len(pipeline.turn_records),
            "escalation_unanswered": pipeline.escalation_unanswered,
            "audio_available": audio_key is not None,
        }
        try:
            await self.session_client.outcome_event(
                session_id, outcome, http=http, audio_key=audio_key
            )
            await self.session_client.state_event(session_id, "done", http=http)
        except Exception as e:  # noqa: BLE001 — last resort: log loudly
            logger.error("Outcome/done report failed for %s: %s", session_id, e)

    async def _summarize(
        self, pipeline: LiveTurnPipeline, http: httpx.AsyncClient
    ) -> str:
        """Post-call summary on the background model; honest fallback."""
        transcript = pipeline.transcript()
        if not transcript:
            return "The call ended before any conversation took place."
        messages = transcript + [{"role": "user", "content": _SUMMARY_INSTRUCTION}]
        try:
            r = await http.post(
                f"{self.cfg.llm_url.rstrip('/')}/v1/chat/completions",
                json={"model": "background", "messages": messages, "stream": False},
                headers={
                    "X-Jarvis-App-Id": self.cfg.app_id,
                    "X-Jarvis-App-Key": self.cfg.app_key,
                },
                timeout=60.0,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return str(content).strip() or "Call completed."
        except Exception as e:  # noqa: BLE001
            logger.error("Wrapup summary failed: %s", e)
            facts = "; ".join(pipeline.outcome_facts)
            return (
                f"Call completed. Recorded facts: {facts}"
                if facts
                else "Call completed (summary unavailable)."
            )

    # ------------------------------------------------------------ helpers

    async def _fail(
        self, session_id: str, reason: str, http: httpx.AsyncClient
    ) -> None:
        logger.error("Session %s failed: %s", session_id, reason)
        try:
            await self.session_client.state_event(
                session_id, "failed", http=http, reason=reason
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Could not report failure for %s: %s", session_id, e)

    async def _state_quietly(
        self, session_id: str, state: str, http: httpx.AsyncClient
    ) -> None:
        try:
            await self.session_client.state_event(session_id, state, http=http)
        except Exception as e:  # noqa: BLE001
            logger.warning("State event %r failed for %s: %s", state, session_id, e)

    async def _end_call_quietly(self, call_sid: str, http: httpx.AsyncClient) -> None:
        try:
            await self.provider.end_call(call_sid, http)
        except Exception as e:  # noqa: BLE001
            logger.error("REST hangup of %s failed: %s", call_sid, e)
