"""jarvis-phone-gateway — telephony media bridge (:7713).

FastAPI skeleton for P1: /health plus the Twilio media WebSocket. The live
turn pipeline (whisper → llm-proxy → tts) is injected at app creation; the
default placeholder stays silent so the skeleton is runnable and testable
without any Jarvis services up.

Security posture at the WS door (PRD security requirement 2):
1. The wss path carries a single-use per-session token — claimed (popped)
   before accept; unknown or replayed tokens are rejected pre-handshake.
2. When Twilio credentials are configured, X-Twilio-Signature is validated
   with the https↔wss scheme fix before accept.
3. The stream-start event must bind to the claimed session (session_id
   custom parameter + callSid) or the socket closes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from audio.vad import RmsVad, VadConfig
from config import GatewayConfig
from queues.dial_queue import DialQueue
from services.app_auth import require_app_auth
from services.dial_worker import DialWorker
from services.line_lookup import lookup_line_type
from services.media_stream import MediaStreamSession
from telephony.twilio_provider import (
    SessionTokenRegistry,
    TwilioProvider,
    validate_ws_signature,
)

logger = logging.getLogger("uvicorn")

SERVICE_NAME = "jarvis-phone-gateway"


# Module-level on purpose: with `from __future__ import annotations`,
# closure-local Pydantic models don't survive FastAPI's type resolution
# (the annotation degrades to a query param).
class EscalationAnswerBody(BaseModel):
    answer: str = Field(..., min_length=1, max_length=2000)


class LineTypeBody(BaseModel):
    number: str = Field(..., min_length=4, max_length=32)


def _setup_jarvis_logging() -> None:
    """JarvisLogger via jarvis-log-client when installed; stdlib otherwise."""
    try:
        from jarvis_log_client import init as init_log_client  # type: ignore

        import os

        app_key = os.getenv("JARVIS_APP_KEY")
        if app_key:
            init_log_client(
                app_id=os.getenv("JARVIS_APP_ID", SERVICE_NAME), app_key=app_key
            )
            logger.info("jarvis-log-client initialized")
    except ImportError:
        logger.info("jarvis-log-client not installed — stdlib logging only")
    except Exception as e:  # noqa: BLE001 — logging must never block boot
        logger.warning("jarvis-log-client init failed: %s", e)


def _discover_service_urls(cfg: GatewayConfig) -> None:
    """Best-effort config-service discovery; env values are the fallback."""
    try:
        from jarvis_config_client import get_service_url  # type: ignore

        for attr, name in (
            ("cc_base_url", "jarvis-command-center"),
            ("whisper_url", "jarvis-whisper-api"),
            ("llm_url", "jarvis-llm-proxy-api"),
            ("tts_url", "jarvis-tts"),
        ):
            try:
                url = get_service_url(name)
                if url:
                    setattr(cfg, attr, url.rstrip("/"))
            except Exception:  # noqa: BLE001 — per-service fallback to env
                pass
        logger.info("Service discovery applied (config-service)")
    except ImportError:
        logger.info("jarvis-config-client not installed — using env URLs")
    except Exception as e:  # noqa: BLE001
        logger.warning("Service discovery failed (env URLs stand): %s", e)


async def _silent_turn_pipeline(
    utterance: np.ndarray, session: MediaStreamSession
) -> np.ndarray | None:
    """Placeholder until the live whisper→LLM→TTS chain is wired."""
    logger.info(
        "Turn pipeline placeholder: %d samples from session %s (no reply)",
        len(utterance), session.session_id,
    )
    return None


def create_app(
    config: GatewayConfig | None = None,
    *,
    turn_pipeline: Callable[..., Awaitable[np.ndarray | None]] | None = None,
    vad_config: VadConfig | None = None,
) -> FastAPI:
    cfg = config or GatewayConfig()
    app = FastAPI(title=SERVICE_NAME)

    app.state.config = cfg
    app.state.token_registry = SessionTokenRegistry()
    app.state.provider = TwilioProvider(
        cfg.twilio_account_sid, cfg.twilio_auth_token, cfg.twilio_from_number
    )
    app.state.turn_pipeline = turn_pipeline or _silent_turn_pipeline
    app.state.vad_config = vad_config or VadConfig(threshold_rms=cfg.vad_rms)
    app.state.active_sessions = {}
    # Live-call runtimes, owned by the dial worker (services/dial_worker.py):
    # keyed by session_id; the media WS composes its session from the runtime
    # (pipeline, recorder, disclosure) when one exists.
    app.state.call_runtimes = {}
    app.state.dial_worker = None
    app.state.dial_worker_task = None

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "active_calls": len(app.state.active_sessions),
            "signature_validation": cfg.signature_validation_enabled,
        }

    @app.websocket("/media/{token}")
    async def media_ws(ws: WebSocket, token: str):
        # 1. Single-use token claim — BEFORE accept. A close() without
        # accept rejects the handshake (403 at the HTTP layer).
        pending = app.state.token_registry.claim(token)
        if pending is None:
            logger.warning("Rejected media WS with unknown/replayed token")
            await ws.close(code=4401)
            return

        # 2. Twilio signature (scheme-fixed) — also before accept.
        if cfg.signature_validation_enabled:
            request_url = (
                cfg.public_url + ws.url.path if cfg.public_url else str(ws.url)
            )
            signature = ws.headers.get("x-twilio-signature")
            if not validate_ws_signature(cfg.twilio_auth_token, request_url, signature):
                logger.warning(
                    "Rejected media WS for session %s: bad X-Twilio-Signature",
                    pending.session_id,
                )
                await ws.close(code=4401)
                return

        await ws.accept()
        # A dial-worker runtime supplies the live pipeline + recorder +
        # disclosure; without one (tests, injected pipelines) the app-level
        # pipeline stands in and there is nothing extra to wire.
        runtime = app.state.call_runtimes.get(pending.session_id)
        session = MediaStreamSession(
            ws=ws,
            provider=app.state.provider,
            vad=RmsVad(app.state.vad_config),
            turn_pipeline=runtime.pipeline if runtime else app.state.turn_pipeline,
            pending=pending,
            recorder=runtime.recorder if runtime else None,
            on_stream_start=runtime.on_stream_start if runtime else None,
        )
        app.state.active_sessions[pending.session_id] = session
        try:
            await session.run()
        finally:
            app.state.active_sessions.pop(pending.session_id, None)
            if runtime is not None:
                runtime.stream_done.set()

    # ------------------------------------------------------------ internal API

    @app.post("/internal/call/{session_id}/escalation-answer")
    async def escalation_answer(
        session_id: str,
        body: EscalationAnswerBody,
        _app_id: str = Depends(require_app_auth),
    ):
        """CC forwards the user's answer into the live call's open window."""
        runtime = app.state.call_runtimes.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="No active call session")
        if not runtime.escalation.deliver(body.answer):
            raise HTTPException(
                status_code=409, detail="No escalation is waiting for an answer"
            )
        return {"status": "delivered"}

    @app.post("/internal/call/{session_id}/cancel")
    async def cancel_call(
        session_id: str,
        _app_id: str = Depends(require_app_auth),
    ):
        """CC-initiated termination (reaper, gate toggled off, user cancel)."""
        runtime = app.state.call_runtimes.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="No active call session")
        media = app.state.active_sessions.get(session_id)
        if media is not None:
            media.request_hangup()
        if runtime.call_sid:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as http:
                try:
                    await app.state.provider.end_call(runtime.call_sid, http)
                except httpx.HTTPError as e:
                    logger.error("REST hangup failed for %s: %s", session_id, e)
        return {"status": "cancelling"}

    @app.post("/internal/lookup/line-type")
    async def line_type(
        body: LineTypeBody,
        _app_id: str = Depends(require_app_auth),
    ):
        """Twilio Lookup proxy for CC's resolve step (creds stay here)."""
        import httpx

        async with httpx.AsyncClient() as http:
            result = await lookup_line_type(
                body.number,
                account_sid=cfg.twilio_account_sid,
                auth_token=cfg.twilio_auth_token,
                http=http,
            )
        return {"number": body.number, "line_type": result}

    # ------------------------------------------------------------ lifecycle

    @app.on_event("startup")
    async def _startup() -> None:
        _setup_jarvis_logging()
        _discover_service_urls(cfg)
        if not cfg.run_dial_worker:
            logger.info("Dial worker disabled (RUN_DIAL_WORKER=false)")
            return
        try:
            import redis  # deferred: chat-only deployments run without it

            # socket_timeout must exceed the worker's BLPOP block (5 s) or
            # every idle pop dies with "Timeout reading from socket".
            redis_client = redis.Redis.from_url(
                cfg.redis_url, socket_timeout=15, socket_connect_timeout=5
            )
        except Exception as e:  # noqa: BLE001 — no Redis ⇒ no dialing, app still serves
            logger.warning("Redis unavailable (%s) — dial worker not started", e)
            return
        worker = DialWorker(
            cfg,
            provider=app.state.provider,
            token_registry=app.state.token_registry,
            dial_queue=DialQueue(redis_client),
            call_runtimes=app.state.call_runtimes,
        )
        app.state.dial_worker = worker
        app.state.dial_worker_task = asyncio.create_task(worker.run_forever())
        logger.info("Dial worker started (queue=phone:dial)")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        worker = app.state.dial_worker
        task = app.state.dial_worker_task
        if worker is not None:
            worker.stop()
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    return app


app = create_app()
