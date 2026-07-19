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

import logging
from typing import Awaitable, Callable

import numpy as np
from fastapi import FastAPI, WebSocket

from audio.vad import RmsVad, VadConfig
from config import GatewayConfig
from services.media_stream import MediaStreamSession
from telephony.twilio_provider import (
    SessionTokenRegistry,
    TwilioProvider,
    validate_ws_signature,
)

logger = logging.getLogger("uvicorn")

SERVICE_NAME = "jarvis-phone-gateway"


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
        session = MediaStreamSession(
            ws=ws,
            provider=app.state.provider,
            vad=RmsVad(app.state.vad_config),
            turn_pipeline=app.state.turn_pipeline,
            pending=pending,
        )
        app.state.active_sessions[pending.session_id] = session
        try:
            await session.run()
        finally:
            app.state.active_sessions.pop(pending.session_id, None)

    return app


app = create_app()
