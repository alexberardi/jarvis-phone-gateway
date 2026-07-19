"""Fake-Twilio WS fixture: a scripted media exchange against the real app.

Plays the Twilio client role over the actual websocket endpoint — real
provider parsing, real VAD, real token/signature gates — with a stubbed
turn pipeline (PRD test strategy: "provider serializer against a
fake-Twilio WS fixture").
"""

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from audio.mulaw import ulaw_decode, ulaw_encode
from audio.vad import VadConfig
from config import GatewayConfig
from main import create_app
from telephony.twilio_provider import compute_signature

LOUD = np.full(160, 3000, dtype=np.int16)
QUIET = np.full(160, 30, dtype=np.int16)

# Small hangover so tests need few frames: 100 ms = 5 quiet frames.
TEST_VAD = VadConfig(threshold_rms=250.0, hangover_ms=100, start_frames=3,
                     preroll_frames=5, max_utterance_s=2.0)


def media_msg(pcm: np.ndarray) -> dict:
    return {
        "event": "media",
        "media": {"payload": base64.b64encode(ulaw_encode(pcm)).decode()},
    }


def start_msg(session_id: str, call_sid: str = "CA1") -> dict:
    return {
        "event": "start",
        "start": {
            "streamSid": "MZ1",
            "callSid": call_sid,
            "customParameters": {"session_id": session_id},
        },
    }


def build_app(**kwargs):
    cfg = GatewayConfig()
    cfg.twilio_auth_token = ""  # signature validation off unless a test enables it
    replies = kwargs.pop("replies", None)
    seen_utterances: list[np.ndarray] = []

    async def stub_pipeline(utterance, session):
        seen_utterances.append(utterance)
        if replies:
            return replies.pop(0)
        return None

    app = create_app(cfg, turn_pipeline=stub_pipeline, vad_config=TEST_VAD, **kwargs)
    return app, seen_utterances


class TestMediaExchange:
    def test_full_turn_roundtrip(self):
        reply_pcm = np.full(400, 2000, dtype=np.int16)
        app, seen = build_app(replies=[reply_pcm])
        registry = app.state.token_registry
        token = registry.issue("sess-1")
        registry.bind_call_sid(token, "CA1")

        client = TestClient(app)
        with client.websocket_connect(f"/media/{token}") as ws:
            ws.send_text(json.dumps(start_msg("sess-1")))
            # Speak: 5 loud frames, then hangover silence endpoints the turn.
            for _ in range(5):
                ws.send_text(json.dumps(media_msg(LOUD)))
            for _ in range(TEST_VAD.hangover_frames):
                ws.send_text(json.dumps(media_msg(QUIET)))

            # Reply: 400 samples -> 160/160/80 mu-law frames + one mark.
            out = [ws.receive_json() for _ in range(4)]
            ws.send_text(json.dumps({"event": "stop"}))

        media_out = [m for m in out if m["event"] == "media"]
        marks = [m for m in out if m["event"] == "mark"]
        assert len(media_out) == 3
        assert len(marks) == 1
        assert all(m["streamSid"] == "MZ1" for m in media_out)
        # The audio that went out decodes back to (approximately) the reply.
        decoded = np.concatenate([
            ulaw_decode(base64.b64decode(m["media"]["payload"])) for m in media_out
        ])
        assert len(decoded) == 400
        assert abs(int(decoded[0]) - 2000) < 128

        # The pipeline saw one utterance including pre-roll + speech.
        assert len(seen) == 1
        assert len(seen[0]) >= 5 * 160

    def test_health_counts_active_calls(self):
        app, _ = build_app()
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["active_calls"] == 0


class TestWsGates:
    def test_unknown_token_rejected(self):
        app, _ = build_app()
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/media/bogus-token"):
                pass

    def test_token_single_use(self):
        app, _ = build_app()
        token = app.state.token_registry.issue("sess-1")
        client = TestClient(app)
        with client.websocket_connect(f"/media/{token}") as ws:
            ws.send_text(json.dumps({"event": "stop"}))
        # Replay of the same token must be rejected.
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/media/{token}"):
                pass

    def test_wrong_session_binding_closes(self):
        app, seen = build_app()
        registry = app.state.token_registry
        token = registry.issue("sess-1")
        client = TestClient(app)
        with client.websocket_connect(f"/media/{token}") as ws:
            ws.send_text(json.dumps(start_msg("EVIL-session")))
            with pytest.raises(WebSocketDisconnect):
                # Server closes; next receive raises.
                ws.send_text(json.dumps(media_msg(LOUD)))
                ws.receive_json()
        assert seen == []

    def test_wrong_callsid_binding_closes(self):
        app, _ = build_app()
        registry = app.state.token_registry
        token = registry.issue("sess-1")
        registry.bind_call_sid(token, "CA-expected")
        client = TestClient(app)
        with client.websocket_connect(f"/media/{token}") as ws:
            ws.send_text(json.dumps(start_msg("sess-1", call_sid="CA-forged")))
            with pytest.raises(WebSocketDisconnect):
                ws.send_text(json.dumps(media_msg(LOUD)))
                ws.receive_json()


class TestSignatureGate:
    def _signed_app(self):
        cfg = GatewayConfig()
        cfg.twilio_auth_token = "auth-tok"
        cfg.public_url = "https://calls.example.com"

        async def stub_pipeline(utterance, session):
            return None

        return create_app(cfg, turn_pipeline=stub_pipeline, vad_config=TEST_VAD)

    def test_valid_wss_signature_accepted(self):
        app = self._signed_app()
        token = app.state.token_registry.issue("sess-1")
        # Twilio signs the wss URL it was given in the TwiML.
        sig = compute_signature("auth-tok", f"wss://calls.example.com/media/{token}")
        client = TestClient(app)
        with client.websocket_connect(
            f"/media/{token}", headers={"X-Twilio-Signature": sig}
        ) as ws:
            ws.send_text(json.dumps({"event": "stop"}))

    def test_missing_or_bad_signature_rejected(self):
        app = self._signed_app()
        token = app.state.token_registry.issue("sess-1")
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/media/{token}"):
                pass
        # Bad signature on a freshly issued token also rejected.
        token2 = app.state.token_registry.issue("sess-2")
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/media/{token2}", headers={"X-Twilio-Signature": "forged"}
            ):
                pass
