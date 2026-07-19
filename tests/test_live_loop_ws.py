"""Full live-loop turn over the real WS endpoint.

The fake-Twilio client drives one scripted call through create_app's actual
websocket route with a REAL LiveTurnPipeline — real VAD, real think-strip /
tool-token / sentence machinery, real recorder and disclosure hook — with
whisper/TTS/LLM stubbed at the network seam. This is the closest CI gets
to the P0 spike's live call.
"""

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import services.turn_pipeline as tp
from audio.mulaw import ulaw_decode
from config import GatewayConfig
from llm.client import LlmProxyStreamClient
from main import create_app
from services.dial_worker import CallRuntime
from services.escalation import EscalationWindow
from services.recording import CallRecorder
from services.turn_pipeline import LiveTurnPipeline
from tests.test_media_ws import LOUD, QUIET, TEST_VAD, media_msg, start_msg

SESSION = {
    "id": "sess-live",
    "initiator_name": "Alex",
    "goal": "Book a table",
    "details": "Party of 4, Friday 7pm.",
    "household_id": "hh-1",
}


class ScriptedLlm(LlmProxyStreamClient):
    def __init__(self, scripts):
        super().__init__("http://llm.test", "app", "key")
        self.scripts = list(scripts)

    async def stream_deltas(self, messages, *, http, **kw):
        if self.scripts:
            for delta in self.scripts.pop(0):
                yield delta


@pytest.fixture
def voice_seams(monkeypatch):
    seen = {"synth": []}

    async def fake_transcribe(pcm, url, http):
        return "do you have a table friday?"

    async def fake_synthesize(text, url, http):
        seen["synth"].append(text)
        # 200 samples of recognizable audio per sentence.
        return np.full(200, 1200, dtype=np.int16), 8.0

    monkeypatch.setattr(tp, "transcribe", fake_transcribe)
    monkeypatch.setattr(tp, "synthesize", fake_synthesize)
    return seen


def test_disclosure_then_full_turn_with_recording(voice_seams):
    cfg = GatewayConfig()
    cfg.twilio_auth_token = ""  # signature gate off for the fixture

    app = create_app(cfg, vad_config=TEST_VAD)

    llm = ScriptedLlm([
        ["<think>plan</think>", "Yes we do!", " [OUTCOME: table available]",
         " Goodbye! [HANGUP]"],
    ])
    pipeline = LiveTurnPipeline(
        session=SESSION,
        whisper_url="http://w",
        tts_url="http://t",
        llm=llm,
        http=None,
        session_client=None,
        escalation=EscalationWindow(),
    )
    runtime = CallRuntime(
        session_id="sess-live",
        session=SESSION,
        pipeline=pipeline,
        escalation=pipeline.escalation,
        recorder=CallRecorder(),
        disclosure_pcm=np.full(160, 3000, dtype=np.int16),
    )
    app.state.call_runtimes["sess-live"] = runtime

    token = app.state.token_registry.issue("sess-live")
    app.state.token_registry.bind_call_sid(token, "CA-live")

    client = TestClient(app)
    outbound = []
    with client.websocket_connect(f"/media/{token}") as ws:
        ws.send_text(json.dumps(start_msg("sess-live", call_sid="CA-live")))
        # Disclosure goes out immediately on validated stream-start.
        first = ws.receive_json()
        assert first["event"] == "media"
        outbound.append(first)
        second = ws.receive_json()  # disclosure mark
        assert second["event"] == "mark"

        # Callee speaks; hangover silence endpoints the utterance.
        for _ in range(5):
            ws.send_text(json.dumps(media_msg(LOUD)))
        for _ in range(TEST_VAD.hangover_frames):
            ws.send_text(json.dumps(media_msg(QUIET)))

        # Reply audio: 2 sentences x 200 samples -> 400 samples -> 3 frames + mark.
        got_mark = False
        while not got_mark:
            msg = ws.receive_json()
            if msg["event"] == "mark":
                got_mark = True
            else:
                outbound.append(msg)
        # [HANGUP] closes the WS from the server side after the reply.

    # Spoken text: think + tokens stripped, both sentences synthesized.
    assert voice_seams["synth"] == ["Yes we do!", "Goodbye!"]
    # Outcome fact captured for the wrapup.
    assert pipeline.outcome_facts == ["table available"]
    # stream_done fired for the dial worker.
    assert runtime.stream_done.is_set()

    # The recording holds callee audio AND both agent bursts (disclosure+reply).
    mixed = runtime.recorder.mix()
    assert len(mixed) > 0
    reply_samples = np.concatenate([
        ulaw_decode(base64.b64decode(m["media"]["payload"])) for m in outbound
    ])
    assert len(reply_samples) >= 160 + 400  # disclosure + two sentences

    # Turn record carries the spike timing vocabulary.
    rec = pipeline.turn_records[0]
    assert rec.heard == "do you have a table friday?"
    assert rec.stt_ms >= 0 and rec.total_ms > 0
    assert "hangup" in rec.events and "outcome" in rec.events
