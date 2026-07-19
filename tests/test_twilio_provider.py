"""Twilio provider: TwiML, WS wire format, signature scheme-fix, token binding."""

import base64
import json
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import pytest

from audio.mulaw import ulaw_encode
from telephony.provider import InboundAudio, MarkReceived, StreamStart, StreamStop
from telephony.twilio_provider import (
    PendingSession,
    SessionTokenRegistry,
    TwilioProvider,
    compute_signature,
    validate_stream_start,
    validate_ws_signature,
)

AUTH_TOKEN = "twilio-auth-token-secret"
provider = TwilioProvider("ACxxxx", AUTH_TOKEN, "+15550001111")


# ---------------------------------------------------------------- TwiML


class TestTwiml:
    def test_connect_stream_with_parameters(self):
        twiml = provider.build_stream_instructions(
            "wss://calls.example.com/media/tok123",
            {"session_id": "sess-1"},
        )
        root = ET.fromstring(twiml)
        stream = root.find("Connect/Stream")
        assert stream is not None
        assert stream.get("url") == "wss://calls.example.com/media/tok123"
        param = stream.find("Parameter")
        assert param.get("name") == "session_id"
        assert param.get("value") == "sess-1"

    def test_xml_escaping(self):
        twiml = provider.build_stream_instructions(
            "wss://x/media/t", {"goal": 'say "hi" & <bye>'}
        )
        # Must parse cleanly despite quotes/ampersands/angle brackets.
        root = ET.fromstring(twiml)
        assert root.find("Connect/Stream/Parameter").get("value") == 'say "hi" & <bye>'


# ---------------------------------------------------------------- WS parsing


class TestWsWireFormat:
    def test_parse_start(self):
        raw = json.dumps({
            "event": "start",
            "start": {
                "streamSid": "MZ123",
                "callSid": "CA456",
                "customParameters": {"session_id": "sess-1"},
            },
        })
        ev = provider.parse_ws_message(raw)
        assert isinstance(ev, StreamStart)
        assert ev.stream_sid == "MZ123"
        assert ev.call_sid == "CA456"
        assert ev.custom_parameters == {"session_id": "sess-1"}

    def test_parse_media_decodes_mulaw(self):
        pcm = np.full(160, 1000, dtype=np.int16)
        raw = json.dumps({
            "event": "media",
            "media": {"payload": base64.b64encode(ulaw_encode(pcm)).decode()},
        })
        ev = provider.parse_ws_message(raw)
        assert isinstance(ev, InboundAudio)
        assert len(ev.pcm) == 160
        assert abs(int(ev.pcm[0]) - 1000) < 64  # mu-law quantization

    def test_parse_mark_stop_and_ignorables(self):
        assert isinstance(
            provider.parse_ws_message('{"event": "mark", "mark": {"name": "t1"}}'),
            MarkReceived,
        )
        assert isinstance(provider.parse_ws_message('{"event": "stop"}'), StreamStop)
        assert provider.parse_ws_message('{"event": "connected"}') is None
        assert provider.parse_ws_message("not json") is None

    def test_media_messages_frame_split(self):
        # 400 samples -> 400 mu-law bytes -> 3 frames (160/160/80).
        pcm = np.zeros(400, dtype=np.int16)
        msgs = list(provider.media_messages("MZ1", pcm))
        assert len(msgs) == 3
        assert all(m["event"] == "media" and m["streamSid"] == "MZ1" for m in msgs)
        sizes = [len(base64.b64decode(m["media"]["payload"])) for m in msgs]
        assert sizes == [160, 160, 80]

    def test_mark_and_clear_messages(self):
        assert provider.mark_message("MZ1", "t3")["mark"]["name"] == "t3"
        assert provider.clear_message("MZ1") == {"event": "clear", "streamSid": "MZ1"}


# ---------------------------------------------------------------- signature


class TestSignature:
    WSS_URL = "wss://calls.example.com/media/tok123"
    HTTPS_URL = "https://calls.example.com/media/tok123"

    def test_naive_https_validation_fails_without_scheme_fix(self):
        # Twilio signs the wss URL; a validator that only checks the
        # https-reconstructed URL never matches. This is the documented trap.
        sig_from_twilio = compute_signature(AUTH_TOKEN, self.WSS_URL)
        assert compute_signature(AUTH_TOKEN, self.HTTPS_URL) != sig_from_twilio

    def test_scheme_fix_accepts_wss_signature_on_https_url(self):
        sig_from_twilio = compute_signature(AUTH_TOKEN, self.WSS_URL)
        assert validate_ws_signature(AUTH_TOKEN, self.HTTPS_URL, sig_from_twilio)

    def test_scheme_fix_accepts_exact_match_too(self):
        sig = compute_signature(AUTH_TOKEN, self.WSS_URL)
        assert validate_ws_signature(AUTH_TOKEN, self.WSS_URL, sig)

    def test_wrong_token_rejected(self):
        sig = compute_signature("other-token", self.WSS_URL)
        assert not validate_ws_signature(AUTH_TOKEN, self.HTTPS_URL, sig)

    def test_wrong_url_rejected(self):
        sig = compute_signature(AUTH_TOKEN, "wss://calls.example.com/media/OTHER")
        assert not validate_ws_signature(AUTH_TOKEN, self.HTTPS_URL, sig)

    def test_missing_signature_rejected(self):
        assert not validate_ws_signature(AUTH_TOKEN, self.HTTPS_URL, None)
        assert not validate_ws_signature(AUTH_TOKEN, self.HTTPS_URL, "")


# ---------------------------------------------------------------- token registry


class TestTokenRegistry:
    def test_issue_claim_single_use(self):
        reg = SessionTokenRegistry()
        token = reg.issue("sess-1")
        pending = reg.claim(token)
        assert pending is not None
        assert pending.session_id == "sess-1"
        # Second claim (replay / duplicate stream) is rejected.
        assert reg.claim(token) is None

    def test_unknown_token_rejected(self):
        assert SessionTokenRegistry().claim("bogus") is None

    def test_bind_call_sid(self):
        reg = SessionTokenRegistry()
        token = reg.issue("sess-1")
        reg.bind_call_sid(token, "CA789")
        assert reg.claim(token).call_sid == "CA789"

    def test_tokens_are_unique_and_long(self):
        reg = SessionTokenRegistry()
        tokens = {reg.issue(f"s{i}") for i in range(50)}
        assert len(tokens) == 50
        assert all(len(t) >= 32 for t in tokens)


class TestStreamStartBinding:
    def test_matching_session_and_callsid(self):
        start = StreamStart("MZ1", "CA1", {"session_id": "sess-1"})
        assert validate_stream_start(start, PendingSession("sess-1", "CA1"))

    def test_wrong_session_param_rejected(self):
        start = StreamStart("MZ1", "CA1", {"session_id": "EVIL"})
        assert not validate_stream_start(start, PendingSession("sess-1", "CA1"))

    def test_wrong_callsid_rejected(self):
        start = StreamStart("MZ1", "CA-forged", {"session_id": "sess-1"})
        assert not validate_stream_start(start, PendingSession("sess-1", "CA1"))

    def test_unbound_callsid_allows_match_on_session_alone(self):
        start = StreamStart("MZ1", "CA1", {"session_id": "sess-1"})
        assert validate_stream_start(start, PendingSession("sess-1", None))


# ---------------------------------------------------------------- REST


class TestRest:
    @pytest.mark.asyncio
    async def test_start_call_posts_twiml_and_returns_sid(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = request.read().decode()
            return httpx.Response(201, json={"sid": "CA999"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            sid = await provider.start_call(
                to_number="+15557654321",
                instructions="<Response/>",
                http=http,
            )
        assert sid == "CA999"
        assert "/Accounts/ACxxxx/Calls.json" in seen["url"]
        assert "To=%2B15557654321" in seen["body"]

    @pytest.mark.asyncio
    async def test_end_call_sets_completed(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = request.read().decode()
            return httpx.Response(200, json={"sid": "CA999"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await provider.end_call("CA999", http)
        assert "/Calls/CA999.json" in seen["url"]
        assert "Status=completed" in seen["body"]
