"""Inbound /internal/* endpoints: app-auth gate + escalation/cancel/lookup."""

import asyncio

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import services.app_auth as app_auth
from config import GatewayConfig
from main import create_app
from services.app_auth import require_app_auth
from services.dial_worker import CallRuntime
from services.escalation import EscalationWindow


def make_app(**cfg_overrides):
    cfg = GatewayConfig()
    cfg.auth_url = "http://auth.test"
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)

    async def stub_pipeline(utterance, session):
        return None

    return create_app(cfg, turn_pipeline=stub_pipeline)


def allow_auth(app):
    async def ok():
        return "test-app"

    app.dependency_overrides[require_app_auth] = ok


def make_runtime(session_id="sess-1", call_sid=None):
    return CallRuntime(
        session_id=session_id,
        session={"id": session_id},
        pipeline=None,
        escalation=EscalationWindow(timeout_s=5.0),
        recorder=None,
        call_sid=call_sid,
    )


class TestAppAuthDependency:
    def setup_method(self):
        app_auth.clear_cache()

    def test_missing_credentials_401(self):
        app = make_app()
        client = TestClient(app)
        r = client.post("/internal/call/x/cancel")
        assert r.status_code == 401

    def test_valid_credentials_pass_and_cache(self, monkeypatch):
        calls = {"n": 0}
        RealClient = httpx.AsyncClient

        def handler(request):
            calls["n"] += 1
            assert request.headers["x-jarvis-app-id"] == "cc"
            return httpx.Response(200, json={"status": "ok"})

        def client_factory(*a, **kw):
            return RealClient(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(app_auth.httpx, "AsyncClient", client_factory)
        app = make_app()
        client = TestClient(app)
        headers = {"X-Jarvis-App-Id": "cc", "X-Jarvis-App-Key": "key"}
        # Unknown session -> 404, but PAST the auth gate.
        assert client.post("/internal/call/x/cancel", headers=headers).status_code == 404
        assert client.post("/internal/call/x/cancel", headers=headers).status_code == 404
        assert calls["n"] == 1  # second request served from the cache

    def test_bad_credentials_401(self, monkeypatch):
        RealClient = httpx.AsyncClient

        def client_factory(*a, **kw):
            return RealClient(
                transport=httpx.MockTransport(lambda r: httpx.Response(401))
            )

        monkeypatch.setattr(app_auth.httpx, "AsyncClient", client_factory)
        app = make_app()
        r = TestClient(app).post(
            "/internal/call/x/cancel",
            headers={"X-Jarvis-App-Id": "evil", "X-Jarvis-App-Key": "nope"},
        )
        assert r.status_code == 401

    def test_auth_unreachable_fails_closed_503(self, monkeypatch):
        RealClient = httpx.AsyncClient

        def client_factory(*a, **kw):
            def boom(request):
                raise httpx.ConnectError("auth down")

            return RealClient(transport=httpx.MockTransport(boom))

        monkeypatch.setattr(app_auth.httpx, "AsyncClient", client_factory)
        app = make_app()
        r = TestClient(app).post(
            "/internal/call/x/cancel",
            headers={"X-Jarvis-App-Id": "cc", "X-Jarvis-App-Key": "key"},
        )
        assert r.status_code == 503


class TestEscalationAnswerEndpoint:
    def test_delivers_into_open_window(self):
        app = make_app()
        allow_auth(app)
        runtime = make_runtime()
        app.state.call_runtimes["sess-1"] = runtime
        runtime.escalation.open()

        client = TestClient(app)
        r = client.post(
            "/internal/call/sess-1/escalation-answer", json={"answer": "6:30 works"}
        )
        assert r.status_code == 200

        async def collect():
            return await runtime.escalation.wait()

        assert asyncio.run(collect()) == "6:30 works"

    def test_no_active_session_404(self):
        app = make_app()
        allow_auth(app)
        r = TestClient(app).post(
            "/internal/call/ghost/escalation-answer", json={"answer": "hi"}
        )
        assert r.status_code == 404

    def test_no_open_window_409(self):
        app = make_app()
        allow_auth(app)
        app.state.call_runtimes["sess-1"] = make_runtime()
        r = TestClient(app).post(
            "/internal/call/sess-1/escalation-answer", json={"answer": "hi"}
        )
        assert r.status_code == 409


class TestCancelEndpoint:
    def test_cancel_requests_hangup_and_ends_twilio_leg(self, monkeypatch):
        app = make_app()
        allow_auth(app)
        runtime = make_runtime(call_sid="CA-9")
        app.state.call_runtimes["sess-1"] = runtime

        class FakeMedia:
            hangup_requested = False

            def request_hangup(self):
                self.hangup_requested = True

        media = FakeMedia()
        app.state.active_sessions["sess-1"] = media

        ended = []

        async def fake_end_call(call_sid, http):
            ended.append(call_sid)

        monkeypatch.setattr(app.state.provider, "end_call", fake_end_call)

        r = TestClient(app).post("/internal/call/sess-1/cancel")
        assert r.status_code == 200
        assert media.hangup_requested
        assert ended == ["CA-9"]

    def test_unknown_session_404(self):
        app = make_app()
        allow_auth(app)
        assert TestClient(app).post("/internal/call/ghost/cancel").status_code == 404


class TestLineTypeEndpoint:
    def test_returns_unknown_without_creds(self):
        # No Twilio creds configured -> lookup short-circuits to "unknown".
        app = make_app()
        allow_auth(app)
        r = TestClient(app).post(
            "/internal/lookup/line-type", json={"number": "+19082781811"}
        )
        assert r.status_code == 200
        assert r.json() == {"number": "+19082781811", "line_type": "unknown"}
