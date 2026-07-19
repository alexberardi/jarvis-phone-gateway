"""Gateway half of the CC↔gateway session contract (PRD Interfaces)."""

import json

import httpx
import pytest

from services.session_client import SessionClient, SessionEventError

client = SessionClient("http://cc.test", "gateway-app", "gateway-key")


def recording_http(responses):
    """MockTransport that records requests and pops canned responses."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


@pytest.mark.asyncio
async def test_get_session():
    http, seen = recording_http([httpx.Response(200, json={"state": "confirmed"})])
    async with http:
        session = await client.get_session("s-1", http=http)
    assert session == {"state": "confirmed"}
    assert seen[0].url.path == "/internal/phone/sessions/s-1"
    assert seen[0].headers["X-Jarvis-App-Id"] == "gateway-app"


@pytest.mark.asyncio
async def test_claim_dial_success():
    http, seen = recording_http([httpx.Response(200, json={"state": "dialing"})])
    async with http:
        assert await client.claim_for_dial("s-1", "http://worker-3:7713", http=http)
    body = json.loads(seen[0].read())
    assert body == {"type": "claim_dial", "worker_url": "http://worker-3:7713"}
    assert seen[0].url.path == "/internal/phone/sessions/s-1/events"


@pytest.mark.asyncio
async def test_claim_dial_conflict_means_do_not_dial():
    # 409 = not confirmed / already claimed — the worker DROPS the job.
    http, _ = recording_http([httpx.Response(409, json={"detail": "not confirmed"})])
    async with http:
        assert not await client.claim_for_dial("s-1", "http://w", http=http)


@pytest.mark.asyncio
async def test_claim_dial_unexpected_status_raises():
    http, _ = recording_http([httpx.Response(500)])
    async with http:
        with pytest.raises(SessionEventError):
            await client.claim_for_dial("s-1", "http://w", http=http)


@pytest.mark.asyncio
async def test_turn_event_and_heartbeat_shapes():
    http, seen = recording_http([httpx.Response(200), httpx.Response(200)])
    async with http:
        await client.turn_event(
            "s-1", {"transcript": "hi", "stt_ms": 60}, http=http
        )
        await client.heartbeat("s-1", http=http)
    assert json.loads(seen[0].read()) == {
        "type": "turn", "turn": {"transcript": "hi", "stt_ms": 60},
    }
    assert json.loads(seen[1].read()) == {"type": "heartbeat"}


@pytest.mark.asyncio
async def test_outcome_event_carries_audio_key():
    http, seen = recording_http([httpx.Response(200)])
    async with http:
        await client.outcome_event(
            "s-1", {"result": "booked"}, audio_key="phone-calls/h/s-1.wav", http=http
        )
    body = json.loads(seen[0].read())
    assert body["type"] == "outcome"
    assert body["audio_key"] == "phone-calls/h/s-1.wav"


@pytest.mark.asyncio
async def test_state_event_failure_raises():
    http, _ = recording_http([httpx.Response(503)])
    async with http:
        with pytest.raises(SessionEventError):
            await client.state_event("s-1", "in_call", http=http)
