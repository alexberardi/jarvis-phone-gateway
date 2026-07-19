"""Dial worker: CAS-gated dialing, lifecycle, wrapup, failure honesty.

Hermetic: fake queue, recording SessionClient stub, stub provider, patched
TTS/upload/summary. What's real: the worker's ordering and drop rules —
the queue-is-transport invariant lives or dies here.
"""

import asyncio

import httpx
import numpy as np
import pytest

import services.dial_worker as dw
import services.turn_pipeline as tp
from config import GatewayConfig
from queues.dial_queue import DialJob
from services.dial_worker import DialWorker
from telephony.twilio_provider import SessionTokenRegistry

SESSION = {
    "id": "sess-1",
    "initiator_name": "Alex",
    "goal": "Book a table",
    "details": "Party of 4.",
    "household_id": "hh-1",
    "dialed_number": "+15555550123",
    "record_enabled": True,
    "max_call_seconds": 600,
}


class StubSessionClient:
    def __init__(self, session=None, claim=True, fetch_error=False):
        self.session = dict(session or SESSION)
        self.claim = claim
        self.fetch_error = fetch_error
        self.events: list[tuple] = []

    async def get_session(self, session_id, *, http):
        if self.fetch_error:
            raise httpx.ConnectError("cc down")
        return dict(self.session)

    async def claim_for_dial(self, session_id, worker_url, *, http):
        self.events.append(("claim", worker_url))
        return self.claim

    async def state_event(self, session_id, state, *, http, **extra):
        self.events.append(("state", state, extra))

    async def heartbeat(self, session_id, *, http):
        self.events.append(("heartbeat",))

    async def turn_event(self, session_id, turn, *, http):
        self.events.append(("turn", turn))

    async def escalation_event(self, session_id, question, *, http):
        self.events.append(("escalation", question))

    async def outcome_event(self, session_id, outcome, *, http, audio_key=None):
        self.events.append(("outcome", outcome, audio_key))

    def states(self):
        return [e[1] for e in self.events if e[0] == "state"]


class StubProvider:
    def __init__(self, fail_dial=False):
        self.fail_dial = fail_dial
        self.dials: list[tuple] = []
        self.ended: list[str] = []

    def build_stream_instructions(self, wss_url, params):
        return f"<Response url={wss_url!r} session={params.get('session_id')!r}/>"

    async def start_call(self, *, to_number, instructions, http):
        if self.fail_dial:
            raise RuntimeError("twilio 21211: invalid number")
        self.dials.append((to_number, instructions))
        return "CA-test-1"

    async def end_call(self, call_sid, http):
        self.ended.append(call_sid)


def make_worker(cc, provider=None, cfg=None):
    cfg = cfg or GatewayConfig()
    cfg.public_url = "https://calls.example.com"
    cfg._public_wss_url = "wss://calls.example.com"
    provider = provider or StubProvider()
    runtimes: dict = {}
    worker = DialWorker(
        cfg,
        provider=provider,
        token_registry=SessionTokenRegistry(),
        dial_queue=None,  # handle_job tests never pop
        call_runtimes=runtimes,
        session_client=cc,
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(500))
        ),
    )
    return worker, provider, runtimes


@pytest.fixture
def voice_stubs(monkeypatch):
    """Disclosure TTS + upload + summary all succeed."""

    async def fake_synthesize(text, url, http):
        return np.full(160, 700, dtype=np.int16), 5.0

    async def fake_upload(household_id, session_id, wav):
        return f"{household_id}/{session_id}.wav"

    async def fake_summary(self, pipeline, http):
        return "Booked Friday 7pm."

    monkeypatch.setattr(dw, "synthesize", fake_synthesize)
    monkeypatch.setattr(dw, "upload_recording", fake_upload)
    monkeypatch.setattr(DialWorker, "_summarize", fake_summary)
    monkeypatch.setattr(dw, "STREAM_START_TIMEOUT_S", 0.5)
    monkeypatch.setattr(dw, "HEARTBEAT_INTERVAL_S", 0.05)


def _finish_stream_when_started(runtimes: dict, delay: float = 0.05):
    """Background: simulate the media WS landing and later ending."""

    async def run():
        for _ in range(200):
            if runtimes:
                break
            await asyncio.sleep(0.005)
        runtime = next(iter(runtimes.values()))
        runtime.started.set()
        await asyncio.sleep(delay)
        runtime.stream_done.set()

    return asyncio.create_task(run())


class TestDropRules:
    @pytest.mark.asyncio
    async def test_unclaimable_session_never_dials(self, voice_stubs):
        cc = StubSessionClient(claim=False)
        worker, provider, _ = make_worker(cc)
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        assert provider.dials == []
        assert cc.states() == []  # dropped silently — CC said no

    @pytest.mark.asyncio
    async def test_session_fetch_failure_drops_job(self, voice_stubs):
        cc = StubSessionClient(fetch_error=True)
        worker, provider, _ = make_worker(cc)
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        assert provider.dials == []

    @pytest.mark.asyncio
    async def test_missing_number_fails_before_dial(self, voice_stubs):
        cc = StubSessionClient(session={**SESSION, "dialed_number": ""})
        worker, provider, _ = make_worker(cc)
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        assert provider.dials == []
        assert "failed" in cc.states()

    @pytest.mark.asyncio
    async def test_disclosure_synth_failure_blocks_dial(self, voice_stubs, monkeypatch):
        async def silent(text, url, http):
            return np.array([], dtype=np.int16), None

        monkeypatch.setattr(dw, "synthesize", silent)
        cc = StubSessionClient()
        worker, provider, _ = make_worker(cc)
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        # The disclosure is never skippable: no audio -> no call.
        assert provider.dials == []
        assert "failed" in cc.states()

    @pytest.mark.asyncio
    async def test_dial_failure_reports_failed(self, voice_stubs):
        cc = StubSessionClient()
        worker, provider, _ = make_worker(cc, provider=StubProvider(fail_dial=True))
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        assert "failed" in cc.states()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, voice_stubs):
        cc = StubSessionClient()
        worker, provider, runtimes = make_worker(cc)
        finisher = _finish_stream_when_started(runtimes)

        await worker.handle_job(DialJob("sess-1", "hh-1"))
        await finisher

        # Dialed the session's number with a bound TwiML.
        assert len(provider.dials) == 1
        number, twiml = provider.dials[0]
        assert number == "+15555550123"
        assert "wss://calls.example.com/media/" in twiml
        assert "sess-1" in twiml

        # Lifecycle order: claim -> in_call -> wrapup -> outcome -> done.
        assert cc.events[0][0] == "claim"
        states = cc.states()
        assert states.index("in_call") < states.index("wrapup")
        outcome = next(e for e in cc.events if e[0] == "outcome")
        assert outcome[1]["summary"] == "Booked Friday 7pm."
        assert outcome[1]["audio_available"] is True
        assert outcome[2] == "hh-1/sess-1.wav"
        assert states[-1] == "done"

        # Runtime cleaned up.
        assert runtimes == {}

    @pytest.mark.asyncio
    async def test_notice_off_skips_recorder_and_redacts(self, voice_stubs):
        cc = StubSessionClient(session={**SESSION, "record_enabled": False})
        worker, provider, runtimes = make_worker(cc)
        finisher = _finish_stream_when_started(runtimes)

        # Capture the runtime mid-flight to inspect wiring.
        captured = {}

        async def capture():
            for _ in range(200):
                if runtimes:
                    captured.update(runtimes)
                    return
                await asyncio.sleep(0.005)

        cap_task = asyncio.create_task(capture())
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        await finisher
        await cap_task

        runtime = captured["sess-1"]
        assert runtime.recorder is None
        assert runtime.pipeline.redact_transcript is True
        outcome = next(e for e in cc.events if e[0] == "outcome")
        assert outcome[1]["audio_available"] is False
        assert outcome[2] is None


class TestSupervision:
    @pytest.mark.asyncio
    async def test_no_stream_within_timeout_ends_call(self, voice_stubs):
        cc = StubSessionClient()
        worker, provider, _ = make_worker(cc)
        # Nothing ever sets runtime.started -> 0.5 s timeout fires.
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        assert provider.ended == ["CA-test-1"]
        assert "failed" in cc.states()

    @pytest.mark.asyncio
    async def test_max_call_seconds_hangs_up(self, voice_stubs, monkeypatch):
        monkeypatch.setattr(dw, "_HANGUP_GRACE_S", 0.05)
        cc = StubSessionClient(session={**SESSION, "max_call_seconds": 0.15})
        worker, provider, runtimes = make_worker(cc)

        async def start_but_never_end():
            for _ in range(200):
                if runtimes:
                    break
                await asyncio.sleep(0.005)
            next(iter(runtimes.values())).started.set()
            # stream_done never set by the "call" — the cap must end it.

        helper = asyncio.create_task(start_but_never_end())
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        await helper
        assert provider.ended == ["CA-test-1"]
        # The call still gets an honest wrapup + outcome.
        assert "wrapup" in cc.states()

    @pytest.mark.asyncio
    async def test_heartbeats_flow_during_long_streams(self, voice_stubs):
        cc = StubSessionClient()
        worker, provider, runtimes = make_worker(cc)
        finisher = _finish_stream_when_started(runtimes, delay=0.3)
        await worker.handle_job(DialJob("sess-1", "hh-1"))
        await finisher
        assert ("heartbeat",) in cc.events
