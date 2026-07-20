"""Live turn pipeline: STT→LLM→TTS chain with tool events + escalation.

Hermetic: whisper/TTS are monkeypatched module functions, the LLM is a
scripted fake client, CC is a recording stub. What's real: think-strip,
tool-token parsing, sentence regrouping, escalation state machine, timing
capture, message-history bookkeeping.
"""

import asyncio

import numpy as np
import pytest

import services.turn_pipeline as tp
from llm.client import LlmStreamError, TurnTimeout
from services.escalation import EscalationWindow
from services.prompt import ESCALATION_FALLBACK_LINE, HOLD_LINE, TURN_FAILURE_LINE
from services.turn_pipeline import LiveTurnPipeline

SESSION = {
    "id": "sess-1",
    "initiator_name": "Alex",
    "goal": "Book a table",
    "details": "Party of 4, Friday 7pm.",
    "household_id": "hh-1",
}

UTTERANCE = np.full(1600, 1500, dtype=np.int16)


class FakeLlm:
    """Scripted delta streams, one list per call."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls: list[list[dict]] = []

    async def stream_deltas(self, messages, *, http, turn_timeout_s=20.0, **kw):
        self.calls.append([dict(m) for m in messages])
        if not self.scripts:
            return
        script = self.scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        for delta in script:
            yield delta


class FakeSessionClient:
    def __init__(self):
        self.events: list[tuple] = []

    async def turn_event(self, session_id, turn, *, http):
        self.events.append(("turn", turn))

    async def escalation_event(self, session_id, question, *, http):
        self.events.append(("escalation", question))


class FakeMediaSession:
    def __init__(self):
        self.hangup_requested = False
        self.spoken: list[np.ndarray] = []
        self.idle_armed = False

    def request_hangup(self):
        self.hangup_requested = True

    def arm_idle_hangup(self, seconds=None):
        self.idle_armed = True

    async def speak(self, pcm):
        self.spoken.append(pcm)


@pytest.fixture
def stub_services(monkeypatch):
    """Patch whisper + TTS; return the recording dicts."""
    seen = {"transcripts": "hello, who is this?", "synth": []}

    async def fake_transcribe(pcm, url, http):
        return seen["transcripts"]

    async def fake_synthesize(text, url, http):
        seen["synth"].append(text)
        return np.full(80, 900, dtype=np.int16), 12.0

    monkeypatch.setattr(tp, "transcribe", fake_transcribe)
    monkeypatch.setattr(tp, "synthesize", fake_synthesize)
    return seen


def make_pipeline(llm, session_client=None, escalation=None, **kw):
    return LiveTurnPipeline(
        session=SESSION,
        whisper_url="http://w",
        tts_url="http://t",
        llm=llm,
        http=None,  # hermetic — nothing touches it after the patches
        session_client=session_client,
        escalation=escalation,
        **kw,
    )


async def drain(pipeline):
    """Wait out fire-and-forget turn-event tasks."""
    if pipeline._bg_tasks:
        await asyncio.gather(*pipeline._bg_tasks, return_exceptions=True)


class TestHappyTurn:
    @pytest.mark.asyncio
    async def test_full_turn_strips_thinks_and_tokens(self, stub_services):
        llm = FakeLlm([
            ["<think>internal reasoning</think>", "We'd love a table.",
             " See you Friday! ", "[OUTCOME: booked Friday 7pm]", "[HANGUP]"],
        ])
        cc = FakeSessionClient()
        pipe = make_pipeline(llm, cc)
        media = FakeMediaSession()

        pcm = await pipe(UTTERANCE, media)
        await drain(pipe)

        assert pcm is not None and len(pcm)
        spoken = " ".join(stub_services["synth"])
        assert "internal reasoning" not in spoken
        assert "[OUTCOME" not in spoken and "[HANGUP]" not in spoken
        assert "We'd love a table." in spoken
        assert pipe.outcome_facts == ["booked Friday 7pm"]
        # The FIRST outcome never ends the call: the model recording a result
        # and hanging up in one breath is exactly how two live calls ended
        # with nothing actually booked. The goodbye still plays and the idle
        # timer bounds the wait.
        assert not media.hangup_requested
        assert media.idle_armed
        assert "hangup_deferred" in pipe.turn_records[-1].events

        # History: system, disclosure, user, assistant.
        roles = [m["role"] for m in pipe.messages]
        assert roles == ["system", "assistant", "user", "assistant"]
        # The model's copy carries the per-turn thinking directive; the
        # transcript below keeps the caller's words verbatim.
        assert pipe.messages[2]["content"].startswith("hello, who is this?")
        assert pipe.messages[2]["content"].endswith("/no_think")

        # Turn event carried transcript + timings.
        kinds = [k for k, _ in cc.events]
        assert "turn" in kinds
        turn = next(e for k, e in cc.events if k == "turn")
        assert turn["heard"] == "hello, who is this?"
        assert turn["timings"]["stt_ms"] >= 0
        assert "hangup" in turn["events"] and "outcome" in turn["events"]

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_llm(self, stub_services):
        stub_services["transcripts"] = ""
        llm = FakeLlm([["should never run"]])
        pipe = make_pipeline(llm)
        assert await pipe(UTTERANCE, FakeMediaSession()) is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_redacted_turn_event_has_no_transcript(self, stub_services):
        llm = FakeLlm([["Sure thing."]])
        cc = FakeSessionClient()
        pipe = make_pipeline(llm, cc, redact_transcript=True)
        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)
        turn = next(e for k, e in cc.events if k == "turn")
        assert "heard" not in turn and "said" not in turn
        assert turn["timings"]["total_ms"] >= 0


class TestFailurePaths:
    @pytest.mark.asyncio
    async def test_turn_timeout_speaks_fallback(self, stub_services):
        llm = FakeLlm([TurnTimeout("req-1", 20.0)])
        pipe = make_pipeline(llm)
        media = FakeMediaSession()
        pcm = await pipe(UTTERANCE, media)
        await drain(pipe)
        assert pcm is not None and len(pcm)
        assert stub_services["synth"] == [TURN_FAILURE_LINE]
        assert not media.hangup_requested  # stay on the line

    @pytest.mark.asyncio
    async def test_llm_error_speaks_fallback(self, stub_services):
        llm = FakeLlm([LlmStreamError("model_not_loaded")])
        pipe = make_pipeline(llm)
        pcm = await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)
        assert pcm is not None
        assert stub_services["synth"] == [TURN_FAILURE_LINE]

    @pytest.mark.asyncio
    async def test_tts_failure_does_not_kill_turn(self, stub_services, monkeypatch):
        async def broken_synthesize(text, url, http):
            raise RuntimeError("tts down")

        monkeypatch.setattr(tp, "synthesize", broken_synthesize)
        llm = FakeLlm([["Hello there."]])
        pipe = make_pipeline(llm)
        # No audio, but no exception either — the call stays alive.
        assert await pipe(UTTERANCE, FakeMediaSession()) is None
        await drain(pipe)


class TestEscalation:
    @pytest.mark.asyncio
    async def test_answered_escalation_continues_the_call(self, stub_services):
        llm = FakeLlm([
            ["[ESCALATE: only 6:30 is available — is that OK?]"],
            ["Great, 6:30 works — see you then."],  # continuation stream
        ])
        cc = FakeSessionClient()
        window = EscalationWindow(timeout_s=5.0)
        pipe = make_pipeline(llm, cc, escalation=window)
        media = FakeMediaSession()

        async def user_answers():
            for _ in range(200):
                if window.is_open:
                    break
                await asyncio.sleep(0.005)
            assert window.deliver("6:30 works")

        answer_task = asyncio.create_task(user_answers())
        pcm = await pipe(UTTERANCE, media)
        await answer_task
        await drain(pipe)

        assert pcm is not None and len(pcm)
        # Hold line spoken immediately (model produced no prose).
        assert HOLD_LINE in stub_services["synth"]
        assert media.spoken, "hold line must go out before the wait"
        # CC got the escalation question.
        assert ("escalation", "only 6:30 is available — is that OK?") in cc.events
        # The user's answer entered the history for the continuation stream.
        continuation_history = llm.calls[1]
        assert any("6:30 works" in m["content"] for m in continuation_history)
        assert not media.hangup_requested
        assert not pipe.escalation_unanswered

    @pytest.mark.asyncio
    async def test_unanswered_escalation_ends_gracefully(self, stub_services):
        llm = FakeLlm([["[ESCALATE: cash only — OK?]"]])
        cc = FakeSessionClient()
        window = EscalationWindow(timeout_s=0.02)
        pipe = make_pipeline(llm, cc, escalation=window)
        media = FakeMediaSession()

        pcm = await pipe(UTTERANCE, media)
        await drain(pipe)

        assert pcm is not None and len(pcm)
        assert ESCALATION_FALLBACK_LINE in stub_services["synth"]
        assert media.hangup_requested
        assert pipe.escalation_unanswered
        assert any("escalation unanswered" in f for f in pipe.outcome_facts)


class TestClosingSequence:
    """The two live failures of 2026-07-20, both in the call-closing path.

    Call 1 hung up before the appointment was confirmed; call 2 went silent
    on the turn right after the business accepted. Opposite ends of the same
    step, so they are pinned together.
    """

    @pytest.mark.asyncio
    async def test_empty_reply_speaks_instead_of_dead_air(self, stub_services):
        """An unclosed <think> block strips to nothing. The turn must still
        produce audio — 'never silence into the void' was only enforced on
        the exception path, so a successful-but-empty reply hung the call."""
        llm = FakeLlm([["<think>still reasoning when the stream ended"]])
        pipe = make_pipeline(llm, FakeSessionClient())
        media = FakeMediaSession()

        pcm = await pipe(UTTERANCE, media)
        await drain(pipe)

        assert pcm is not None and len(pcm), "empty reply produced no audio"
        spoken = " ".join(stub_services["synth"])
        assert "reasoning" not in spoken, "think content leaked into the call"
        assert spoken.strip(), "nothing was said"
        assert "empty_reply" in pipe.turn_records[-1].events

    @pytest.mark.asyncio
    async def test_empty_reply_with_hangup_still_says_goodbye(self, stub_services):
        """A call must never terminate on silence either.

        Note the token is OUTSIDE any think block on purpose: anything inside
        an unclosed <think> is discarded wholesale, tokens included, so a
        [HANGUP] in there never reaches the parser.
        """
        llm = FakeLlm([["[HANGUP]"]])
        pipe = make_pipeline(llm, FakeSessionClient())
        media = FakeMediaSession()

        pcm = await pipe(UTTERANCE, media)
        await drain(pipe)

        assert pcm is not None and len(pcm)
        assert media.hangup_requested, "hangup should still be honoured"
        assert " ".join(stub_services["synth"]).strip()

    @pytest.mark.asyncio
    async def test_hangup_honoured_once_an_outcome_already_exists(self, stub_services):
        """The gate costs exactly one extra exchange — a later hangup ends
        the call normally, so a confirmed booking is not held hostage."""
        llm = FakeLlm([
            ["Great, I'll take Thursday at 2.", "[OUTCOME: booked Thursday 2pm]",
             "[HANGUP]"],
            ["Perfect, thank you — goodbye.", "[HANGUP]"],
        ])
        pipe = make_pipeline(llm, FakeSessionClient())
        media = FakeMediaSession()

        await pipe(UTTERANCE, media)          # deferred
        assert not media.hangup_requested

        await pipe(UTTERANCE, media)          # honoured
        await drain(pipe)
        assert media.hangup_requested
        assert "hangup_deferred" not in pipe.turn_records[-1].events

    @pytest.mark.asyncio
    async def test_hangup_without_outcome_is_not_deferred(self, stub_services):
        """'Please stop calling' must end the call immediately — the gate is
        about unconfirmed RESULTS, not about refusing to hang up."""
        llm = FakeLlm([["Sorry to bother you — goodbye.", "[HANGUP]"]])
        pipe = make_pipeline(llm, FakeSessionClient())
        media = FakeMediaSession()

        await pipe(UTTERANCE, media)
        await drain(pipe)

        assert media.hangup_requested
        assert not media.idle_armed


class TestCalleeClosedTheCall:
    """Live 2026-07-20 (pizza order): the shop said "Great. I'll see you in
    about 30 minutes. Thank you." twice, and both closing turns went wrong —
    the first drew "Sorry — could you repeat that?", the second had its
    correct goodbye+hangup deferred into an idle window. The shop hung up on
    us. A closing cue from THEM is the signal both paths were missing.
    """

    @pytest.mark.asyncio
    async def test_hangup_not_deferred_once_they_have_signed_off(self, stub_services):
        llm = FakeLlm([
            ["Goodbye! Have a great meal.", "[OUTCOME: ordered large pepperoni]",
             "[HANGUP]"],
        ])
        pipe = make_pipeline(llm, FakeSessionClient())
        media = FakeMediaSession()

        stub_services["transcripts"] = (
            "Great. I'll see you in about 30 minutes. Thank you."
        )
        await pipe(UTTERANCE, media)
        await drain(pipe)

        assert media.hangup_requested, "hung on after they said goodbye"
        assert "hangup_deferred" not in pipe.turn_records[-1].events
        assert not media.idle_armed

    @pytest.mark.asyncio
    async def test_still_deferred_when_they_only_offered(self, stub_services):
        """The guard must not swallow the original bug: an offer is not a
        confirmation, so that hangup is still held back."""
        llm = FakeLlm([
            ["Thursday at 2pm works, I'll book that.",
             "[OUTCOME: booked Thursday 2pm]", "[HANGUP]"],
        ])
        pipe = make_pipeline(llm, FakeSessionClient())
        media = FakeMediaSession()

        stub_services["transcripts"] = "Yeah, what about 2 p.m.?"
        await pipe(UTTERANCE, media)
        await drain(pipe)

        assert not media.hangup_requested
        assert "hangup_deferred" in pipe.turn_records[-1].events

    @pytest.mark.asyncio
    async def test_empty_reply_to_a_farewell_says_goodbye_and_ends(self, stub_services):
        llm = FakeLlm([["<think>unclosed reasoning"]])
        pipe = make_pipeline(llm, FakeSessionClient())
        media = FakeMediaSession()

        stub_services["transcripts"] = "Great. See you in 30 minutes. Thank you."
        await pipe(UTTERANCE, media)
        await drain(pipe)

        spoken = " ".join(stub_services["synth"])
        assert "repeat" not in spoken.lower(), (
            "asked them to repeat their own goodbye"
        )
        assert media.hangup_requested
        assert "closed_by_callee" in pipe.turn_records[-1].events

    @pytest.mark.asyncio
    async def test_empty_reply_mid_conversation_still_asks_again(self, stub_services):
        """Only a CLOSING turn converts an empty reply into a goodbye."""
        llm = FakeLlm([["<think>unclosed reasoning"]])
        pipe = make_pipeline(llm, FakeSessionClient())
        media = FakeMediaSession()

        stub_services["transcripts"] = "We have a table at seven or eight."
        await pipe(UTTERANCE, media)
        await drain(pipe)

        assert not media.hangup_requested
        assert "repeat" in " ".join(stub_services["synth"]).lower()
