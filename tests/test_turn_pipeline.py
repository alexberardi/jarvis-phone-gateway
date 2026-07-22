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
    """Scripted delta streams, one list per call.

    ``verdict``/``complete_error`` drive the guard's ask-classifier, which
    rides the same client; ``completions`` records every call it makes, so a
    test can assert the guard stayed off the wire when it had nothing to do.
    """

    def __init__(self, scripts, verdict="NONE", complete_error=None):
        self.scripts = list(scripts)
        self.calls: list[list[dict]] = []
        self.verdict = verdict
        self.complete_error = complete_error
        self.completions: list[list[dict]] = []

    async def complete(self, messages, *, http=None, **kw):
        self.completions.append([dict(m) for m in messages])
        if self.complete_error:
            raise self.complete_error
        return self.verdict

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
    def __init__(self, time_verdict=None, check_error=None):
        self.events: list[tuple] = []
        self.time_verdict = time_verdict
        self.check_error = check_error
        self.checked: list[str] = []

    async def turn_event(self, session_id, turn, *, http):
        self.events.append(("turn", turn))

    async def escalation_event(self, session_id, question, *, http):
        self.events.append(("escalation", question))

    async def check_time(self, session_id, utterance, *, http):
        self.checked.append(utterance)
        if self.check_error:
            raise self.check_error
        return self.time_verdict or {"time_detected": False, "available": None}


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


class TestSpokenOutputGuard:
    """End of the guard's chain: what actually reaches TTS.

    The unit tests prove detection and intent in isolation; these prove the
    pipeline honours the verdict, and that a withheld sentence never becomes
    dead air or a silent hole in the transcript.
    """

    GUARDED = {
        **SESSION,
        "restricted_details": [
            {
                "key": "callback_number",
                "label": "Callback number",
                "value": "(908) 555-0147",
            }
        ],
    }

    def _pipeline(self, llm, cc=None):
        return LiveTurnPipeline(
            session=self.GUARDED,
            whisper_url="http://w",
            tts_url="http://t",
            llm=llm,
            http=None,
            session_client=cc or FakeSessionClient(),
        )

    @pytest.mark.asyncio
    async def test_volunteered_number_never_reaches_tts(self, stub_services):
        """The failure this exists for: a rule forbade it and the model did
        it anyway. The rest of the reply still goes out."""
        llm = FakeLlm([
            ["We can do Friday at seven. ", "My callback number is 908-555-0147."]
        ], verdict="NONE")
        pipe = self._pipeline(llm)

        stub_services["transcripts"] = "How many in your party?"
        await pipe(UTTERANCE, media := FakeMediaSession())
        await drain(pipe)

        spoken = " ".join(stub_services["synth"])
        assert "555" not in spoken and "0147" not in spoken
        assert "Friday at seven" in spoken
        assert "guard_suppressed" in pipe.turn_records[-1].events
        assert not media.hangup_requested

    @pytest.mark.asyncio
    async def test_the_transcript_records_only_what_was_said(self, stub_services):
        """A suppressed sentence must leave the model's own history too —
        otherwise it believes it already gave the number and reasons from
        that on every later turn."""
        llm = FakeLlm([
            ["Sure thing. ", "You can reach me at 908-555-0147."]
        ], verdict="NONE")
        pipe = self._pipeline(llm)

        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        history = " ".join(
            m["content"] for m in pipe.messages if m["role"] == "assistant"
        )
        assert "0147" not in history
        assert "0147" not in pipe.turn_records[-1].said

    @pytest.mark.asyncio
    async def test_asked_for_it_lets_it_through(self, stub_services):
        """The tier is give-if-asked, not never. Blocking a genuine request
        would break the calls the feature exists to make."""
        llm = FakeLlm([
            ["Of course — it's 908-555-0147."]
        ], verdict="1")
        pipe = self._pipeline(llm)

        stub_services["transcripts"] = "What's a good callback number for you?"
        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert "908-555-0147" in " ".join(stub_services["synth"])
        assert "guard_suppressed" not in pipe.turn_records[-1].events

    @pytest.mark.asyncio
    async def test_classifier_outage_suppresses_rather_than_leaks(self, stub_services):
        llm = FakeLlm([
            ["It's 908-555-0147."]
        ], complete_error=RuntimeError("llm-proxy down"))
        pipe = self._pipeline(llm)

        stub_services["transcripts"] = "What's your callback number?"
        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert "0147" not in " ".join(stub_services["synth"])

    @pytest.mark.asyncio
    async def test_fully_suppressed_reply_does_not_ask_them_to_repeat(
        self, stub_services
    ):
        """"Could you repeat that?" would invite the same question and the
        same suppression — a loop, on a live call."""
        llm = FakeLlm([["My callback number is 908-555-0147."]], verdict="NONE")
        pipe = self._pipeline(llm)

        stub_services["transcripts"] = "And a number for you?"
        await pipe(UTTERANCE, media := FakeMediaSession())
        await drain(pipe)

        spoken = " ".join(stub_services["synth"])
        assert "0147" not in spoken
        assert "repeat" not in spoken.lower()
        assert "not able to share" in spoken.lower()
        assert not media.hangup_requested

    @pytest.mark.asyncio
    async def test_a_call_with_no_restricted_details_never_calls_the_classifier(
        self, stub_services
    ):
        """The common case must cost nothing — no extra round trip per turn."""
        llm = FakeLlm([["We're all set for Friday."]], verdict="1")
        pipe = make_pipeline(llm, FakeSessionClient())  # plain SESSION

        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert llm.completions == []

    @pytest.mark.asyncio
    async def test_classifier_is_called_once_for_two_leaky_sentences(
        self, stub_services
    ):
        llm = FakeLlm([
            ["Reach me at 908-555-0147. ", "Again, that's 9085550147."]
        ], verdict="NONE")
        pipe = self._pipeline(llm)

        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert len(llm.completions) == 1


class TestSchedulingCheck:
    """Deterministic availability verdict injected per turn.

    The model can't do interval math under /no_think (it false-declines valid
    times and false-accepts invalid ones); CC does the check and the gateway
    states the verdict. These prove the verdict reaches the model, only on
    scheduling turns, and never blocks a call.
    """

    SCHED = {**SESSION, "constraints": "Acceptable times: Thu 9am-8pm"}

    def _pipeline(self, llm, cc):
        return LiveTurnPipeline(
            session=self.SCHED,
            whisper_url="http://w",
            tts_url="http://t",
            llm=llm,
            http=None,
            session_client=cc,
        )

    def _note_in_last_call(self, llm):
        # FakeLlm.stream_deltas records each message list it was given.
        msgs = llm.calls[-1]
        return next(
            (m["content"] for m in msgs if "Scheduling check" in m.get("content", "")),
            None,
        )

    @pytest.mark.asyncio
    async def test_available_verdict_reaches_the_model(self, stub_services):
        llm = FakeLlm([["Yes, Thursday at noon works."]])
        cc = FakeSessionClient(time_verdict={
            "time_detected": True, "available": True,
            "proposed_label": "Thursday at noon", "acceptable_summary": None,
        })
        pipe = self._pipeline(llm, cc)
        stub_services["transcripts"] = "Can you do Thursday at noon?"

        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert cc.checked == ["Can you do Thursday at noon?"]
        note = self._note_in_last_call(llm)
        assert note and "confirmed OPEN" in note and "Thursday at noon" in note

    @pytest.mark.asyncio
    async def test_unavailable_verdict_tells_the_model_to_decline(self, stub_services):
        llm = FakeLlm([["That time isn't available."]])
        cc = FakeSessionClient(time_verdict={
            "time_detected": True, "available": False,
            "proposed_label": "Wednesday at 12",
            "acceptable_summary": "Acceptable times: Thu 9am-8pm",
        })
        pipe = self._pipeline(llm, cc)
        stub_services["transcripts"] = "How about Wednesday at 12?"

        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        note = self._note_in_last_call(llm)
        assert note and "NOT open" in note and "Thu 9am-8pm" in note

    @pytest.mark.asyncio
    async def test_the_note_does_not_persist_into_history(self, stub_services):
        """A verdict is guidance for THIS turn only — leaving it in history
        would give a later turn a stale answer and shift the cached prefix."""
        llm = FakeLlm([["Yes, that works."], ["Sure."]])
        cc = FakeSessionClient(time_verdict={
            "time_detected": True, "available": True,
            "proposed_label": "Thursday at noon", "acceptable_summary": None,
        })
        pipe = self._pipeline(llm, cc)
        stub_services["transcripts"] = "Thursday at noon?"
        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert not any("Scheduling check" in m.get("content", "") for m in pipe.messages)

    @pytest.mark.asyncio
    async def test_no_time_turn_makes_no_check(self, stub_services):
        llm = FakeLlm([["Sure."]])
        cc = FakeSessionClient()
        pipe = self._pipeline(llm, cc)
        stub_services["transcripts"] = "What's the address?"  # no day/time
        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert cc.checked == []  # gated out before any CC call

    @pytest.mark.asyncio
    async def test_non_scheduling_call_never_checks(self, stub_services):
        """A pizza order has no constraint envelope — the check is skipped
        even if the utterance mentions a number."""
        llm = FakeLlm([["Sure."]])
        cc = FakeSessionClient()
        pipe = LiveTurnPipeline(
            session=SESSION,  # no "constraints"
            whisper_url="http://w", tts_url="http://t", llm=llm, http=None,
            session_client=cc,
        )
        stub_services["transcripts"] = "I'll have 2 pizzas at 6."
        await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert cc.checked == []

    @pytest.mark.asyncio
    async def test_check_failure_fails_open(self, stub_services):
        """CC down must not stall or drop the turn — the model just carries it
        unaided, exactly as before this feature."""
        llm = FakeLlm([["Let me see."]])
        cc = FakeSessionClient(check_error=RuntimeError("CC down"))
        pipe = self._pipeline(llm, cc)
        stub_services["transcripts"] = "Thursday at noon?"

        pcm = await pipe(UTTERANCE, FakeMediaSession())
        await drain(pipe)

        assert pcm is not None  # turn completed
        assert self._note_in_last_call(llm) is None  # no verdict injected
