"""The spoken-output guard: detection, intent, and which way each fails.

The property under test throughout is asymmetric. A false positive costs one
suppressed sentence; a false negative is the disclosure the guard exists to
prevent. So every ambiguous case below asserts suppression.
"""

import pytest

from services.spoken_guard import (
    RestrictedField,
    SpokenOutputGuard,
    find_restricted,
    mentions,
    parse_restricted,
    _parse_verdict,
)

MEMBER_ID = RestrictedField("insurance_member_id", "Insurance member ID", "XZ-9912345")
CALLBACK = RestrictedField("callback_number", "Callback number", "(908) 555-0147")
ADDRESS = RestrictedField(
    "address", "Address", "742 Evergreen Ave, Springfield, IL 62704"
)


class TestSnapshotParsing:
    def test_reads_ccs_shape(self):
        fields = parse_restricted(
            [{"key": "address", "label": "Address", "value": "742 Evergreen Ave"}]
        )
        assert fields == [RestrictedField("address", "Address", "742 Evergreen Ave")]

    @pytest.mark.parametrize(
        "raw", [None, "", {}, [], ["nope"], [{"key": "a"}], [{"value": "v"}]]
    )
    def test_unusable_payloads_yield_nothing(self, raw):
        assert parse_restricted(raw) == []

    def test_missing_label_falls_back_to_key(self):
        fields = parse_restricted([{"key": "gate_code", "value": "4417"}])
        assert fields[0].label == "gate_code"


class TestDetection:
    """Deterministic, and matched by PARTS — the model rewrites what it says."""

    def test_exact_value(self):
        assert mentions("My member ID is XZ-9912345.", MEMBER_ID.value)

    def test_reformatted_number(self):
        """The single most likely real leak: same digits, different shape."""
        assert mentions("You can reach me at 908-555-0147.", CALLBACK.value)
        assert mentions("It's 9085550147.", CALLBACK.value)

    def test_digits_spoken_as_words(self):
        assert mentions(
            "That's X Z nine nine one two three four five.", MEMBER_ID.value
        )

    def test_partially_rewritten_address(self):
        """'Ave' -> 'Avenue' defeats whole-string matching, and it fails in
        the direction that leaks. The house number catches it."""
        assert mentions("We're at 742 Evergreen Avenue.", ADDRESS.value)

    def test_street_name_without_the_number(self):
        assert mentions("It's the place on Evergreen in Springfield.", ADDRESS.value)

    def test_ordinary_speech_is_not_a_match(self):
        assert not mentions("Sure, that works for us.", MEMBER_ID.value)
        assert not mentions("Oh, great — see you at six.", CALLBACK.value)

    def test_a_lone_oh_is_not_a_zero(self):
        """'oh' folds to 0 only inside a spoken digit run; otherwise ordinary
        speech would start colliding with numbers."""
        assert not mentions("Oh, sure.", RestrictedField("x", "X", "0"). value)

    def test_one_shared_town_name_is_not_enough(self):
        """A local call mentions the town constantly."""
        assert not mentions("Are you the Springfield location?", ADDRESS.value)

    def test_short_values_are_not_substring_matched(self):
        assert not mentions("Table for 42 at seven.", "42")

    def test_find_restricted_reports_every_field_disclosed(self):
        hit = find_restricted(
            "I'm at 742 Evergreen Avenue, reach me at 908-555-0147.",
            [MEMBER_ID, CALLBACK, ADDRESS],
        )
        assert {f.key for f in hit} == {"callback_number", "address"}


class FakeLlm:
    """Records what the classifier was shown — the security assertion."""

    def __init__(self, reply="NONE", error=None):
        self.reply = reply
        self.error = error
        self.calls: list[list[dict]] = []

    async def complete(self, messages, *, http=None, **kw):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply


class TestVerdictParsing:
    @pytest.mark.parametrize(
        "reply,expected",
        [("1", {1}), ("1, 2", {1, 2}), ("2,1", {1, 2}), (" 1 ", {1})],
    )
    def test_numbers(self, reply, expected):
        assert _parse_verdict(reply, 2) == expected

    @pytest.mark.parametrize(
        "reply", ["NONE", "none", "", "   ", None, "I cannot help with that",
                  "0", "3", "99", "the caller asked for none of these"]
    )
    def test_anything_else_means_not_asked(self, reply):
        assert _parse_verdict(reply, 2) == set()

    def test_reasoning_is_stripped_before_digits_are_read(self):
        """The guard's own chain of thought must not be able to authorize a
        disclosure. The live model is a thinking model, and a <think> block
        weighing "is it item 1 or item 2?" is full of digits — reading it
        would answer "both", which is the guard failing OPEN. Regression for
        a real defect found against the box model, 2026-07-20."""
        reply = (
            "<think>They might mean item 1, or possibly item 2. Item 2 is the "
            "insurance one, so 2 is wrong here.</think>\nNONE"
        )
        assert _parse_verdict(reply, 2) == set()

    def test_the_verdict_after_reasoning_still_counts(self):
        reply = "<think>Item 2 is the member ID, which is what they want.</think>\n2"
        assert _parse_verdict(reply, 2) == {2}

    def test_truncated_reasoning_yields_nothing(self):
        """At a small token budget the think block consumed the entire reply
        and nothing else came back. That must read as "not asked", never as
        an answer."""
        assert _parse_verdict("<think>Okay, let's see. The user asked", 2) == set()


class TestAskClassification:
    async def test_asked_unlocks_only_that_field(self):
        llm = FakeLlm(reply="2")
        guard = SpokenOutputGuard(llm)
        keys = await guard.asked_keys(
            "what's your member ID?", [CALLBACK, MEMBER_ID], http=None
        )
        assert keys == {"insurance_member_id"}

    async def test_the_classifier_is_never_shown_a_value(self):
        """The whole reason the guard is safe to expose to a hostile callee:
        talking it into a 'yes' reveals nothing, because the prompt only ever
        contained labels."""
        llm = FakeLlm(reply="1")
        await SpokenOutputGuard(llm).asked_keys(
            "what is it?", [MEMBER_ID, CALLBACK, ADDRESS], http=None
        )
        prompt = str(llm.calls[0])

        for field in (MEMBER_ID, CALLBACK, ADDRESS):
            assert field.value not in prompt
            assert field.label in prompt

    async def test_the_no_think_directive_rides_along(self):
        """Without it the live model burns its whole budget reasoning and the
        verdict comes back empty — the guard degrading into "suppress
        everything", which looks like it works until a real call needs a real
        answer."""
        llm = FakeLlm(reply="1")
        await SpokenOutputGuard(llm).asked_keys("what's your ID?", [MEMBER_ID], http=None)

        assert "/no_think" in str(llm.calls[0])

    async def test_transport_failure_fails_closed(self):
        guard = SpokenOutputGuard(FakeLlm(error=RuntimeError("llm down")))
        assert await guard.asked_keys("what's your ID?", [MEMBER_ID], http=None) == set()

    async def test_timeout_fails_closed(self):
        guard = SpokenOutputGuard(FakeLlm(error=TimeoutError()))
        assert await guard.asked_keys("what's your ID?", [MEMBER_ID], http=None) == set()

    async def test_no_restricted_fields_makes_no_call(self):
        """The common call has nothing to guard and must pay nothing."""
        llm = FakeLlm(reply="1")
        assert await SpokenOutputGuard(llm).asked_keys("anything", [], http=None) == set()
        assert llm.calls == []

    async def test_empty_utterance_makes_no_call(self):
        llm = FakeLlm(reply="1")
        assert await SpokenOutputGuard(llm).asked_keys("  ", [MEMBER_ID], http=None) == set()
        assert llm.calls == []
