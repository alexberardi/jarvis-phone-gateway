"""Streaming tool-token protocol parser: [HANGUP] / [ESCALATE] / [OUTCOME] / [DTMF]."""

from llm.tool_tokens import (
    Dtmf,
    Escalate,
    Hangup,
    Outcome,
    ToolTokenParser,
)


def parse_all(deltas):
    p = ToolTokenParser()
    text_parts, events = [], []
    for d in deltas:
        t, e = p.feed(d)
        text_parts.append(t)
        events.extend(e)
    text_parts.append(p.flush())
    return "".join(text_parts), events


def test_plain_text_passthrough():
    text, events = parse_all(["Good", "bye now."])
    assert text == "Goodbye now."
    assert events == []


def test_hangup_extracted_from_text():
    text, events = parse_all(["Goodbye! ", "[HANGUP]"])
    assert text == "Goodbye! "
    assert events == [Hangup()]


def test_hangup_split_across_deltas():
    text, events = parse_all(["Bye. [HA", "NG", "UP]"])
    assert text == "Bye. "
    assert events == [Hangup()]


def test_escalate_with_payload():
    text, events = parse_all(["One moment. [ESCALATE: only 6:30 available — ok?]"])
    assert text == "One moment. "
    assert events == [Escalate(question="only 6:30 available — ok?")]


def test_escalate_payload_split_across_deltas():
    _, events = parse_all(["[ESCALATE: can we ", "do 7pm instead?]"])
    assert events == [Escalate(question="can we do 7pm instead?")]


def test_outcome_and_dtmf():
    _, events = parse_all(["[OUTCOME: booked Friday 7pm, conf #A12]", "[DTMF: 1#]"])
    assert events == [Outcome(facts="booked Friday 7pm, conf #A12"), Dtmf(digits="1#")]


def test_unknown_bracket_content_is_spoken():
    text, events = parse_all(["The price [sic] is right."])
    assert text == "The price [sic] is right."
    assert events == []


def test_unknown_bracket_is_released_promptly_not_held():
    p = ToolTokenParser()
    # "[sic]" can never become a token — its text must not wait for flush.
    text, events = p.feed("a [sic] b")
    assert text == "a [sic] b"
    assert events == []


def test_unclosed_candidate_is_text_at_flush():
    text, events = parse_all(["Wait [ESCALATE: never closed"])
    assert text == "Wait [ESCALATE: never closed"
    assert events == []


def test_text_between_multiple_tokens():
    text, events = parse_all(["A [OUTCOME: x] B [HANGUP] C"])
    assert text == "A  B  C"
    assert events == [Outcome(facts="x"), Hangup()]


def test_lowercase_is_not_a_token():
    text, events = parse_all(["[hangup]"])
    assert text == "[hangup]"
    assert events == []


def test_runaway_bracket_does_not_buffer_forever():
    p = ToolTokenParser()
    text1, _ = p.feed("[ESCALATE: " + "x" * 600)
    # Once past the cap the candidate is released as text.
    assert "[ESCALATE: " in text1 + p.flush()
