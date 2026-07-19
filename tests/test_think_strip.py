"""Streaming think-block stripping — never speak reasoning."""

from llm.think_strip import ThinkStripper


def strip_all(deltas):
    s = ThinkStripper()
    out = "".join(s.feed(d) for d in deltas)
    return out + s.flush()


def test_passthrough_without_think():
    assert strip_all(["Hello ", "there."]) == "Hello there."


def test_whole_block_in_one_delta():
    assert strip_all(["<think>reasoning</think>Hi."]) == "Hi."


def test_block_split_across_deltas():
    assert strip_all(["<thi", "nk>secret ", "stuff</th", "ink>Hello."]) == "Hello."


def test_text_before_and_after_block():
    assert strip_all(["Sure. <think>hmm</think> Yes."]) == "Sure.  Yes."


def test_multiple_blocks():
    assert strip_all(["<think>a</think>One.<think>b</think>Two."]) == "One.Two."


def test_unclosed_block_never_leaks():
    # Stream ends mid-think: nothing after <think> may be emitted.
    assert strip_all(["Okay. <think>never say this"]) == "Okay. "


def test_partial_open_tag_that_never_completes_is_text():
    # "<thin" held back until flush proves it wasn't a tag.
    assert strip_all(["a <thin", "g> b"]) == "a <thing> b"


def test_lone_angle_bracket_passthrough():
    assert strip_all(["5 < 7 and 9 > 3."]) == "5 < 7 and 9 > 3."


def test_incremental_emission_not_all_at_flush():
    s = ThinkStripper()
    # Clean text must flow immediately, not be buffered to the end.
    assert s.feed("Hello there, ") == "Hello there, "
    assert s.feed("friend.") == "friend."
    assert s.flush() == ""
