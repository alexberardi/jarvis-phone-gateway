"""Speech formatting: identifiers are dictated char-by-char; ordinary numbers are not.

The failure this guards against: the TTS reads a long digit run as a cardinal and
runs a spelled name together, so a caller can't copy down a member ID or phone
number. All data here is fictional (no real PII).
"""

import pytest

from services.speech_format import format_for_speech as f


class TestIdentifiersGetDictated:
    @pytest.mark.parametrize(
        "text, expect",
        [
            # letters get phonetic cues; digits are read one at a time
            ("Your member ID is ZQ-0001234.", "Z as in Zebra, Q as in Queen, 0, 0, 0, 1, 2, 3, 4"),
            ("The policy number is 7654321.", "7, 6, 5, 4, 3, 2, 1"),
            ("Call back at 5551234567.", "5, 5, 5, 1, 2, 3, 4, 5, 6, 7"),
            ("Reach us at 555-123-4567.", "5, 5, 5, 1, 2, 3, 4, 5, 6, 7"),
            ("It is spelled S-M-I-T-H.", "S as in Sam, M as in Mary, I as in Igloo, T as in Tom, H as in Henry"),
            ("Confirmation is ABC12345.", "A as in Apple, B as in Boy, C as in Cat, 1, 2, 3, 4, 5"),
        ],
    )
    def test_spelled_out(self, text, expect):
        assert expect in f(text)

    def test_trailing_punctuation_preserved(self):
        assert f("It is ZQ-0001234.").endswith("4.")

    def test_letters_are_phonetic_and_upper_cased(self):
        assert "Z as in Zebra, Q as in Queen, 0" in f("id zq0001234 please")


class TestOrdinaryNumbersUntouched:
    @pytest.mark.parametrize(
        "text",
        [
            "The appointment is Tuesday at 4pm.",
            "It was booked for 2026.",
            "Party of 4 at 7:30 please.",
            "That will be $45 at pickup.",
            "I live at 123 Main Street.",
            "See you in 30 minutes.",
            "Order number 12 is ready.",  # 2 digits, not an ID
        ],
    )
    def test_unchanged(self, text):
        assert f(text) == text


class TestEdges:
    def test_empty(self):
        assert f("") == ""

    def test_idempotent_on_already_spaced_digits(self):
        # The model may already read a number out; leave its single digits alone.
        assert f("It is 9 9 1 2 3 4 5.") == "It is 9 9 1 2 3 4 5."

    def test_plain_prose_never_changes(self):
        s = "Hi, I'm calling to confirm the appointment for Jordan."
        assert f(s) == s
