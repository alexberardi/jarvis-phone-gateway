"""Format model text so the TTS speaks IDENTIFIERS the way a person dictates them.

Piper/Kokoro read a long digit run as a CARDINAL ("9912345" → "nine million nine
hundred twelve thousand…") and run spelled-out letters together — wrong for member
IDs, policy/confirmation numbers, phone numbers, and the name spellings a caller
must hear digit-by-digit / letter-by-letter. This normalizes ONLY the text handed
to TTS (the transcript keeps the human-readable "XZ-9912345"); it separates the
characters of an identifier with commas so the engine voices each one, with a pause.

Deliberately conservative — it must NOT touch ordinary numbers a caller reads as
quantities: a year ("2026"), a time ("7:30", "at 4"), a party size ("party of 4"),
a short address number, or a price ("$45"). So a BARE digit run is only spelled when
it is 5+ digits long (IDs/phones are; years/times/quantities are not) and not a
dollar amount.
"""

from __future__ import annotations

import re

# Characters that can wrap a token in a sentence but aren't part of the identifier.
_EDGE = ".,;:!?\"')(“”‘’"

# Spell LETTERS phonetically so the listener can't confuse similar-sounding ones over
# a phone (B/P, M/N, S/F) — the standard way people spell a name or ID by phone. Live
# (ddcf2c2f): Jarvis spelled 'B-E-R-A-R-D-I', the pharmacy read it back as 'P-E-R-A-R-D-I'
# and Jarvis didn't catch it. Plain-word cues ('as in Boy'), not NATO, which a layperson
# may not know.
_PHONETIC = {
    "A": "Apple", "B": "Boy", "C": "Cat", "D": "Dog", "E": "Echo", "F": "Frank",
    "G": "George", "H": "Henry", "I": "Igloo", "J": "July", "K": "King", "L": "Lion",
    "M": "Mary", "N": "Nancy", "O": "Ocean", "P": "Paul", "Q": "Queen", "R": "Robert",
    "S": "Sam", "T": "Tom", "U": "Umbrella", "V": "Victor", "W": "William",
    "X": "X-ray", "Y": "Yellow", "Z": "Zebra",
}


def _spell(token: str) -> str:
    """Voice an identifier one character at a time: letters phonetically, digits as
    single digits — 'ZQ-0012' → 'Z as in Zebra, Q as in Queen, 0, 0, 1, 2'.

    Separators inside the token (-, ., space) are dropped; commas give the listener a
    beat between each character."""
    parts: list[str] = []
    for c in token:
        if c.isalpha():
            parts.append(f"{c.upper()} as in {_PHONETIC[c.upper()]}")
        elif c.isdigit():
            parts.append(c)
    return ", ".join(parts)


def _is_spelled_letters(token: str) -> bool:
    """Single letters joined by hyphens or dots — the model's spelled-name form
    ('S-M-I-T-H', 'S.M.I.T.H'). Space-joined can't be one token."""
    return bool(re.fullmatch(r"[A-Za-z](?:[-.][A-Za-z]){2,}", token))


def _is_identifier(token: str) -> bool:
    """Should this token be dictated character-by-character?"""
    if _is_spelled_letters(token):
        return True
    # Only alphanumerics plus internal - . separators can be an identifier.
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.\-]*[A-Za-z0-9])?", token):
        return False
    digits = sum(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    has_sep = any(c in ".-" for c in token)
    if has_alpha:
        # An alphanumeric mix (member ID, confirmation code): 2+ digits looks like an ID,
        # not a word with a stray number.
        return digits >= 2
    # Pure number. A separated one (phone: 555-123-4567) needs 7–15 digits; a bare run
    # needs 5+ (an ID/phone — never a year/time/quantity). Decimals (45.99) are excluded
    # by the digit count, never a phone.
    if has_sep:
        return 7 <= digits <= 15
    return digits >= 5


def format_for_speech(text: str) -> str:
    """Rewrite identifiers in ``text`` for digit-/letter-by-letter TTS. Idempotent on
    ordinary prose. Applied to the TTS input only, never to the recorded transcript."""
    if not text:
        return text
    out: list[str] = []
    for token in re.split(r"(\s+)", text):
        if not token or token.isspace():
            out.append(token)
            continue
        # A leading '$' means a price — leave it as a cardinal amount.
        if token.lstrip(_EDGE).startswith("$"):
            out.append(token)
            continue
        lead = token[: len(token) - len(token.lstrip(_EDGE))]
        trail = token[len(token.rstrip(_EDGE)):]
        core = token[len(lead): len(token) - len(trail)]
        out.append(lead + _spell(core) + trail if _is_identifier(core) else token)
    return "".join(out)
