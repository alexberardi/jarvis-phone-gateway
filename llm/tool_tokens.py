"""Streaming parser for the call brain's text-token tool protocol.

PRD decision 6: v1 tool calling is plain text tokens in the reply stream —
``[HANGUP]``, ``[ESCALATE: question]``, ``[OUTCOME: facts]``, and
``[DTMF: digits]`` (reserved, wired P2) — because llm-proxy's streaming path
silently drops native tools. The P0 spike proved ``[HANGUP]`` works.

The parser is stream-safe: feed deltas, receive (speakable_text, events).
A ``[`` that could still grow into a token is held back until a ``]``
arrives or the prefix stops matching any token name; anything that is not a
recognized token is released verbatim (the model saying "[sic]" must be
spoken, not eaten). Unclosed candidates are bounded (payload cap) so a
runaway bracket can't buffer forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Longest payload we will buffer while waiting for a closing bracket.
_MAX_TOKEN_LEN = 512

_TOKEN_RE = re.compile(
    r"^\[(?:(HANGUP)|(ESCALATE|OUTCOME|DTMF):\s*(.*))\]$",
    re.DOTALL,
)

# Bracket openings that could still become a token, e.g. "[HAN", "[ESCALATE: par...".
_PREFIXES = ("HANGUP]", "ESCALATE:", "OUTCOME:", "DTMF:")


@dataclass
class Hangup:
    pass


@dataclass
class Escalate:
    question: str


@dataclass
class Outcome:
    facts: str


@dataclass
class Dtmf:
    digits: str


ToolEvent = Hangup | Escalate | Outcome | Dtmf

_EVENT_FOR = {"ESCALATE": Escalate, "OUTCOME": Outcome, "DTMF": Dtmf}


def _could_become_token(candidate: str) -> bool:
    """candidate starts after '[' and has no ']' yet — worth holding?"""
    if len(candidate) > _MAX_TOKEN_LEN:
        return False
    for prefix in _PREFIXES:
        if candidate.startswith(prefix[: len(candidate)]) or (
            candidate.startswith(prefix)
        ):
            return True
    return False


def _parse_complete(token_body: str) -> ToolEvent | None:
    m = _TOKEN_RE.match(f"[{token_body}]")
    if not m:
        return None
    if m.group(1):
        return Hangup()
    return _EVENT_FOR[m.group(2)](m.group(3).strip())


class ToolTokenParser:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> tuple[str, list[ToolEvent]]:
        self._buf += delta
        text_parts: list[str] = []
        events: list[ToolEvent] = []
        while self._buf:
            i = self._buf.find("[")
            if i == -1:
                text_parts.append(self._buf)
                self._buf = ""
                break
            text_parts.append(self._buf[:i])
            rest = self._buf[i + 1:]
            j = rest.find("]")
            if j == -1:
                if _could_become_token(rest):
                    self._buf = self._buf[i:]  # hold and wait for more
                    break
                # Cannot be a token — release the '[' and keep scanning.
                text_parts.append("[")
                self._buf = rest
                continue
            body = rest[:j]
            event = _parse_complete(body)
            if event is not None:
                events.append(event)
            else:
                text_parts.append(f"[{body}]")
            self._buf = rest[j + 1:]
        return "".join(text_parts), events

    def flush(self) -> str:
        """End of stream: an unfinished candidate is plain text after all."""
        out, self._buf = self._buf, ""
        return out
