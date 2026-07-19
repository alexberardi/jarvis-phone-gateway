"""Streaming <think>-block stripper.

The live model is a thinking model (P0 failure ladder item 4): replies open
with ``<think>...</think>`` reasoning that must NEVER be spoken. ``/no_think``
reduces but does not eliminate it, so this filter is mandatory on the token
stream. Tags arrive split across deltas — a suffix that could still become a
tag is held back until disambiguated.
"""

from __future__ import annotations

_OPEN = "<think>"
_CLOSE = "</think>"


def _partial_suffix_len(text: str, tag: str) -> int:
    """Longest k < len(tag) such that text ends with tag[:k]."""
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class ThinkStripper:
    """Feed token deltas, receive speakable text with think blocks removed."""

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        while True:
            if self._inside:
                i = self._buf.find(_CLOSE)
                if i == -1:
                    # Discard think content but keep a tail that could be a
                    # partial close tag; the block itself never accumulates.
                    keep = _partial_suffix_len(self._buf, _CLOSE)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[i + len(_CLOSE):]
                self._inside = False
            else:
                i = self._buf.find(_OPEN)
                if i == -1:
                    keep = _partial_suffix_len(self._buf, _OPEN)
                    cut = len(self._buf) - keep
                    out.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(_OPEN):]
                self._inside = True
        return "".join(out)

    def flush(self) -> str:
        """End of stream: release held text. Inside an unclosed think block,
        nothing is released — think content never leaks."""
        if self._inside:
            self._buf = ""
            return ""
        out, self._buf = self._buf, ""
        return out
