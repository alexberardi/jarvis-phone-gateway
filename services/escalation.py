"""Mid-call escalation window (PRD UX flow).

[ESCALATE: question] → the agent speaks a hold line, CC pushes the question
to the user, and the answer arrives back over POST
/internal/call/{session_id}/escalation-answer. The wait is BOUNDED (~25 s):
the graceful degradation — "I'll check and call you back" — is the expected
common case in P1 (push may be inbox-only; cold app start eats the window).

One window at a time per call: a second [ESCALATE] while one is pending is
answered with the fallback path rather than stacking holds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

DEFAULT_ESCALATION_WINDOW_S = 25.0


@dataclass
class EscalationWindow:
    timeout_s: float = DEFAULT_ESCALATION_WINDOW_S
    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _answer: str | None = None
    _open: bool = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> bool:
        """Start a window; False if one is already pending."""
        if self._open:
            return False
        self._event = asyncio.Event()
        self._answer = None
        self._open = True
        return True

    def deliver(self, answer: str) -> bool:
        """Called by the inbound endpoint. False if no window is waiting."""
        if not self._open:
            return False
        self._answer = answer
        self._event.set()
        return True

    async def wait(self) -> str | None:
        """Answer text, or None when the window times out."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self.timeout_s)
            return self._answer
        except asyncio.TimeoutError:
            return None
        finally:
            self._open = False
