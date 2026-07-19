"""Escalation window: bounded wait, one at a time, deliver semantics."""

import asyncio

import pytest

from services.escalation import EscalationWindow


class TestEscalationWindow:
    @pytest.mark.asyncio
    async def test_answer_delivered_within_window(self):
        w = EscalationWindow(timeout_s=5.0)
        assert w.open()

        async def answer_later():
            await asyncio.sleep(0.01)
            assert w.deliver("6:30 works")

        task = asyncio.create_task(answer_later())
        assert await w.wait() == "6:30 works"
        await task
        assert not w.is_open

    @pytest.mark.asyncio
    async def test_timeout_returns_none_and_closes(self):
        w = EscalationWindow(timeout_s=0.02)
        assert w.open()
        assert await w.wait() is None
        assert not w.is_open

    def test_second_open_while_pending_refused(self):
        w = EscalationWindow()
        assert w.open()
        assert not w.open()

    def test_deliver_without_open_window_refused(self):
        w = EscalationWindow()
        assert not w.deliver("answer to nothing")

    @pytest.mark.asyncio
    async def test_window_reusable_after_completion(self):
        w = EscalationWindow(timeout_s=0.02)
        assert w.open()
        assert await w.wait() is None
        assert w.open()  # closed windows can open again
        w.deliver("yes")
        assert await w.wait() == "yes"
