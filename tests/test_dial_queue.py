"""Dial-queue consumer: minimal job shape, strict parsing, drop-don't-trust."""

import json

from queues.dial_queue import DIAL_QUEUE_KEY, DialJob, DialQueue, parse_job


class StubRedis:
    def __init__(self):
        self.items: list[bytes] = []

    def blpop(self, keys, timeout: int = 0):
        assert keys == [DIAL_QUEUE_KEY]
        if not self.items:
            return None
        return (DIAL_QUEUE_KEY.encode(), self.items.pop(0))

    def rpush(self, key, *values):
        self.items.extend(v.encode() if isinstance(v, str) else v for v in values)


def test_push_pop_roundtrip():
    q = DialQueue(StubRedis())
    q.push(DialJob(session_id="s-1", household_id="h-1"))
    job = q.pop()
    assert job == DialJob(session_id="s-1", household_id="h-1")


def test_pop_timeout_returns_none():
    assert DialQueue(StubRedis()).pop(timeout_s=0) is None


def test_malformed_payloads_dropped():
    assert parse_job(b"not json") is None
    assert parse_job(b"[1,2]") is None
    assert parse_job(json.dumps({"session_id": "s"}).encode()) is None
    assert parse_job(json.dumps({"household_id": "h"}).encode()) is None
    assert parse_job(json.dumps({"session_id": "", "household_id": "h"}).encode()) is None
    assert parse_job(json.dumps({"session_id": 5, "household_id": "h"}).encode()) is None


def test_extra_fields_ignored_not_trusted():
    # The queue is transport, never authorization: a forged "state" field
    # must not survive into the job the worker acts on.
    raw = json.dumps({
        "session_id": "s-1", "household_id": "h-1",
        "state": "confirmed", "dialed_number": "+15550009999",
    }).encode()
    job = parse_job(raw)
    assert job == DialJob(session_id="s-1", household_id="h-1")
    assert not hasattr(job, "state")
