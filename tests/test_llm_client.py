"""llm-proxy stream client: frame contract, sentence regrouping, turn-cap cancel."""

import asyncio
import json

import httpx
import pytest

from llm.client import (
    LlmProxyStreamClient,
    LlmStreamError,
    TurnTimeout,
    sentences,
)

client = LlmProxyStreamClient("http://llm.test", "app", "key")


def sse(frames: list[dict]) -> str:
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames)


def mock_http(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_deltas_streamed_until_done_frame():
    body = sse([
        {"delta": "Hel"},
        {"delta": "lo."},
        {"done": True, "content": "Hello.", "usage": {}},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Jarvis-App-Id"] == "app"
        assert request.headers["X-Request-Id"]  # plumbed for cancellation
        assert json.loads(request.read())["stream"] is True
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async with mock_http(handler) as http:
        got = [d async for d in client.stream_deltas([{"role": "user", "content": "hi"}], http=http)]
    assert got == ["Hel", "lo."]


@pytest.mark.asyncio
async def test_error_frame_raises():
    body = sse([{"delta": "x"}, {"error": "model exploded"}])

    def handler(request):
        return httpx.Response(200, text=body)

    async with mock_http(handler) as http:
        with pytest.raises(LlmStreamError, match="model exploded"):
            async for _ in client.stream_deltas([], http=http):
                pass


@pytest.mark.asyncio
async def test_non_200_raises():
    def handler(request):
        return httpx.Response(503, text="down")

    async with mock_http(handler) as http:
        with pytest.raises(LlmStreamError, match="503"):
            async for _ in client.stream_deltas([], http=http):
                pass


@pytest.mark.asyncio
async def test_cancelled_frame_ends_stream_cleanly():
    body = sse([{"delta": "a"}, {"cancelled": True}])

    def handler(request):
        return httpx.Response(200, text=body)

    async with mock_http(handler) as http:
        got = [d async for d in client.stream_deltas([], http=http)]
    assert got == ["a"]


@pytest.mark.asyncio
async def test_turn_cap_fires_cancel_endpoint():
    """The 20s cap must POST the cancel endpoint, never silently abandon."""
    cancels: list[str] = []

    class StubResponse:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"delta": "tok"}'
            await asyncio.sleep(3600)  # generation hangs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class StubHttp:
        def stream(self, method, url, **kwargs):
            self.request_id = kwargs["headers"]["X-Request-Id"]
            return StubResponse()

        async def post(self, url, **kwargs):
            cancels.append(url)
            return httpx.Response(200, json={"status": "cancelling"})

    http = StubHttp()
    got = []
    with pytest.raises(TurnTimeout):
        async for d in client.stream_deltas([], http=http, turn_timeout_s=0.2):
            got.append(d)
    assert got == ["tok"]
    assert len(cancels) == 1
    assert cancels[0].endswith(f"/v1/chat/completions/cancel/{http.request_id}")


@pytest.mark.asyncio
async def test_sentences_regrouping():
    async def deltas():
        for d in ["Hi the", "re. How are", " you? Grea", "t"]:
            yield d

    got = [s async for s in sentences(deltas())]
    assert got == ["Hi there.", "How are you?", "Great"]
