"""Twilio Lookup line-type: advisory, never fatal."""

import httpx
import pytest

from services.line_lookup import lookup_line_type


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestLookup:
    @pytest.mark.asyncio
    async def test_mobile_detected(self):
        def handler(request):
            assert "line_type_intelligence" in str(request.url)
            return httpx.Response(
                200, json={"line_type_intelligence": {"type": "mobile"}}
            )

        async with _client(handler) as http:
            result = await lookup_line_type(
                "+15555550123", account_sid="AC1", auth_token="tok", http=http
            )
        assert result == "mobile"

    @pytest.mark.asyncio
    async def test_twilio_error_degrades_to_unknown(self):
        def handler(request):
            return httpx.Response(404, json={"message": "not found"})

        async with _client(handler) as http:
            assert (
                await lookup_line_type(
                    "+1000", account_sid="AC1", auth_token="tok", http=http
                )
                == "unknown"
            )

    @pytest.mark.asyncio
    async def test_missing_creds_short_circuits_to_unknown(self):
        async with _client(lambda r: httpx.Response(500)) as http:
            assert (
                await lookup_line_type("+1", account_sid="", auth_token="", http=http)
                == "unknown"
            )
