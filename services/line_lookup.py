"""Twilio Lookup v2 line-type intelligence (PRD decision 10).

CC's resolve step needs the line type ("this appears to be a mobile
number" on the confirm card; cell numbers are inside TCPA artificial-voice
rules with no business carve-out) — but Twilio credentials are gateway-only
secrets (security requirement 5), so CC asks the gateway. Any Twilio error
degrades to "unknown": resolution must not fail because Lookup hiccupped.
~$0.008/query.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_LOOKUP_API = "https://lookups.twilio.com/v2/PhoneNumbers"


async def lookup_line_type(
    number: str,
    *,
    account_sid: str,
    auth_token: str,
    http: httpx.AsyncClient,
) -> str:
    """E.164 number → "mobile" | "landline" | "nonFixedVoip" | ... | "unknown"."""
    if not account_sid or not auth_token:
        return "unknown"
    try:
        r = await http.get(
            f"{_LOOKUP_API}/{number}",
            params={"Fields": "line_type_intelligence"},
            auth=(account_sid, auth_token),
            timeout=10.0,
        )
        r.raise_for_status()
        info = r.json().get("line_type_intelligence") or {}
        return str(info.get("type") or "unknown")
    except Exception as e:  # noqa: BLE001 — lookup is advisory, never fatal
        logger.warning("Line-type lookup failed for %s: %s", number, e)
        return "unknown"
