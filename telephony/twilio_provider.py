"""Twilio Media Streams provider.

Wire facts (verified in P0):
- Bidirectional streams require TwiML ``<Connect><Stream>`` placed via
  calls.create — the REST Streams subresource is unidirectional-only.
- Audio frames are headerless base64 mu-law, 8 kHz mono, 20 ms (160 bytes).
- The WS upgrade is signed with ``X-Twilio-Signature``, but an upgrade has
  no body so the HMAC covers only the URL — and Twilio signs the ``wss://``
  form while server-side reconstruction yields ``https://``. Validating
  naively therefore always fails (or, worse, a "fix" that skips validation
  always passes). ``validate_ws_signature`` tries both scheme forms.
- On a static per-worker URL the signature is a replayable constant, so
  every session gets a single-use token in the wss path (SessionTokenRegistry)
  — this also makes each call's signature unique. Security requirement 2.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Mapping
from xml.sax.saxutils import escape, quoteattr

import httpx
import numpy as np

from audio.mulaw import ulaw_decode, ulaw_encode
from telephony.provider import (
    InboundAudio,
    MarkReceived,
    ProviderEvent,
    StreamStart,
    StreamStop,
    TelephonyProvider,
)

logger = logging.getLogger(__name__)

_TWILIO_API = "https://api.twilio.com/2010-04-01"
_FRAME_BYTES = 160  # 20 ms of 8 kHz mu-law


def compute_signature(auth_token: str, url: str, params: Mapping[str, str] | None = None) -> str:
    """Twilio's request signature: base64(HMAC-SHA1(url + sorted k+v pairs))."""
    payload = url
    if params:
        for key in sorted(params):
            payload += key + params[key]
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def validate_ws_signature(auth_token: str, request_url: str, signature: str | None) -> bool:
    """Validate X-Twilio-Signature on a media-stream WS upgrade.

    Tries the URL as given AND its ws(s)/http(s) counterpart — Twilio signs
    the wss form it was handed in TwiML, while ASGI reconstruction yields
    https. A missing signature never validates.
    """
    if not signature:
        return False
    candidates = {request_url}
    for a, b in (("https://", "wss://"), ("http://", "ws://")):
        if request_url.startswith(a):
            candidates.add(b + request_url[len(a):])
        elif request_url.startswith(b):
            candidates.add(a + request_url[len(b):])
    return any(
        hmac.compare_digest(compute_signature(auth_token, url), signature)
        for url in candidates
    )


@dataclass
class PendingSession:
    """What the registry knows about a call between TwiML and stream-start."""

    session_id: str
    call_sid: str | None = None  # filled in after calls.create


class SessionTokenRegistry:
    """Single-use wss-path tokens: issue at TwiML time, claim at WS upgrade.

    Claiming pops — a second connection with the same token (replay, or a
    duplicate stream) is rejected with no state left behind. Thread-safe:
    issuing happens on the dial worker, claiming on the ASGI loop.
    """

    def __init__(self) -> None:
        self._pending: dict[str, PendingSession] = {}
        self._lock = threading.Lock()

    def issue(self, session_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._pending[token] = PendingSession(session_id=session_id)
        return token

    def bind_call_sid(self, token: str, call_sid: str) -> None:
        with self._lock:
            pending = self._pending.get(token)
            if pending is not None:
                pending.call_sid = call_sid

    def claim(self, token: str) -> PendingSession | None:
        with self._lock:
            return self._pending.pop(token, None)

    def revoke(self, token: str) -> None:
        with self._lock:
            self._pending.pop(token, None)


def validate_stream_start(start: StreamStart, pending: PendingSession) -> bool:
    """Bind the stream-start event to the claimed session.

    The TwiML carried the session id as a <Parameter>; Twilio echoes it in
    customParameters. The callSid must also match the one calls.create
    returned (when we have it — trial/dev flows may not bind it).
    """
    if start.custom_parameters.get("session_id") != pending.session_id:
        return False
    if pending.call_sid is not None and start.call_sid != pending.call_sid:
        return False
    return True


class TwilioProvider(TelephonyProvider):
    name = "twilio"

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number

    # ------------------------------------------------------------ instructions

    def build_stream_instructions(
        self, wss_url: str, session_params: Mapping[str, str]
    ) -> str:
        params = "".join(
            f"<Parameter name={quoteattr(k)} value={quoteattr(v)}/>"
            for k, v in session_params.items()
        )
        return (
            "<Response><Connect>"
            f"<Stream url={quoteattr(wss_url)}>{params}</Stream>"
            "</Connect></Response>"
        )

    # ------------------------------------------------------------ WS wire format

    def parse_ws_message(self, raw: str) -> ProviderEvent | None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Unparseable Twilio WS message (%d bytes)", len(raw))
            return None
        event = msg.get("event")
        if event == "start":
            start = msg.get("start", {})
            return StreamStart(
                stream_sid=start.get("streamSid", ""),
                call_sid=start.get("callSid"),
                custom_parameters=dict(start.get("customParameters") or {}),
            )
        if event == "media":
            payload = msg.get("media", {}).get("payload", "")
            return InboundAudio(pcm=ulaw_decode(base64.b64decode(payload)))
        if event == "mark":
            return MarkReceived(name=msg.get("mark", {}).get("name", ""))
        if event == "stop":
            return StreamStop()
        return None  # "connected" and future events are ignorable

    def media_messages(
        self, stream_sid: str, pcm_8k: np.ndarray
    ) -> Iterator[dict[str, Any]]:
        mulaw = ulaw_encode(pcm_8k)
        for i in range(0, len(mulaw), _FRAME_BYTES):
            yield {
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": base64.b64encode(mulaw[i : i + _FRAME_BYTES]).decode()
                },
            }

    def mark_message(self, stream_sid: str, name: str) -> dict[str, Any]:
        return {"event": "mark", "streamSid": stream_sid, "mark": {"name": name}}

    def clear_message(self, stream_sid: str) -> dict[str, Any]:
        return {"event": "clear", "streamSid": stream_sid}

    # ------------------------------------------------------------ REST

    async def start_call(
        self, *, to_number: str, instructions: str, http: httpx.AsyncClient
    ) -> str:
        r = await http.post(
            f"{_TWILIO_API}/Accounts/{self.account_sid}/Calls.json",
            data={"To": to_number, "From": self.from_number, "Twiml": instructions},
            auth=(self.account_sid, self.auth_token),
        )
        r.raise_for_status()
        return r.json()["sid"]

    async def end_call(self, call_sid: str, http: httpx.AsyncClient) -> None:
        r = await http.post(
            f"{_TWILIO_API}/Accounts/{self.account_sid}/Calls/{call_sid}.json",
            data={"Status": "completed"},
            auth=(self.account_sid, self.auth_token),
        )
        r.raise_for_status()
