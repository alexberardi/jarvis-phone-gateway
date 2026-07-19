"""Telephony provider interface (PRD decision 7).

Twilio is the launch provider; Telnyx (~half price, own media WS) is the
understudy. Everything provider-specific — TwiML/stream instructions, WS
message wire format, REST dial/hangup — lives behind this interface so the
call loop and tests never import a vendor SDK. The fake-Twilio WS fixture in
tests exists precisely because this seam is mockable.

Inbound WS traffic parses into the small event vocabulary below; audio is
always delivered/accepted as 8 kHz int16 linear PCM (the provider owns the
mu-law leg).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import httpx
import numpy as np


@dataclass
class StreamStart:
    """First event on a media stream — carries the binding material."""

    stream_sid: str
    call_sid: str | None
    custom_parameters: dict[str, str] = field(default_factory=dict)


@dataclass
class InboundAudio:
    """One decoded media frame: 8 kHz int16 linear PCM."""

    pcm: np.ndarray


@dataclass
class MarkReceived:
    """Provider acknowledged playback up to a named mark."""

    name: str


@dataclass
class StreamStop:
    """Provider ended the media stream."""


ProviderEvent = StreamStart | InboundAudio | MarkReceived | StreamStop


class TelephonyProvider(ABC):
    """Vendor seam. Implementations: TwilioProvider (launch), Telnyx (P2?)."""

    name: str

    @abstractmethod
    def build_stream_instructions(
        self, wss_url: str, session_params: Mapping[str, str]
    ) -> str:
        """Vendor payload that points the call's media at ``wss_url``.

        For Twilio this is TwiML <Connect><Stream>; session_params become
        <Parameter> entries echoed back in the stream-start event (the
        callSid/session binding — security requirement 2).
        """

    @abstractmethod
    def parse_ws_message(self, raw: str) -> ProviderEvent | None:
        """Decode one inbound WS text message; None for ignorable events."""

    @abstractmethod
    def media_messages(
        self, stream_sid: str, pcm_8k: np.ndarray
    ) -> Iterator[dict[str, Any]]:
        """Outbound media messages (JSON-serializable) for one PCM buffer."""

    @abstractmethod
    def mark_message(self, stream_sid: str, name: str) -> dict[str, Any]:
        """Playback marker — provider echoes MarkReceived when audio drained."""

    @abstractmethod
    def clear_message(self, stream_sid: str) -> dict[str, Any]:
        """Flush queued playback (barge-in, P2)."""

    @abstractmethod
    async def start_call(
        self, *, to_number: str, instructions: str, http: httpx.AsyncClient
    ) -> str:
        """Place the outbound call; returns the provider call id."""

    @abstractmethod
    async def end_call(self, call_sid: str, http: httpx.AsyncClient) -> None:
        """Terminate the call server-side (reaper path included)."""
