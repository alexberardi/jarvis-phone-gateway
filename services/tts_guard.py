"""jarvis-tts response guard.

P0 failure ladder item 6: jarvis-tts returns HTTP 200 with a JSON error
body on empty/invalid text, and an LLM stream legitimately produces empty
fragments (post-[HANGUP], post-think-strip). Piping that response straight
to the resampler injects JSON bytes as audio into the live call. Every TTS
response MUST pass through this guard before its bytes are treated as PCM.

Sample-rate fact: the actual baked-in Piper voice is 16 000 Hz — the docs'
22 050 claim is wrong. There is no per-call provider pinning, so the rate
is (re)read from X-Audio-Sample-Rate on EVERY response.
"""

from __future__ import annotations

import numpy as np

from audio.mulaw import resample_pcm

DEFAULT_SAMPLE_RATE = 16000


class TtsResponseError(RuntimeError):
    """The TTS response is not audio (error body, wrong status)."""


def validate_tts_response_headers(
    status_code: int,
    content_type: str | None,
    sample_rate_header: str | None,
) -> int:
    """Gate a TTS response; returns the stream's sample rate.

    Raises TtsResponseError for non-200s and for the 200-with-JSON-error-body
    trap. The caller must not read body bytes as audio unless this returned.
    """
    if status_code != 200:
        raise TtsResponseError(f"tts returned HTTP {status_code}")
    if content_type and "json" in content_type.lower():
        raise TtsResponseError("tts returned a JSON body (error masquerading as 200)")
    try:
        return int(sample_rate_header) if sample_rate_header else DEFAULT_SAMPLE_RATE
    except ValueError:
        raise TtsResponseError(
            f"tts sent an unparseable X-Audio-Sample-Rate: {sample_rate_header!r}"
        )


class PcmChunkAdapter:
    """Byte chunks -> int16 PCM at 8 kHz, odd-byte-safe.

    HTTP chunk boundaries land mid-sample; a trailing odd byte is carried
    into the next chunk (spike-proven logic).
    """

    def __init__(self, src_rate: int, dst_rate: int = 8000):
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._carry = b""

    def feed(self, chunk: bytes) -> np.ndarray:
        data = self._carry + chunk
        usable_len = len(data) - (len(data) % 2)
        self._carry = data[usable_len:]
        usable = data[:usable_len]
        if not usable:
            return np.array([], dtype=np.int16)
        pcm = np.frombuffer(usable, dtype=np.int16)
        return resample_pcm(pcm, self.src_rate, self.dst_rate)
