"""G.711 mu-law codec + sample-rate utilities.

The telephone leg is headerless 8 kHz mu-law (Twilio Media Streams); the
Jarvis services speak linear PCM16 (whisper wants a WAV container, Piper
emits 16 kHz PCM16). Everything here is pure numpy + scipy's polyphase
resampler — no audioop (removed in Python 3.13), no native telephony deps.

Whisper gotcha (P0 spike): /transcribe 500s on a mu-law WAV. Always
ulaw_decode to linear PCM16 first and wrap THAT in the WAV container —
whisper resamples any linear rate (including 8 kHz) internally.
"""

from __future__ import annotations

import io
import math
import wave

import numpy as np
from scipy.signal import resample_poly

_BIAS = 0x84
_CLIP = 32635


def ulaw_decode(data: bytes) -> np.ndarray:
    """G.711 mu-law bytes -> int16 PCM. Proven in the P0 spike."""
    u = ~np.frombuffer(data, dtype=np.uint8)
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = (((mantissa.astype(np.int32) << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


def ulaw_encode(pcm: np.ndarray) -> bytes:
    """int16 PCM -> G.711 mu-law bytes. Proven in the P0 spike."""
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    x = np.clip(np.abs(x), 0, _CLIP) + _BIAS
    exponent = (np.floor(np.log2(x)) - 7).clip(0, 7).astype(np.int32)
    mantissa = (x >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def rms(pcm: np.ndarray) -> float:
    """Root-mean-square level of an int16 frame (0.0 for empty input)."""
    return float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if len(pcm) else 0.0


def resample_pcm(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase-resample int16 PCM between arbitrary rates (clip-safe)."""
    if src_rate == dst_rate:
        return pcm
    g = math.gcd(dst_rate, src_rate)
    out = resample_poly(pcm.astype(np.float32), dst_rate // g, src_rate // g)
    return np.clip(out, -32768, 32767).astype(np.int16)


def pcm_to_wav_bytes(pcm: np.ndarray, rate: int) -> bytes:
    """Wrap mono int16 PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def pcm8k_to_whisper_wav(pcm_8k: np.ndarray) -> bytes:
    """Linear-PCM WAV for whisper /transcribe.

    The input must already be DECODED (linear PCM16) — never hand whisper a
    mu-law WAV (500s). whisper-api resamples 8 kHz to 16 kHz internally.
    """
    return pcm_to_wav_bytes(pcm_8k, 8000)
