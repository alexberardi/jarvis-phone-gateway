"""mu-law codec + resampling + WAV wrapping (ported from the spike self-test)."""

import io
import wave

import numpy as np

from audio.mulaw import (
    pcm8k_to_whisper_wav,
    pcm_to_wav_bytes,
    resample_pcm,
    rms,
    ulaw_decode,
    ulaw_encode,
)


def test_roundtrip_error_bounded():
    # The spike's startup self-test: full-swing ramp, max error < 1024
    # (mu-law is logarithmic; error grows with amplitude).
    pcm = np.linspace(-30000, 30000, 4000).astype(np.int16)
    decoded = ulaw_decode(ulaw_encode(pcm)).astype(np.int32)
    err = np.abs(decoded - pcm.astype(np.int32))
    assert err.max() < 1024


def test_roundtrip_small_amplitudes_tight():
    pcm = np.linspace(-100, 100, 500).astype(np.int16)
    decoded = ulaw_decode(ulaw_encode(pcm)).astype(np.int32)
    assert np.abs(decoded - pcm.astype(np.int32)).max() <= 8


def test_encode_length_one_byte_per_sample():
    pcm = np.zeros(160, dtype=np.int16)
    assert len(ulaw_encode(pcm)) == 160


def test_rms():
    assert rms(np.zeros(160, dtype=np.int16)) == 0.0
    assert rms(np.array([], dtype=np.int16)) == 0.0
    assert rms(np.full(160, 1000, dtype=np.int16)) == 1000.0


def test_resample_16k_to_8k_halves_length():
    pcm = np.random.default_rng(0).integers(-3000, 3000, 3200).astype(np.int16)
    out = resample_pcm(pcm, 16000, 8000)
    assert out.dtype == np.int16
    assert len(out) == 1600


def test_resample_8k_to_16k_doubles_length():
    pcm = np.random.default_rng(1).integers(-3000, 3000, 800).astype(np.int16)
    out = resample_pcm(pcm, 8000, 16000)
    assert len(out) == 1600


def test_resample_same_rate_is_identity():
    pcm = np.arange(100, dtype=np.int16)
    assert resample_pcm(pcm, 8000, 8000) is pcm


def test_wav_container_shape():
    pcm = np.arange(-500, 500, dtype=np.int16)
    wav = pcm_to_wav_bytes(pcm, 8000)
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 8000
        assert w.getnframes() == len(pcm)


def test_whisper_wav_is_linear_pcm_8k():
    # Whisper 500s on mu-law WAV — the helper must produce linear PCM16.
    pcm = np.arange(0, 1600, dtype=np.int16)
    wav = pcm8k_to_whisper_wav(pcm)
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getsampwidth() == 2  # 16-bit linear, not 8-bit mu-law
        assert w.getframerate() == 8000
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.array_equal(frames, pcm)
