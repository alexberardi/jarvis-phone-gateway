"""TTS response guard: the 200-with-JSON-error trap + per-response sample rate."""

import numpy as np
import pytest

from services.tts_guard import (
    PcmChunkAdapter,
    TtsResponseError,
    validate_tts_response_headers,
)


class TestHeaderGuard:
    def test_valid_pcm_response_returns_rate(self):
        rate = validate_tts_response_headers(200, "audio/raw", "16000")
        assert rate == 16000

    def test_json_body_rejected(self):
        # jarvis-tts returns 200 + JSON error body on empty text (P0 item 6).
        with pytest.raises(TtsResponseError, match="JSON"):
            validate_tts_response_headers(200, "application/json", None)

    def test_json_content_type_case_insensitive(self):
        with pytest.raises(TtsResponseError):
            validate_tts_response_headers(200, "Application/JSON; charset=utf-8", None)

    def test_non_200_rejected(self):
        with pytest.raises(TtsResponseError, match="503"):
            validate_tts_response_headers(503, "audio/raw", "16000")

    def test_missing_rate_header_defaults_16k(self):
        # Real Piper voice is 16 kHz — the docs' 22050 claim is wrong.
        assert validate_tts_response_headers(200, "audio/raw", None) == 16000

    def test_garbage_rate_header_rejected(self):
        with pytest.raises(TtsResponseError, match="Sample-Rate"):
            validate_tts_response_headers(200, "audio/raw", "fast")


class TestPcmChunkAdapter:
    def test_resamples_16k_to_8k(self):
        adapter = PcmChunkAdapter(src_rate=16000)
        pcm16k = np.zeros(1600, dtype=np.int16)
        out = adapter.feed(pcm16k.tobytes())
        assert len(out) == 800

    def test_odd_byte_carry_across_chunks(self):
        adapter = PcmChunkAdapter(src_rate=8000)  # identity rate: exact compare
        samples = np.arange(100, dtype=np.int16)
        raw = samples.tobytes()
        # Split on an odd boundary — sample 50 is torn across chunks.
        out1 = adapter.feed(raw[:101])
        out2 = adapter.feed(raw[101:])
        joined = np.concatenate([out1, out2])
        assert np.array_equal(joined, samples)

    def test_empty_chunk_yields_empty(self):
        adapter = PcmChunkAdapter(src_rate=16000)
        assert len(adapter.feed(b"")) == 0
