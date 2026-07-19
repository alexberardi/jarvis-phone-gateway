"""Call recorder: inbound-clocked mixing, clip guard, graceful upload."""

import io
import wave

import numpy as np
import pytest

import services.recording as recording
from services.recording import CallRecorder, upload_recording


def _tone(n: int, value: int) -> np.ndarray:
    return np.full(n, value, dtype=np.int16)


class TestMixer:
    def test_inbound_only(self):
        r = CallRecorder()
        r.add_inbound(_tone(100, 1000))
        mixed = r.mix()
        assert len(mixed) == 100
        assert mixed[0] == 1000

    def test_outbound_anchored_at_inbound_position(self):
        r = CallRecorder()
        r.add_inbound(_tone(100, 0))       # 100 samples of callee silence
        r.add_outbound(_tone(50, 2000))    # agent speaks at t=100
        r.add_inbound(_tone(100, 1000))    # callee audio continues underneath
        mixed = r.mix()
        assert len(mixed) == 200
        assert mixed[99] == 0
        assert mixed[100] == 2000 + 1000   # overlap mixes, not overwrites
        assert mixed[150] == 1000

    def test_outbound_can_extend_past_inbound(self):
        r = CallRecorder()
        r.add_inbound(_tone(10, 0))
        r.add_outbound(_tone(100, 500))
        assert len(r.mix()) == 110

    def test_clip_guard(self):
        r = CallRecorder()
        r.add_outbound(_tone(10, 30000))  # anchored at t=0
        r.add_inbound(_tone(10, 30000))   # overlaps the same 10 samples
        mixed = r.mix()
        assert mixed.dtype == np.int16
        assert mixed.max() == 32767  # clipped, not wrapped

    def test_wav_bytes_is_8k_mono_pcm16(self):
        r = CallRecorder()
        r.add_inbound(_tone(800, 100))
        with wave.open(io.BytesIO(r.wav_bytes())) as w:
            assert w.getframerate() == 8000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getnframes() == 800


class TestUpload:
    @pytest.mark.asyncio
    async def test_success_returns_key(self, monkeypatch):
        calls = {}

        def fake_upload(bucket, key, data):
            calls["args"] = (bucket, key, len(data))

        monkeypatch.setattr(recording, "_upload_sync", fake_upload)
        key = await upload_recording("hh-1", "sess-9", b"RIFFdata")
        assert key == "hh-1/sess-9.wav"
        assert calls["args"][0] == recording.DEFAULT_BUCKET

    @pytest.mark.asyncio
    async def test_failure_degrades_to_none(self, monkeypatch):
        def boom(bucket, key, data):
            raise RuntimeError("minio down")

        monkeypatch.setattr(recording, "_upload_sync", boom)
        assert await upload_recording("hh-1", "sess-9", b"x") is None
