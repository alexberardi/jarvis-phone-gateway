"""RMS VAD state machine: silence → speech → hangover → endpoint."""

import numpy as np

from audio.vad import RmsVad, VadConfig


CFG = VadConfig(threshold_rms=250.0, frame_ms=20, hangover_ms=100, start_frames=3,
                preroll_frames=5, max_utterance_s=1.0)

LOUD = np.full(160, 3000, dtype=np.int16)   # telephone speech ~2650-3100 RMS
QUIET = np.full(160, 50, dtype=np.int16)


def test_silence_never_endpoints():
    vad = RmsVad(CFG)
    for _ in range(200):
        assert vad.feed(QUIET) is None
    assert not vad.in_speech


def test_speech_needs_consecutive_loud_frames():
    vad = RmsVad(CFG)
    # Two loud frames + quiet resets the debounce counter.
    vad.feed(LOUD)
    vad.feed(LOUD)
    vad.feed(QUIET)
    assert not vad.in_speech
    # Three consecutive loud frames enter speech.
    vad.feed(LOUD)
    vad.feed(LOUD)
    vad.feed(LOUD)
    assert vad.in_speech


def test_endpoint_after_hangover_silence():
    vad = RmsVad(CFG)
    for _ in range(5):
        assert vad.feed(LOUD) is None
    # 100 ms hangover = 5 quiet frames at 20 ms.
    utterance = None
    for _ in range(CFG.hangover_frames):
        utterance = vad.feed(QUIET)
    assert utterance is not None
    assert not vad.in_speech


def test_utterance_includes_preroll():
    vad = RmsVad(CFG)
    for _ in range(4):
        vad.feed(QUIET)  # pre-speech audio
    for _ in range(5):
        vad.feed(LOUD)
    utterance = None
    for _ in range(CFG.hangover_frames):
        utterance = vad.feed(QUIET)
    assert utterance is not None
    # 5 loud + hangover quiet + preroll (capped at 5) — onset preserved.
    assert len(utterance) > 5 * 160


def test_max_utterance_forces_endpoint():
    vad = RmsVad(CFG)  # max 1.0 s = 50 frames
    utterance = None
    for _ in range(60):
        got = vad.feed(LOUD)
        if got is not None:
            utterance = got
            break
    assert utterance is not None


def test_suppress_discards_state():
    vad = RmsVad(CFG)
    for _ in range(5):
        vad.feed(LOUD)
    assert vad.in_speech
    # Agent starts speaking — half-duplex discard.
    assert vad.feed(LOUD, suppress=True) is None
    assert not vad.in_speech
    # Silence afterwards must not produce a phantom utterance.
    for _ in range(20):
        assert vad.feed(QUIET) is None
