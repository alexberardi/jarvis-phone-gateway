"""RMS voice-activity detection with hangover endpointing.

Pure state machine — no I/O, no clocks. The caller feeds fixed-size PCM
frames (Twilio media frames are 20 ms) and receives a complete utterance
when the endpoint fires. Defaults come from the P0 spike: threshold 250
(measured telephone speech ~2 650–3 100 RMS; 500 missed it) and an 800 ms
silence hangover. P2 replaces the fixed hangover with semantic endpointing
(smart-turn) — keep this class swappable behind the same feed() contract.

Half-duplex rule (v1): while the agent is speaking, the caller passes
``suppress=True`` and the detector discards state instead of endpointing on
the agent's own echo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio.mulaw import rms


@dataclass
class VadConfig:
    threshold_rms: float = 250.0
    frame_ms: int = 20
    hangover_ms: int = 800
    # Consecutive loud frames required to enter speech (debounce).
    start_frames: int = 3
    # Frames of pre-speech audio kept so the utterance doesn't clip its onset.
    preroll_frames: int = 15
    # Force an endpoint after this much continuous speech.
    max_utterance_s: float = 15.0

    @property
    def hangover_frames(self) -> int:
        return max(1, self.hangover_ms // self.frame_ms)

    @property
    def max_utterance_frames(self) -> int:
        return int(self.max_utterance_s * 1000 / self.frame_ms)


class RmsVad:
    def __init__(self, config: VadConfig | None = None):
        self.config = config or VadConfig()
        self._frames: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._frames = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    def feed(self, frame: np.ndarray, suppress: bool = False) -> np.ndarray | None:
        """Push one PCM frame; returns a finished utterance or None.

        ``suppress`` discards accumulation (agent is speaking — half-duplex).
        """
        if suppress:
            self.reset()
            return None

        loud = rms(frame) > self.config.threshold_rms

        if not self._in_speech:
            self._frames.append(frame)
            # Bounded pre-roll so the utterance keeps its onset.
            self._frames = self._frames[-self.config.preroll_frames:]
            if loud:
                self._speech_frames += 1
                if self._speech_frames >= self.config.start_frames:
                    self._in_speech = True
                    self._silence_frames = 0
            else:
                self._speech_frames = 0
            return None

        self._frames.append(frame)
        self._silence_frames = 0 if loud else self._silence_frames + 1
        too_long = len(self._frames) > self.config.max_utterance_frames
        if self._silence_frames >= self.config.hangover_frames or too_long:
            utterance = np.concatenate(self._frames)
            self.reset()
            return utterance
        return None
