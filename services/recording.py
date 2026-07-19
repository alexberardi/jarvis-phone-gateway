"""Local call recording: both directions mixed to one 8 kHz WAV → MinIO.

PRD decision 9: we already hold every frame both ways, so the recording is
a free local mix — no Twilio recording product, audio never leaves the
house except to the household's own MinIO. The inbound (callee) stream is
the wall clock: Twilio delivers continuous 8 kHz frames including silence,
so outbound (agent) audio is mixed in at the inbound-timeline position
where playback started. Sample-add with an int32 accumulator + clip guard.

Notice-off rule: when the session's recording notice is disabled, callers
must simply never construct/upload a recorder — enforced by the dial
worker, not here.

Upload is boto3-in-a-thread (same env surface as llm-proxy's
storage/object_store.py: S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID/SECRET,
S3_FORCE_PATH_STYLE). Failures degrade gracefully: the call outcome simply
carries no audio key.
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

from audio.mulaw import pcm_to_wav_bytes

logger = logging.getLogger(__name__)

RECORDING_BUCKET_ENV = "PHONE_CALLS_BUCKET"
DEFAULT_BUCKET = "phone-calls"
_RATE = 8000


class CallRecorder:
    def __init__(self) -> None:
        self._inbound: list[np.ndarray] = []
        self._inbound_len = 0
        # (start_offset_samples, pcm) for each outbound burst.
        self._outbound: list[tuple[int, np.ndarray]] = []

    def add_inbound(self, pcm_8k: np.ndarray) -> None:
        self._inbound.append(pcm_8k)
        self._inbound_len += len(pcm_8k)

    def add_outbound(self, pcm_8k: np.ndarray) -> None:
        """Agent audio, anchored at the current inbound-timeline position."""
        self._outbound.append((self._inbound_len, pcm_8k))

    def mix(self) -> np.ndarray:
        inbound = (
            np.concatenate(self._inbound)
            if self._inbound
            else np.array([], dtype=np.int16)
        )
        total = len(inbound)
        for start, pcm in self._outbound:
            total = max(total, start + len(pcm))
        mixed = np.zeros(total, dtype=np.int32)
        mixed[: len(inbound)] += inbound.astype(np.int32)
        for start, pcm in self._outbound:
            mixed[start : start + len(pcm)] += pcm.astype(np.int32)
        return np.clip(mixed, -32768, 32767).astype(np.int16)

    def wav_bytes(self) -> bytes:
        return pcm_to_wav_bytes(self.mix(), _RATE)


def _upload_sync(bucket: str, key: str, data: bytes) -> None:
    # Deferred imports: boto3 is only needed when recording is enabled.
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
        region_name=os.getenv("S3_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(
            s3={
                "addressing_style": "path"
                if os.getenv("S3_FORCE_PATH_STYLE", "true").lower() == "true"
                else "auto"
            }
        ),
    )
    try:
        client.create_bucket(Bucket=bucket)
    except Exception:  # noqa: BLE001 — exists/no-perms both fine; put decides
        pass
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType="audio/wav")


async def upload_recording(
    household_id: str, session_id: str, wav_data: bytes
) -> str | None:
    """Store the mixed WAV; returns the object key or None on any failure."""
    bucket = os.getenv(RECORDING_BUCKET_ENV, DEFAULT_BUCKET)
    key = f"{household_id}/{session_id}.wav"
    try:
        await asyncio.to_thread(_upload_sync, bucket, key, wav_data)
        return key
    except Exception as e:  # noqa: BLE001 — audio-less outcome beats a dead call
        logger.error("Recording upload failed for %s: %s", session_id, e)
        return None
