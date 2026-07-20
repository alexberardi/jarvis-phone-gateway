"""The live call turn pipeline: utterance PCM → whisper → LLM → TTS → PCM.

Composes the proven foundation pieces into the per-turn chain the P0 spike
validated (median 2.5 s voice-to-voice, verbatim STT on G.711):

    utterance (8 kHz PCM from VAD)
      → whisper /transcribe        (linear-PCM WAV, speaker_recognition off)
      → llm-proxy live, streamed   (think-strip → tool tokens → sentences)
      → jarvis-tts /speak/stream   (per sentence, content-type guarded)
      → 8 kHz PCM reply

Tool events (PRD decision 6): [HANGUP] ends the call after the reply;
[OUTCOME: facts] accumulates for the wrapup summary; [ESCALATE: question]
opens the bounded escalation window (fallback line + graceful end on
timeout — the expected P1 case); [DTMF] is parsed but ignored until P2.

Per-turn stage timings follow the spike's metrics.json vocabulary
(stt_ms / llm_ttft_ms / tts_ttfb_ms / total_ms + the fixed endpoint
hangover) and ride to CC on every turn event (PRD Observability: the
latency budget is only enforceable if the gateway emits what it measures).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from audio.mulaw import pcm8k_to_whisper_wav
from llm.client import LlmProxyStreamClient, LlmStreamError, TurnTimeout, sentences
from llm.think_strip import ThinkStripper
from llm.tool_tokens import Dtmf, Escalate, Hangup, Outcome, ToolTokenParser
from services.escalation import EscalationWindow
from services.prompt import (
    EMPTY_REPLY_LINE,
    ESCALATION_FALLBACK_LINE,
    FALLBACK_GOODBYE_LINE,
    HOLD_LINE,
    TURN_FAILURE_LINE,
    initial_messages,
    sounds_like_farewell,
    with_no_think,
)
from services.session_client import SessionClient
from services.tts_guard import PcmChunkAdapter, validate_tts_response_headers

logger = logging.getLogger(__name__)

_EMPTY = np.array([], dtype=np.int16)


async def transcribe(
    utterance_pcm_8k: np.ndarray, whisper_url: str, http: httpx.AsyncClient
) -> str:
    """whisper /transcribe on a linear-PCM WAV (never mu-law — it 500s)."""
    wav = pcm8k_to_whisper_wav(utterance_pcm_8k)
    r = await http.post(
        f"{whisper_url.rstrip('/')}/transcribe",
        files={"file": ("utterance.wav", wav, "audio/wav")},
        data={"speaker_recognition": "false"},
        timeout=30.0,
    )
    r.raise_for_status()
    body = r.json()
    return str(body.get("text") or body.get("transcript") or "").strip()


async def synthesize(
    text: str, tts_url: str, http: httpx.AsyncClient
) -> tuple[np.ndarray, float | None]:
    """One sentence through /speak/stream → (8 kHz PCM, ttfb_ms).

    Every response passes the tts_guard (the 200-with-JSON-error trap) and
    re-reads X-Audio-Sample-Rate — no per-call provider pinning exists.
    """
    text = text.strip()
    if not text:
        return _EMPTY, None
    t0 = time.monotonic()
    async with http.stream(
        "POST",
        f"{tts_url.rstrip('/')}/speak/stream",
        json={"text": text},
        timeout=30.0,
    ) as r:
        rate = validate_tts_response_headers(
            r.status_code,
            r.headers.get("content-type"),
            r.headers.get("x-audio-sample-rate"),
        )
        adapter = PcmChunkAdapter(src_rate=rate, dst_rate=8000)
        chunks: list[np.ndarray] = []
        ttfb_ms: float | None = None
        async for chunk in r.aiter_bytes():
            if chunk and ttfb_ms is None:
                ttfb_ms = (time.monotonic() - t0) * 1000
            pcm = adapter.feed(chunk)
            if len(pcm):
                chunks.append(pcm)
    return (np.concatenate(chunks) if chunks else _EMPTY), ttfb_ms


@dataclass
class TurnRecord:
    """One completed turn — transcript halves + timings, spike vocabulary."""

    turn: int
    heard: str
    said: str
    stt_ms: float
    llm_ttft_ms: float | None
    tts_ttfb_ms: float | None
    total_ms: float
    events: list[str] = field(default_factory=list)

    def as_event(self) -> dict[str, Any]:
        return {
            "n": self.turn,
            "heard": self.heard,
            "said": self.said,
            "timings": {
                "stt_ms": round(self.stt_ms, 1),
                "llm_ttft_ms": round(self.llm_ttft_ms, 1)
                if self.llm_ttft_ms is not None
                else None,
                "tts_ttfb_ms": round(self.tts_ttfb_ms, 1)
                if self.tts_ttfb_ms is not None
                else None,
                "total_ms": round(self.total_ms, 1),
            },
            "events": self.events,
        }


class LiveTurnPipeline:
    """Per-call pipeline instance. Matches media_stream's TurnPipeline shape:
    ``await pipeline(utterance_pcm, media_session) -> pcm | None``."""

    def __init__(
        self,
        *,
        session: dict[str, Any],
        whisper_url: str,
        tts_url: str,
        llm: LlmProxyStreamClient,
        http: httpx.AsyncClient,
        session_client: SessionClient | None = None,
        escalation: EscalationWindow | None = None,
        turn_timeout_s: float = 20.0,
        redact_transcript: bool = False,
    ):
        self.session = session
        self.session_id = str(session.get("id") or session.get("session_id") or "")
        self.whisper_url = whisper_url
        self.tts_url = tts_url
        self.llm = llm
        self.http = http
        self.session_client = session_client
        self.escalation = escalation or EscalationWindow()
        self.turn_timeout_s = turn_timeout_s
        self.messages: list[dict[str, str]] = initial_messages(session)
        self.outcome_facts: list[str] = []
        # True once any turn has recorded an [OUTCOME]; gates the hangup that
        # would otherwise fire on the same turn the outcome first appears.
        self._outcome_recorded_earlier = False
        self.turn_records: list[TurnRecord] = []
        self.escalation_unanswered = False
        # Notice-off rule (PRD decision 9): with the recording notice
        # disabled, nothing conversational may persist — turn events carry
        # timings only; the in-memory history exists solely for the wrapup
        # summary and dies with this object.
        self.redact_transcript = redact_transcript
        self._bg_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------ generation

    async def _generate_reply(
        self, media_session: Any
    ) -> tuple[np.ndarray, str, list[Any], float | None, float | None]:
        """One LLM stream → (pcm, spoken_text, events, llm_ttft_ms, tts_ttfb_ms)."""
        stripper = ThinkStripper()
        parser = ToolTokenParser()
        events: list[Any] = []
        t0 = time.monotonic()
        llm_ttft_ms: float | None = None
        tts_ttfb_ms: float | None = None
        spoken_parts: list[str] = []
        pcm_parts: list[np.ndarray] = []

        async def speakable_deltas():
            nonlocal llm_ttft_ms
            async for delta in self.llm.stream_deltas(
                self.messages, http=self.http, turn_timeout_s=self.turn_timeout_s
            ):
                if llm_ttft_ms is None:
                    llm_ttft_ms = (time.monotonic() - t0) * 1000
                text, new_events = parser.feed(stripper.feed(delta))
                events.extend(new_events)
                if text:
                    yield text
            tail, tail_events = parser.feed(stripper.flush())
            events.extend(tail_events)
            tail += parser.flush()
            if tail:
                yield tail

        async for sentence in sentences(speakable_deltas()):
            pcm, ttfb = await self._speak_sentence(sentence)
            if ttfb is not None and tts_ttfb_ms is None:
                tts_ttfb_ms = ttfb
            if len(pcm):
                spoken_parts.append(sentence)
                pcm_parts.append(pcm)

        reply_pcm = np.concatenate(pcm_parts) if pcm_parts else _EMPTY
        return reply_pcm, " ".join(spoken_parts), events, llm_ttft_ms, tts_ttfb_ms

    async def _speak_sentence(self, sentence: str) -> tuple[np.ndarray, float | None]:
        try:
            return await synthesize(sentence, self.tts_url, self.http)
        except Exception as e:  # noqa: BLE001 — one bad sentence must not kill the turn
            logger.error("TTS failed for sentence %r: %s", sentence[:60], e)
            return _EMPTY, None

    # ------------------------------------------------------------ the turn

    async def __call__(
        self, utterance_pcm: np.ndarray, media_session: Any
    ) -> np.ndarray | None:
        turn_no = len(self.turn_records) + 1
        t0 = time.monotonic()

        heard = await transcribe(utterance_pcm, self.whisper_url, self.http)
        stt_ms = (time.monotonic() - t0) * 1000
        if not heard:
            return None
        # Directive re-asserted per turn; the transcript records `heard` raw.
        self.messages.append({"role": "user", "content": with_no_think(heard)})

        try:
            pcm, said, events, llm_ttft_ms, tts_ttfb_ms = await self._generate_reply(
                media_session
            )
        except (TurnTimeout, LlmStreamError) as e:
            # Spoken fallback, bounded dead air — never silence into the void.
            logger.error("LLM turn %d failed for %s: %s", turn_no, self.session_id, e)
            pcm, _ = await self._speak_sentence(TURN_FAILURE_LINE)
            self._record_turn(
                turn_no, heard, TURN_FAILURE_LINE, stt_ms, None, None, t0,
                ["llm_failure"],
            )
            return pcm if len(pcm) else None

        if said:
            self.messages.append({"role": "assistant", "content": said})

        event_names: list[str] = []
        hangup = False
        outcome_this_turn = False
        escalate_q: str | None = None
        for ev in events:
            if isinstance(ev, Hangup):
                hangup = True
                event_names.append("hangup")
            elif isinstance(ev, Outcome):
                self.outcome_facts.append(ev.facts)
                outcome_this_turn = True
                event_names.append("outcome")
            elif isinstance(ev, Escalate):
                escalate_q = ev.question
                event_names.append("escalate")
            elif isinstance(ev, Dtmf):
                logger.info("[DTMF] parsed but ignored until P2: %r", ev.digits)
                event_names.append("dtmf_ignored")

        if escalate_q is not None and not hangup:
            extra_pcm, extra_said = await self._run_escalation(
                escalate_q, media_session, bool(said)
            )
            if len(extra_pcm):
                pcm = np.concatenate([pcm, extra_pcm]) if len(pcm) else extra_pcm
            if extra_said:
                said = f"{said} {extra_said}".strip()

        # The goal is not achieved just because the model said it was. Live
        # 2026-07-19 (food order) and 2026-07-20 (appointment): the model
        # recorded the outcome and hung up in the SAME reply, before the
        # business had confirmed anything — both calls ended with nothing
        # actually booked. The prompt forbids this and the model did it twice,
        # so enforce it here: the first outcome never ends the call. A hangup
        # on any later turn is honoured, so this costs one extra exchange.
        they_closed = sounds_like_farewell(heard)
        if (
            hangup
            and outcome_this_turn
            and not self._outcome_recorded_earlier
            and not they_closed
        ):
            hangup = False
            event_names.append("hangup_deferred")
            logger.warning(
                "Deferred [HANGUP] on turn %d for %s — outcome recorded this "
                "same turn and they had not signed off, giving them a chance "
                "to confirm",
                turn_no, self.session_id,
            )
            # The goodbye still plays; we just hold the line briefly. If they
            # had genuinely already confirmed and have nothing to add, the
            # idle timer ends the call instead of burning the 600s cap.
            arm = getattr(media_session, "arm_idle_hangup", None)
            if arm is not None:
                arm()
        if outcome_this_turn:
            self._outcome_recorded_earlier = True

        # Never end the call, or hold the line, on silence. An empty reply is
        # a successful generation with nothing speakable in it (control tokens
        # only, or an unclosed <think> block the stripper discarded).
        if not said.strip():
            # An empty reply to "see you in 30 minutes, thank you" must not
            # become "could you repeat that?" — that is what happened live on
            # 2026-07-20 and it forced the shop to say goodbye twice.
            if they_closed and not hangup:
                hangup = True
                event_names.append("closed_by_callee")
            recovery = FALLBACK_GOODBYE_LINE if hangup else EMPTY_REPLY_LINE
            logger.warning(
                "Empty reply on turn %d for %s — speaking fallback %r",
                turn_no, self.session_id, recovery,
            )
            pcm, _ = await self._speak_sentence(recovery)
            said = recovery
            self.messages.append({"role": "assistant", "content": said})
            event_names.append("empty_reply")

        if hangup:
            media_session.request_hangup()

        self._record_turn(
            turn_no, heard, said, stt_ms, llm_ttft_ms, tts_ttfb_ms, t0, event_names
        )
        # Fire-and-forget: reporting to CC must never add to voice latency.
        task = asyncio.create_task(self.post_turn_event(self.turn_records[-1]))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return pcm if len(pcm) else None

    # ------------------------------------------------------------ escalation

    async def _run_escalation(
        self, question: str, media_session: Any, model_spoke: bool
    ) -> tuple[np.ndarray, str]:
        """Bounded wait for the user's answer; fallback + graceful end on
        timeout (PRD: the degradation path IS the expected P1 case)."""
        if not self.escalation.open():
            logger.warning("Second [ESCALATE] while one pending — falling back")
            pcm, _ = await self._speak_sentence(ESCALATION_FALLBACK_LINE)
            media_session.request_hangup()
            self.escalation_unanswered = True
            return pcm, ESCALATION_FALLBACK_LINE

        await self._post_escalation_event(question)
        hold_pcm = _EMPTY
        hold_said = ""
        if not model_spoke:
            hold_pcm, _ = await self._speak_sentence(HOLD_LINE)
            hold_said = HOLD_LINE
            if len(hold_pcm):
                await media_session.speak(hold_pcm)
                hold_pcm = _EMPTY  # already sent; don't double-play

        answer = await self.escalation.wait()
        if answer is None:
            self.escalation_unanswered = True
            self.outcome_facts.append(f"escalation unanswered: {question}")
            pcm, _ = await self._speak_sentence(ESCALATION_FALLBACK_LINE)
            media_session.request_hangup()
            return pcm, f"{hold_said} {ESCALATION_FALLBACK_LINE}".strip()

        self.messages.append(
            {
                "role": "user",
                "content": f"[{self.session.get('initiator_name') or 'The user'} "
                f"answered your question: {answer}]",
            }
        )
        pcm, said, events, _, _ = await self._generate_reply(media_session)
        if said:
            self.messages.append({"role": "assistant", "content": said})
        for ev in events:
            if isinstance(ev, Hangup):
                media_session.request_hangup()
            elif isinstance(ev, Outcome):
                self.outcome_facts.append(ev.facts)
        return pcm, f"{hold_said} {said}".strip()

    async def _post_escalation_event(self, question: str) -> None:
        if self.session_client is None:
            return
        try:
            await self.session_client.escalation_event(
                self.session_id, question, http=self.http
            )
        except Exception as e:  # noqa: BLE001 — CC being down must not kill the call
            logger.error("Escalation event post failed: %s", e)

    # ------------------------------------------------------------ records

    def _record_turn(
        self,
        turn_no: int,
        heard: str,
        said: str,
        stt_ms: float,
        llm_ttft_ms: float | None,
        tts_ttfb_ms: float | None,
        t0: float,
        event_names: list[str],
    ) -> None:
        record = TurnRecord(
            turn=turn_no,
            heard=heard,
            said=said,
            stt_ms=stt_ms,
            llm_ttft_ms=llm_ttft_ms,
            tts_ttfb_ms=tts_ttfb_ms,
            total_ms=(time.monotonic() - t0) * 1000,
            events=event_names,
        )
        self.turn_records.append(record)
        logger.info(
            "Turn %d session=%s stt=%.0fms ttft=%s ttfb=%s total=%.0fms events=%s",
            turn_no, self.session_id, record.stt_ms,
            f"{record.llm_ttft_ms:.0f}ms" if record.llm_ttft_ms else "-",
            f"{record.tts_ttfb_ms:.0f}ms" if record.tts_ttfb_ms else "-",
            record.total_ms, event_names or "-",
        )

    async def post_turn_event(self, record: TurnRecord) -> None:
        """Best-effort per-turn report to CC (doubles as the heartbeat)."""
        if self.session_client is None:
            return
        event = record.as_event()
        if self.redact_transcript:
            event.pop("heard", None)
            event.pop("said", None)
        try:
            await self.session_client.turn_event(
                self.session_id, event, http=self.http
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Turn event post failed: %s", e)

    def transcript(self) -> list[dict[str, str]]:
        """User-facing transcript (no system prompt, no tool bookkeeping)."""
        return [m for m in self.messages if m["role"] != "system"]
