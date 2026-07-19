# jarvis-phone-gateway

Telephony media bridge (port **7713**) for AI phone calls: terminates Twilio
bidirectional Media Streams over public wss, owns the per-call media loop
(mu-law ↔ PCM, VAD endpointing, whisper → llm-proxy → tts turns), and
reports every session to jarvis-command-center. **The gateway never writes
CC's Postgres** — all state flows through CC's `/internal/phone/*` API.

> Design source of truth: the phone-calls PRD (`jarvis/prds/phone-calls.md`)
> — locked decisions, security requirements, latency budget, phases. Read it
> before changing anything architectural. P0 spike evidence lives at
> `~/jarvis-spikes/phone-call-p0/`.

**Status: P1 foundation.** Everything below the turn pipeline is built and
tested; the live whisper/LLM/TTS wiring, recording → MinIO, and the dial
worker loop are the next phase (TODOs at the bottom).

---

## Topology

```
CC make_phone_call tool ──▶ Redis phone:dial ──▶ dial worker (this service)
                                                   │ claim_for_dial (CC CAS
                                                   │  confirmed→dialing)
                                                   ▼
                          Twilio calls.create w/ TwiML <Connect><Stream
                              url=wss://PUBLIC_URL/media/{single-use token}>
                                                   │
Twilio Media Streams ◀──────── wss ───────────────┘
   │  base64 mu-law 8k frames, both directions
   ▼
/media/{token} WS  ─▶ token claim ─▶ X-Twilio-Signature (scheme-fixed)
   ─▶ stream-start binding (session_id param + callSid)
   ─▶ MediaStreamSession: codec → VAD → turn pipeline → frames out
        │                                   │
        │ per-turn events + heartbeat       ├─▶ whisper /transcribe (WAV, linear PCM)
        ▼                                   ├─▶ llm-proxy /v1/chat/completions (stream)
CC /internal/phone/sessions/{id}/events     └─▶ tts /speak/stream (guarded)
```

## Module map

| Module | What it owns |
|---|---|
| `audio/mulaw.py` | G.711 codec, resampling, whisper-safe WAV wrapping (all numpy/scipy — no audioop) |
| `audio/vad.py` | RMS VAD + hangover endpointing, pure state machine (spike defaults: 250 RMS / 800 ms) |
| `telephony/provider.py` | Vendor seam (PRD decision 7). Event vocabulary: StreamStart / InboundAudio / MarkReceived / StreamStop |
| `telephony/twilio_provider.py` | TwiML, WS wire format, REST dial/hangup, signature validation, `SessionTokenRegistry` |
| `llm/client.py` | llm-proxy stream client ({delta}/{done} frames), 20 s turn cap **with upstream cancel**, sentence regrouping |
| `llm/tool_tokens.py` | `[HANGUP]` `[ESCALATE:]` `[OUTCOME:]` `[DTMF:]` streaming parser |
| `llm/think_strip.py` | `<think>` block stripping on the token stream |
| `services/tts_guard.py` | TTS content-type gate + per-response sample rate + odd-byte-safe PCM chunking |
| `services/session_client.py` | CC contract: claim_dial CAS, state/turn/heartbeat/outcome events |
| `services/media_stream.py` | Per-call session: WS loop composing codec + VAD + turn pipeline |
| `queues/dial_queue.py` | `phone:dial` consumer, strict job parsing |
| `main.py` | `create_app()` factory: /health + `/media/{token}` with the three WS gates |
| `config.py` | Env config (bootstrap + secrets only) |

---

## Invariants & gotchas

1. **The dial queue is transport, never authorization** (PRD security req 1).
   Stack Redis is unauthenticated; anyone on jarvis-net can LPUSH. A popped
   job is only `{session_id, household_id}` — extra fields are discarded at
   parse. The ONLY authorization is CC's atomic `confirmed → dialing`
   compare-and-set (`SessionClient.claim_for_dial`); 409 ⇒ drop the job.
2. **Three gates on the media WS, in order** (PRD security req 2): single-use
   path token (claim pops — replays and duplicate streams die pre-handshake),
   `X-Twilio-Signature` with the **https↔wss scheme fix** (Twilio signs the
   wss URL; naive https-only reconstruction always fails — there's a test
   proving the trap), then stream-start binding (session_id custom parameter
   + callSid). Signature validation is on iff `TWILIO_AUTH_TOKEN` is set.
3. **Never hand whisper mu-law.** `/transcribe` 500s on a mu-law WAV. Decode
   to linear PCM16 first (`pcm8k_to_whisper_wav`); whisper resamples 8 kHz
   internally.
4. **Every TTS response goes through the guard.** jarvis-tts returns HTTP 200
   with a JSON error body on empty text — and think-strip/[HANGUP] removal
   legitimately produces empty fragments. Unguarded, those JSON bytes become
   audio on the live call (P0 failure ladder item 6). Also: sample rate is
   re-read from `X-Audio-Sample-Rate` per response (real Piper voice is
   16 kHz; docs claiming 22 050 are wrong).
5. **The turn cap cancels, never abandons.** llm-proxy streams are
   `{"delta"}/{"done"}` frames (NOT OpenAI chunks); every stream carries our
   `X-Request-Id` and timeout fires `POST /v1/chat/completions/cancel/{id}`
   before raising. An abandoned stream is exactly the model-service wedge
   from the P0 incident (fixed llm-proxy-side in PR #47, but don't rely on
   the backstop).
6. **Think blocks are never spoken.** The live model is a thinking model;
   `/no_think` helps but is not sufficient. `ThinkStripper` runs on every
   token stream; unclosed blocks never leak.
7. **Half-duplex v1.** While the agent speaks, inbound audio is suppressed at
   the VAD (`suppress=True`) so we don't endpoint on our own playback echo.
   Barge-in (`clear` + `mark` tracking) is P2 — the provider messages exist.
8. **Twilio facts** (verified P0): bidirectional streams require TwiML
   `<Connect><Stream>` via calls.create (REST Streams subresource is
   unidirectional-only); frames are headerless base64 mu-law 8 kHz, 160
   bytes / 20 ms.
9. **Secrets stay in env** (PRD security req 5): Twilio SID/token/number are
   gateway-only — never the settings DB, never CC. The auth token is also
   the signature-verification key.
10. **Ingress is a named Cloudflare tunnel hostname** — quick tunnels proved
    unfit during P0 (LAN DNS interference, config hijack, edge flakes).

---

## Testing

```bash
.venv/bin/python -m pytest        # full suite, no external services needed
```

- Unit: codec (spike self-test ported), VAD state machine, TwiML/wire
  format, signature scheme fix, tool tokens, think strip, TTS guard, dial
  queue, CC client (MockTransport).
- `tests/test_media_ws.py` is the **fake-Twilio WS fixture**: a scripted
  media exchange over the real endpoint (real gates, real VAD, stubbed turn
  pipeline).
- Replay fixtures (spike utterance WAVs) and the live Twilio smoke lane are
  planned for the live-wiring phase (see PRD test strategy).

---

## Config surface (env — bootstrap + secrets only)

| Variable | Purpose |
|---|---|
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM_NUMBER` | Telephony credentials (gateway-only) |
| `PUBLIC_URL` | This worker's public https base (named tunnel); TwiML wss URLs + signature checks derive from it |
| `CC_BASE_URL` / `WHISPER_URL` / `LLM_URL` / `TTS_URL` | Upstream services |
| `JARVIS_APP_ID` / `JARVIS_APP_KEY` | App-to-app credentials (gateway gets its own pair in jarvis-auth) |
| `JARVIS_CONFIG_URL` | Service discovery (wiring TODO) |
| `REDIS_URL` | Dial queue |
| `SERVER_HOST` / `SERVER_PORT` (7713) | Bind |
| `VAD_RMS` (250) | Endpointing threshold |

Non-secret runtime knobs (`phone_calls.*` caps, TTLs, retention) live in the
settings DB and are CC-owned; the gateway learns them via session payloads.

---

## TODO — live-wiring phase (deliberately not built yet)

- The real turn pipeline: whisper → llm client (+ think strip + tool tokens)
  → tts guard → `MediaStreamSession.speak`, with per-turn stage timings
  emitted via `SessionClient.turn_event` (the observability section's
  metrics.json format).
- Dial worker loop: `DialQueue.pop` → `claim_for_dial` → issue token →
  `build_stream_instructions` → `start_call` → reaper-friendly heartbeats.
- Local recording (both directions) → mixed WAV → MinIO, and the outcome
  event carrying the audio key.
- Disclosure first turn + system prompt assembly from the session's details
  brief (guardrail boundary: the brief is all the agent may say).
- `jarvis-config-client` discovery + `JarvisLogger` (jarvis-log-client)
  wiring; `settings_definitions.py` if any gateway-local settings emerge.
- Escalation/cancel inbound endpoints (`/internal/call/{session_id}/...`)
  once the CC side of the contract exists.
- Service-integration checklist (config-service `known_services.py`, `jarvis`
  CLI arrays, installer/admin compose generators, jarvis-auth app pair).
