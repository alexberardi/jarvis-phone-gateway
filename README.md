# jarvis-phone-gateway

Telephony media bridge for Jarvis AI phone calls (port **7713**): terminates
Twilio bidirectional Media Streams and runs each call's media loop
(mu-law ↔ PCM, VAD endpointing, whisper → LLM → TTS turns), reporting every
session to jarvis-command-center.

Status: **P1 foundation** — audio codec/VAD, Twilio provider + security
gates, LLM stream client with tool-token protocol, CC session contract
client, and the media WebSocket skeleton are built and tested. The live
turn pipeline wiring is the next phase.

```bash
./run.sh                       # dev server on :7713
.venv/bin/python -m pytest     # test suite (no external services needed)
```

See `CLAUDE.md` for architecture, invariants, and the full design context
(PRD: `jarvis/prds/phone-calls.md`).
