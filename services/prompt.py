"""Call-brain prompt assembly (PRD decisions 6 + 10, security requirement 4).

Non-negotiables encoded here:
- The DISCLOSURE is always the first thing the callee hears — AI + recording
  notice in one sentence (Duplex precedent; state all-party consent posture).
- Honest "yes" if asked is-this-a-robot (Utah SB149 if-asked duty, FTC §5).
- Instant wrap-up on a hang-up request; per-business do-not-call is honored
  upstream at resolve time.
- The details brief is the guardrail boundary: what's in it is all the agent
  may say and pursue. Payment card data is never read, collected, or
  confirmed — orders are "pickup, pay at the counter".
- Tool protocol is plain text tokens ([HANGUP], [ESCALATE: q], [OUTCOME: f])
  because llm-proxy's streaming path drops native tools (decision 6).
"""

from __future__ import annotations

from typing import Any

DISCLOSURE_TEMPLATE = (
    "Hi, I'm an automated AI assistant calling on behalf of {user}. "
    "This call may be recorded."
)

HOLD_LINE = "Let me check on that — one moment, please."
ESCALATION_FALLBACK_LINE = (
    "I couldn't confirm that right now. Let me check and call you back. "
    "Thank you for your time, goodbye."
)
TURN_FAILURE_LINE = "Sorry, I'm having a little trouble — could you say that again?"


def build_disclosure(session: dict[str, Any]) -> str:
    """First agent turn, spoken before anything else. Never skipped."""
    user = session.get("initiator_name") or "a customer"
    return DISCLOSURE_TEMPLATE.format(user=user)


def build_system_prompt(session: dict[str, Any]) -> str:
    """System prompt for the live model, built ONLY from the session brief."""
    goal = (session.get("goal") or "").strip()
    details = (session.get("details") or "").strip()
    envelope = (session.get("constraints") or "").strip()
    user = session.get("initiator_name") or "a customer"

    sections = [
        "You are Jarvis, an automated AI assistant making a real phone call "
        f"on behalf of {user}. You are speaking with a business over the "
        "phone. Keep replies to one or two short, natural sentences — this "
        "is a spoken conversation, not text. /no_think",
        f"Your goal for this call: {goal}" if goal else "",
        (
            "Details you may use — this brief is the COMPLETE set of facts "
            "you may state or act on; do not invent, promise, or agree to "
            f"anything outside it:\n{details}"
        )
        if details
        else "",
        (
            "Acceptable options and constraints (negotiate only within "
            f"these):\n{envelope}"
        )
        if envelope
        else "",
        "Rules you must never break:\n"
        "- If asked whether you are a robot, an AI, or a real person, answer "
        "truthfully that you are an AI assistant.\n"
        "- If the person asks you to stop calling or to hang up, apologize "
        "briefly, say goodbye, and emit [HANGUP].\n"
        "- Never give, read, confirm, or discuss payment card numbers or any "
        "payment credentials. If payment is required, say it will be handled "
        "at pickup, in person.\n"
        "- Never share personal information beyond what the brief contains.",
        "Tools — emit these tokens in your reply text when needed:\n"
        "- [HANGUP] — end the call after your current sentence (say a natural "
        "goodbye first).\n"
        "- [ESCALATE: question] — the other person asked something the brief "
        "does not answer; ask them to hold while you check.\n"
        "- [OUTCOME: facts] — record a concrete result the moment it is "
        "confirmed (e.g. [OUTCOME: booked Friday 7pm party of 4 under Alex]).\n"
        "When the goal is achieved or clearly impossible, confirm, record the "
        "[OUTCOME: ...], say goodbye, and emit [HANGUP].",
    ]
    return "\n\n".join(s for s in sections if s)


def initial_messages(session: dict[str, Any]) -> list[dict[str, str]]:
    """Conversation seed: system prompt + the already-spoken disclosure.

    The disclosure is inserted as the first assistant turn so the model
    knows it was said and never re-introduces itself.
    """
    return [
        {"role": "system", "content": build_system_prompt(session)},
        {"role": "assistant", "content": build_disclosure(session)},
    ]
