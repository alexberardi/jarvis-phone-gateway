"""Prompt assembly: the compliance-critical strings (PRD decision 10)."""

from services.prompt import (
    DISCLOSURE_TEMPLATE,
    build_disclosure,
    build_system_prompt,
    initial_messages,
)

SESSION = {
    "id": "sess-1",
    "initiator_name": "Alex",
    "goal": "Book a table for 4 on Friday at 7pm",
    "details": "Party of 4. Friday 7pm preferred. Name: Alex.",
    "constraints": "acceptable: Fri 6-8pm; conflict: Fri 6:30",
}


class TestDisclosure:
    def test_names_the_user_and_mentions_ai_and_recording(self):
        d = build_disclosure(SESSION)
        assert "Alex" in d
        assert "AI assistant" in d
        assert "recorded" in d

    def test_falls_back_to_a_customer(self):
        d = build_disclosure({})
        assert "a customer" in d
        assert d == DISCLOSURE_TEMPLATE.format(user="a customer")


class TestSystemPrompt:
    def test_contains_goal_details_and_envelope(self):
        p = build_system_prompt(SESSION)
        assert "Book a table for 4" in p
        assert "Party of 4" in p
        assert "Fri 6-8pm" in p

    def test_compliance_rules_always_present(self):
        p = build_system_prompt({})
        assert "truthfully" in p  # honest is-this-a-robot
        assert "payment" in p.lower()  # never payment data
        assert "[HANGUP]" in p and "[ESCALATE:" in p and "[OUTCOME:" in p

    def test_brief_is_declared_the_complete_boundary(self):
        p = build_system_prompt(SESSION)
        assert "COMPLETE set of facts" in p

    def test_omits_empty_sections(self):
        p = build_system_prompt({"goal": "", "details": "", "constraints": ""})
        assert "Your goal for this call" not in p
        assert "Acceptable options" not in p


class TestInitialMessages:
    def test_system_then_spoken_disclosure(self):
        msgs = initial_messages(SESSION)
        assert [m["role"] for m in msgs] == ["system", "assistant"]
        assert msgs[1]["content"] == build_disclosure(SESSION)
