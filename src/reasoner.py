"""
The actual "AI" part of the AI Finance Controller.

Everything in matcher.py is deterministic - it never guesses. This file is
for the cases matcher.py flagged but couldn't fully explain: why is this
settlement off by 12 rupees, why did this transaction never settle. A human
reconciliation analyst would look at these and reason about likely causes.
This file asks an LLM to do that reasoning, in plain English, with a
confidence level - instead of just leaving a bare "MISMATCH" flag.

DESIGN CHOICE: pluggable backend. If no API key is set, we fall back to a
MockReasoner that uses simple, transparent heuristics - so the pipeline is
always testable end-to-end, even without spending API credits. Swap in a
real key (Anthropic/OpenAI/Gemini) and the same code calls a real model.
This mirrors the mock_client.py pattern already in your orchestrator repo.
"""

import os
import json


class MockReasoner:
    """No API key needed. Deterministic, rule-based 'reasoning' so you can
    test the full pipeline for free. Clearly labeled as mock in its output -
    never pretend this is a real model's judgment."""

    name = "mock"

    def explain(self, case: dict) -> dict:
        kind = case["kind"]

        if kind == "AMOUNT_MISMATCH":
            diff = case["diff"]
            if abs(diff) <= 40:
                return {
                    "explanation": (
                        f"Settlement {case['settlement_id']} is off by Rs {diff:+.2f} "
                        f"from the expected Rs {case['expected_amount']:.2f}. Small enough "
                        f"that it's consistent with a fee/GST rounding or calculation error "
                        f"rather than a missing transaction."
                    ),
                    "likely_cause": "fee_calculation_error",
                    "confidence": "medium",
                    "recommended_action": "Route to finance for a fee recalculation check, not a full investigation.",
                }
            return {
                "explanation": f"Settlement {case['settlement_id']} is off by Rs {diff:+.2f} - too large to be a rounding issue.",
                "likely_cause": "unknown_large_discrepancy",
                "confidence": "low",
                "recommended_action": "Escalate for manual review - discrepancy too large to auto-explain.",
            }

        if kind == "ORPHAN":
            return {
                "explanation": (
                    f"Settlement {case['settlement_id']} references transaction(s) that don't "
                    f"exist anywhere in the Razorpay ledger. Either a data-sync gap between "
                    f"systems, or a transaction ID typo on the bank/settlement side."
                ),
                "likely_cause": "orphan_reference",
                "confidence": "low",
                "recommended_action": "Escalate to data engineering - possible sync issue between ledger and settlement feed.",
            }

        if kind == "MISSING_SETTLEMENT":
            return {
                "explanation": (
                    f"Transaction {case['transaction_id']} was captured but never appears in any "
                    f"settlement batch. Given typical T+2 settlement timing, if this transaction is "
                    f"more than a few days old, it's likely stuck or lost rather than just pending."
                ),
                "likely_cause": "unsettled_payment",
                "confidence": "medium",
                "recommended_action": "Flag to settlements team for follow-up; escalate if unresolved after 5 business days.",
            }

        return {
            "explanation": "Unrecognized case type.",
            "likely_cause": "unknown",
            "confidence": "low",
            "recommended_action": "Manual review required.",
        }


class AnthropicReasoner:
    """Real LLM reasoning via the Anthropic API. Requires ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, model="claude-sonnet-4-5-20250929"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def explain(self, case: dict) -> dict:
        prompt = f"""You are a financial reconciliation analyst reviewing an exception
from an automated matcher. Here is the case, already classified by deterministic logic:

{json.dumps(case, indent=2)}

Explain in 1-2 sentences why this likely happened, in plain English for a
non-technical merchant. Then give:
- likely_cause: a short snake_case label
- confidence: "high", "medium", or "low" - how sure you are of the cause
- recommended_action: one concrete next step

Respond ONLY with valid JSON: {{"explanation": "...", "likely_cause": "...", "confidence": "...", "recommended_action": "..."}}"""

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "explanation": text,
                "likely_cause": "unparsed",
                "confidence": "low",
                "recommended_action": "Manual review required (model response wasn't valid JSON).",
            }


def get_reasoner():
    """Auto-selects a real LLM if a key is present, otherwise the mock."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicReasoner()
        except Exception as e:
            print(f"[reasoner] Failed to init Anthropic client ({e}), falling back to mock.")
    return MockReasoner()
