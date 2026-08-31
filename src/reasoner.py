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

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a local .env file if present - never commits, see .gitignore
except ImportError:
    pass  # dotenv not installed - still works if the key is set as a real env var


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

    def chat(self, prompt: str) -> str:
        return ("[mock reasoner has no free-form Q&A - it only knows the canned "
                "exception explanations above. Add a real API key to use the Q&A agent.]")


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

    def chat(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


class GeminiReasoner:
    """Real LLM reasoning via Google's Gemini API. Requires GEMINI_API_KEY.

    Uses the current `google-genai` SDK (the old `google-generativeai`
    package is deprecated as of 2026 - don't use it).

    Model ID defaults to gemini-2.5-flash (a stable, well-documented, cheap
    model as of writing). Google renames/retires models over time - if this
    ID ever fails with a "not found" error, check the current list at
    https://ai.google.dev/gemini-api/docs/models and set GEMINI_MODEL in
    your .env to override the default below.
    """

    name = "gemini"

    def __init__(self, model=None):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

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

        resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "explanation": text,
                "likely_cause": "unparsed",
                "confidence": "low",
                "recommended_action": "Manual review required (model response wasn't valid JSON).",
            }

    def chat(self, prompt: str) -> str:
        resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return resp.text.strip()


class GroqReasoner:
    """Real LLM reasoning via Groq's API (OpenAI-compatible, very fast
    inference, generous free tier). Requires GROQ_API_KEY.

    Model defaults to openai/gpt-oss-20b - confirmed available on Groq's
    free tier as of this build (verified directly via client.models.list(),
    not just docs - Groq's docs listed models our actual key didn't have
    access to, so trust the API over the docs if this ever mismatches
    again). Set GROQ_MODEL in your .env to override.
    """

    name = "groq"

    def __init__(self, model=None):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model_name = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

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

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "explanation": text,
                "likely_cause": "unparsed",
                "confidence": "low",
                "recommended_action": "Manual review required (model response wasn't valid JSON).",
            }

    def chat(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


def get_reasoner():
    """Auto-selects a real LLM if a key is present, otherwise the mock.
    Checks Anthropic first, then Groq, then Gemini, then falls back."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicReasoner()
        except Exception as e:
            print(f"[reasoner] Failed to init Anthropic client ({e}), falling back to mock.")
    if os.environ.get("GROQ_API_KEY"):
        try:
            return GroqReasoner()
        except Exception as e:
            print(f"[reasoner] Failed to init Groq client ({e}), falling back to mock.")
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return GeminiReasoner()
        except Exception as e:
            print(f"[reasoner] Failed to init Gemini client ({e}), falling back to mock.")
    return MockReasoner()
