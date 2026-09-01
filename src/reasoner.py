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
import re

from causes import (
    CAUSES_BY_KIND,
    CAUSE_DESCRIPTIONS,
    CAUSE_FEE_OR_GST,
    CAUSE_UNRECORDED_REFUND,
    CAUSE_DROPPED_TXN,
    CAUSE_OVERPAYMENT,
    CAUSE_ORPHAN,
    CAUSE_STUCK,
    CAUSE_NOT_YET_DUE,
)

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a local .env file if present - never commits, see .gitignore
except ImportError:
    pass  # dotenv not installed - still works if the key is set as a real env var


def build_explain_prompt(case: dict) -> str:
    """The shared prompt for every real-LLM backend.

    One prompt, four providers - so a difference in measured accuracy between
    Groq and Anthropic is a difference between the MODELS, not between two
    prompts that happened to be worded differently.
    """
    allowed = CAUSES_BY_KIND.get(case["kind"], [])
    menu = "\n".join(f'  - "{c}": {CAUSE_DESCRIPTIONS[c]}' for c in allowed)

    # Compact JSON, not indent=2. On a free API tier the binding constraint is
    # tokens-per-minute, and pretty-printing a case costs ~40% more tokens for
    # zero benefit to the model - which translates directly into rate-limit
    # backoff and a grading run that takes 15 minutes instead of 5.
    return f"""You are a financial reconciliation analyst reviewing an exception
flagged by an automated matcher. The matcher has already done the arithmetic;
your job is to work out WHY this happened.

THE CASE:
{json.dumps(case, separators=(",", ":"), default=str)}

The "signals" block contains factual arithmetic observations computed by the
matcher - what the gap is, whether it exactly equals one transaction's
contribution, whether it looks like a fraction of one, whether the bank paid
more or less than owed, and how the timing sits against the T+2 payout cycle.
Use them as evidence.

You must choose exactly one likely_cause from this list:
{menu}

Then give:
- explanation: 1-2 sentences in plain English for a non-technical merchant,
  referencing the actual amounts involved.
- confidence: "high", "medium", or "low" - be honest. If the evidence is
  genuinely ambiguous, say low. An overconfident wrong answer is worse than
  an honest uncertain one.
- recommended_action: one concrete next step. If nothing is actually wrong,
  say so rather than inventing work.

Respond ONLY with valid JSON:
{{"explanation": "...", "likely_cause": "...", "confidence": "...", "recommended_action": "..."}}"""


def parse_explanation(text: str, allowed_causes) -> dict:
    """Parse a model's JSON reply and refuse to silently accept a cause
    outside the taxonomy.

    Distinguishes three different failures that must NOT be scored the same
    way, because conflating them turns an infrastructure bug into a false
    finding about model quality:
      - empty_response: we got nothing back at all (token budget exhausted by
        the model's internal reasoning, truncation, a dropped call). The model
        never answered, so this is our bug, not its mistake.
      - unparsed: it answered, but not in JSON we can read.
      - off_taxonomy: it answered with a cause outside the allowed set.
    """
    text = (text or "").strip()
    if not text:
        return {
            "explanation": "",
            "likely_cause": "empty_response",
            "confidence": "low",
            "recommended_action": "Retry - the model returned no content.",
        }

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Models often wrap valid JSON in prose or a reasoning preamble.
        # Pull out the first balanced object rather than giving up on it.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            return {
                "explanation": text,
                "likely_cause": "unparsed",
                "confidence": "low",
                "recommended_action": "Manual review required (model response wasn't valid JSON).",
            }

    cause = parsed.get("likely_cause")
    if cause not in allowed_causes:
        parsed["likely_cause"] = "off_taxonomy"
        parsed.setdefault("explanation", "")
        parsed["explanation"] = (
            f"[model returned an out-of-taxonomy cause: {cause!r}] "
            + str(parsed.get("explanation", ""))
        )
    parsed.setdefault("confidence", "low")
    parsed.setdefault("recommended_action", "Manual review required.")
    return parsed


class RuleBaselineReasoner:
    """A deliberately dumb, transparent rule-based classifier over the same
    signals the LLM sees. Serves two purposes:

    1. It means the whole pipeline runs with no API key, for free.
    2. It is the CONTROL for the AI accuracy metric. "Our LLM got X% on root
       cause attribution" is only interesting next to "and a handful of
       if-statements got Y%". If a judge asks "would a simple rule have done
       this?", the honest answer is a number, not a shrug.

    Note what this baseline cannot do, and why it isn't a replacement for the
    model: every rule below exists because we knew in advance exactly which
    four causes we injected. On real data, a cause nobody enumerated ahead of
    time falls straight through to the else-branch. It also produces no
    merchant-readable explanation beyond a template, and cannot answer an
    open-ended question at all.
    """

    name = "rule_baseline"

    def explain(self, case: dict) -> dict:
        kind = case["kind"]
        signals = case.get("signals") or {}

        if kind == "ORPHAN":
            return {
                "explanation": (
                    f"Settlement {case['settlement_id']} references transaction(s) that don't "
                    f"exist anywhere in the ledger."
                ),
                "likely_cause": CAUSE_ORPHAN,
                "confidence": "high",
                "recommended_action": "Escalate to data engineering - likely a sync gap between ledger and settlement feed.",
            }

        if kind == "MISSING_SETTLEMENT":
            overdue = signals.get("due_date_passed_within_data_window")
            if overdue is False:
                return {
                    "explanation": (
                        f"Transaction {case['transaction_id']} was captured on "
                        f"{signals.get('transaction_timestamp')} and isn't due to settle until "
                        f"{signals.get('settlement_due_by')}, after this reporting window closes."
                    ),
                    "likely_cause": CAUSE_NOT_YET_DUE,
                    "confidence": "high",
                    "recommended_action": "No action needed - not yet due under the T+2 cycle.",
                }
            return {
                "explanation": (
                    f"Transaction {case['transaction_id']} was due to settle by "
                    f"{signals.get('settlement_due_by')} and still has no settlement "
                    f"{signals.get('days_past_due_at_window_end')} day(s) later."
                ),
                "likely_cause": CAUSE_STUCK,
                "confidence": "high",
                "recommended_action": "Flag to the settlements team - payout appears stuck.",
            }

        if kind == "AMOUNT_MISMATCH":
            diff = case.get("diff", 0)
            exact = signals.get("exact_match_to_one_transaction")
            fraction = signals.get("refund_sized_fraction_of_one_transaction")

            if signals.get("direction") == "bank_paid_more_than_owed":
                return {
                    "explanation": f"Settlement {case['settlement_id']} came in Rs {diff:+.2f} ABOVE what was owed.",
                    "likely_cause": CAUSE_OVERPAYMENT,
                    "confidence": "high" if exact else "medium",
                    "recommended_action": "Confirm with the bank before spending - overpayments get clawed back.",
                }
            if exact:
                return {
                    "explanation": (
                        f"The Rs {abs(diff):.2f} gap is exactly what {exact['transaction_id']} "
                        f"should have contributed, so that payment appears not to have been paid out."
                    ),
                    "likely_cause": CAUSE_DROPPED_TXN,
                    "confidence": "high",
                    "recommended_action": f"Query {exact['transaction_id']} with the settlements team.",
                }
            if fraction and not fraction.get("has_recorded_refund"):
                return {
                    "explanation": (
                        f"The Rs {abs(diff):.2f} gap is {fraction['pct_of_net']}% of "
                        f"{fraction['transaction_id']}'s net - refund-shaped, but no refund is recorded."
                    ),
                    "likely_cause": CAUSE_UNRECORDED_REFUND,
                    "confidence": "medium",
                    "recommended_action": f"Check for an unrecorded refund against {fraction['transaction_id']}.",
                }
            return {
                "explanation": (
                    f"Settlement {case['settlement_id']} is short by Rs {abs(diff):.2f}, which isn't "
                    f"tied to any single transaction's amount - consistent with a fee/GST calculation error."
                ),
                "likely_cause": CAUSE_FEE_OR_GST,
                "confidence": "medium",
                "recommended_action": "Route to finance for a fee/GST recalculation check.",
            }

        return {
            "explanation": "Unrecognized case type.",
            "likely_cause": "unknown",
            "confidence": "low",
            "recommended_action": "Manual review required.",
        }

    def chat(self, prompt: str) -> str:
        return ("[rule baseline has no free-form Q&A - it only classifies exceptions "
                "into fixed categories. Add a real API key to use the Q&A agent.]")


# Backwards-compatible alias: this used to be called MockReasoner.
MockReasoner = RuleBaselineReasoner


class AnthropicReasoner:
    """Real LLM reasoning via the Anthropic API. Requires ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, model="claude-sonnet-4-5-20250929"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def explain(self, case: dict) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": build_explain_prompt(case)}],
        )
        return parse_explanation(resp.content[0].text, CAUSES_BY_KIND.get(case["kind"], []))

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
        resp = self.client.models.generate_content(
            model=self.model_name, contents=build_explain_prompt(case)
        )
        return parse_explanation(resp.text, CAUSES_BY_KIND.get(case["kind"], []))

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

    def _complete(self, prompt: str, json_mode: bool, max_tokens: int) -> str:
        """One call, with graceful degradation for parameters this particular
        model/tier may not support (json mode, reasoning_effort). Unsupported
        params raise rather than being ignored, so we retry without them."""
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            resp = self.client.chat.completions.create(**kwargs)

        message = resp.choices[0].message
        content = (message.content or "").strip()
        if not content:
            # Reasoning-style models (gpt-oss) can put everything in a separate
            # reasoning channel and return empty content. The answer may still
            # be in there.
            content = (getattr(message, "reasoning", None) or "").strip()
        return content

    def explain(self, case: dict) -> dict:
        """Retries an EMPTY response before scoring it.

        This mattered: gpt-oss-20b silently returned empty content on ~38% of
        a hard-case run because it spent its (unset, therefore default) token
        budget on internal reasoning. Those were being recorded as wrong
        answers, which would have understated the model badly and turned our
        own missing max_tokens into a published finding about AI accuracy.
        """
        prompt = build_explain_prompt(case)
        allowed = CAUSES_BY_KIND.get(case["kind"], [])

        for attempt, (json_mode, max_tokens) in enumerate(
            [(True, 1200), (False, 2000), (False, 3000)], start=1
        ):
            text = self._complete(prompt, json_mode=json_mode, max_tokens=max_tokens)
            result = parse_explanation(text, allowed)
            if result["likely_cause"] not in ("empty_response", "unparsed"):
                return result
            last = result
        return last

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
