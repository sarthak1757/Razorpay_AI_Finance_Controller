"""
Ties the deterministic matcher to the reasoning layer.

Flow:
  1. Run matcher.py's rule-based matching (fast, certain, no AI).
  2. Pull out only the cases it couldn't cleanly resolve as MATCHED/SETTLED.
     (DUPLICATE is excluded on purpose - the matcher is already fully
     confident about those, no reasoning needed, don't waste an LLM call.)
  3. Hand each one to the reasoner (real LLM or mock) for a plain-English
     explanation, a likely cause, a confidence level, and a next action.

This file is what turns "AMOUNT_MISMATCH" from a bare flag into something a
human (or a judge) can actually read and trust.
"""

from matcher import load_data, match
from reasoner import get_reasoner


def build_cases(settlement_results, transaction_results):
    cases = []

    for r in settlement_results:
        if r["status"] == "AMOUNT_MISMATCH":
            cases.append({
                "kind": "AMOUNT_MISMATCH",
                "settlement_id": r["settlement_id"],
                "expected_amount": r["expected_amount"],
                "actual_amount": r["actual_amount"],
                "diff": r["diff"],
                "included_transactions": r["included_transactions"],
                # Arithmetic evidence from the matcher - what the reasoner
                # actually reasons OVER. Without this it would be guessing.
                "signals": r.get("signals"),
            })
        elif r["status"] == "ORPHAN":
            cases.append({
                "kind": "ORPHAN",
                "settlement_id": r["settlement_id"],
                "actual_amount": r["actual_amount"],
                "included_transactions": r["included_transactions"],
            })

    for tid, r in transaction_results.items():
        if r["status"] == "MISSING_SETTLEMENT":
            cases.append({
                "kind": "MISSING_SETTLEMENT",
                "transaction_id": tid,
                "signals": r.get("signals"),
            })

    return cases


def run():
    transactions, settlements, refunds = load_data()
    settlement_results, transaction_results = match(transactions, settlements, refunds)
    cases = build_cases(settlement_results, transaction_results)

    reasoner = get_reasoner()
    print(f"Using reasoner: {reasoner.name}")
    print(f"Investigating {len(cases)} exceptions...\n")

    results = []
    for case in cases:
        explanation = reasoner.explain(case)
        results.append({**case, **explanation})
        label = case.get("settlement_id") or case.get("transaction_id")
        print(f"[{case['kind']}] {label}")
        print(f"  -> {explanation['explanation']}")
        print(f"     cause={explanation['likely_cause']} confidence={explanation['confidence']}")
        print(f"     action: {explanation['recommended_action']}\n")

    return settlement_results, transaction_results, results


if __name__ == "__main__":
    run()
