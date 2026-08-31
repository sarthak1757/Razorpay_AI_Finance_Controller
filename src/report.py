"""
Pulls the matcher + exception agent together into the actual deliverable:
a match rate, a throughput number, and an honest exception list - exactly
what Track 4's bar asks for: "Throughput plus measured accuracy plus an
honest exception list. One cherry-picked match proves nothing."

This file computes real numbers off the full batch, not a hand-picked
example, and saves them to data/report.json so they can be dropped straight
into the pitch video / architecture doc later.
"""

import json
import time

from matcher import load_data, match
from exception_agent import build_cases
from reasoner import get_reasoner


def generate_report():
    start = time.perf_counter()

    transactions, settlements, refunds = load_data()
    settlement_results, transaction_results = match(transactions, settlements, refunds)
    cases = build_cases(settlement_results, transaction_results)

    reasoner = get_reasoner()
    exceptions = []
    for case in cases:
        explanation = reasoner.explain(case)
        exceptions.append({**case, **explanation})

    elapsed = time.perf_counter() - start

    total_settlements = len(settlement_results)
    matched_settlements = sum(1 for r in settlement_results if r["status"] == "MATCHED")

    duplicate_ids = {tid for tid, r in transaction_results.items() if r["status"] == "DUPLICATE"}
    real_transactions = {tid: r for tid, r in transaction_results.items() if tid not in duplicate_ids}
    settled_transactions = sum(1 for r in real_transactions.values() if r["status"] == "SETTLED")

    txn_by_id = {t["transaction_id"]: t for t in transactions}
    amount_at_risk = sum(
        txn_by_id[tid]["net_amount"]
        for tid, r in transaction_results.items()
        if r["status"] == "MISSING_SETTLEMENT"
    )
    amount_flagged_mismatch = sum(
        abs(r["diff"]) for r in settlement_results if r["status"] == "AMOUNT_MISMATCH"
    )

    total_records = len(transactions) + len(settlements)

    report = {
        "throughput": {
            "total_records_processed": total_records,
            "transactions": len(transactions),
            "settlements": len(settlements),
            "processing_time_seconds": round(elapsed, 4),
            "records_per_second": round(total_records / elapsed, 1) if elapsed > 0 else None,
        },
        "accuracy": {
            "settlement_match_rate": round(matched_settlements / total_settlements, 4),
            "settlements_matched": matched_settlements,
            "settlements_total": total_settlements,
            "transaction_settlement_rate": round(settled_transactions / len(real_transactions), 4),
            "transactions_settled": settled_transactions,
            "transactions_total_excl_duplicates": len(real_transactions),
            "duplicate_entries_flagged": len(duplicate_ids),
        },
        "money_flagged": {
            "amount_at_risk_unsettled": round(amount_at_risk, 2),
            "amount_flagged_mismatch": round(amount_flagged_mismatch, 2),
            "currency": "INR",
        },
        "exceptions": exceptions,
        "reasoner_used": reasoner.name,
    }

    with open("data/report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_summary(report):
    t = report["throughput"]
    a = report["accuracy"]
    m = report["money_flagged"]

    print("=" * 60)
    print("RECONCILIATION REPORT")
    print("=" * 60)
    print(f"\nThroughput:")
    print(f"  {t['total_records_processed']} records processed in {t['processing_time_seconds']}s "
          f"({t['records_per_second']} records/sec)")
    print(f"  ({t['transactions']} transactions, {t['settlements']} settlements)")

    print(f"\nAccuracy:")
    print(f"  Settlement match rate: {a['settlement_match_rate']*100:.1f}% "
          f"({a['settlements_matched']}/{a['settlements_total']})")
    print(f"  Transaction settlement rate: {a['transaction_settlement_rate']*100:.1f}% "
          f"({a['transactions_settled']}/{a['transactions_total_excl_duplicates']}, duplicates excluded)")
    print(f"  Duplicate entries flagged: {a['duplicate_entries_flagged']}")

    print(f"\nMoney flagged for review:")
    print(f"  Rs {m['amount_at_risk_unsettled']:.2f} sitting in unsettled transactions")
    print(f"  Rs {m['amount_flagged_mismatch']:.2f} in amount discrepancies across mismatched settlements")

    print(f"\nHonest exception list ({len(report['exceptions'])} items, reasoner: {report['reasoner_used']}):")
    for e in report["exceptions"]:
        label = e.get("settlement_id") or e.get("transaction_id")
        print(f"  [{e['kind']}] {label} - {e['likely_cause']} (confidence: {e['confidence']})")

    print("\n(Full detail saved to data/report.json)")


if __name__ == "__main__":
    report = generate_report()
    print_summary(report)
