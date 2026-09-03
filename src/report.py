"""
Pulls the matcher + exception agent together into the actual deliverable:
a match rate, a throughput number, and an honest exception list - exactly
what Track 4's bar asks for: "Throughput plus measured accuracy plus an
honest exception list. One cherry-picked match proves nothing."

This file computes real numbers off the full batch, not a hand-picked
example, and saves them to data/report.json so they can be dropped straight
into the pitch video / architecture doc later.
"""

import argparse
import json
import os
import time

from matcher import load_data, match
from exception_agent import build_cases
from reasoner import get_reasoner


def generate_report(data_dir="data", out_path=None):
    """Times the two phases SEPARATELY, on purpose.

    An earlier version reported one blended "records/sec" - total wall clock
    divided by all records. That number was misleading in both directions: it
    made the deterministic matcher look thousands of times slower than it is,
    while implying every record pays LLM latency. Neither is true. Only the
    handful of flagged exceptions ever reach the model; the other ~90% are
    pure arithmetic. Reporting one number hides the actual design decision -
    that AI is used sparingly and deliberately - and invites a fair question
    ("why is this so slow?") whose honest answer is "it isn't".
    """
    load_start = time.perf_counter()
    transactions, settlements, refunds = load_data(data_dir)
    load_seconds = time.perf_counter() - load_start

    match_start = time.perf_counter()
    settlement_results, transaction_results = match(transactions, settlements, refunds)
    match_seconds = time.perf_counter() - match_start

    cases = build_cases(settlement_results, transaction_results)

    reasoner = get_reasoner()
    reason_start = time.perf_counter()
    exceptions = []
    for case in cases:
        explanation = reasoner.explain(case)
        exceptions.append({**case, **explanation})
    reason_seconds = time.perf_counter() - reason_start

    elapsed = load_seconds + match_seconds + reason_seconds

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
            "total_wall_clock_seconds": round(elapsed, 4),
            # Phase 1: deterministic. Scales with record count. This is the
            # number that matters for "can this handle a real merchant's volume".
            "deterministic_matching": {
                "seconds": round(match_seconds, 6),
                "records_per_second": round(total_records / match_seconds) if match_seconds > 0 else None,
            },
            # Phase 2: AI. Scales with EXCEPTION count, not record count -
            # which is the whole point of the architecture.
            "ai_reasoning": {
                "exceptions_reasoned": len(cases),
                "seconds": round(reason_seconds, 4),
                "seconds_per_exception": round(reason_seconds / len(cases), 3) if cases else None,
                "pct_of_records_needing_ai": round(len(cases) / total_records * 100, 2) if total_records else None,
            },
            "data_load_seconds": round(load_seconds, 4),
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
        "data_dir": data_dir,
    }

    out_path = out_path or os.path.join(data_dir, "report.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    report["_out_path"] = out_path

    return report


def print_summary(report):
    t = report["throughput"]
    a = report["accuracy"]
    m = report["money_flagged"]

    print("=" * 60)
    print("RECONCILIATION REPORT")
    print("=" * 60)
    d, ai = t["deterministic_matching"], t["ai_reasoning"]
    print(f"\nThroughput ({t['total_records_processed']} records: "
          f"{t['transactions']} transactions + {t['settlements']} settlements):")
    print(f"  Deterministic matching: {d['seconds']:.4f}s "
          f"({d['records_per_second']:,} records/sec)")
    print(f"  AI reasoning:           {ai['seconds']:.2f}s for {ai['exceptions_reasoned']} exceptions "
          f"({ai['seconds_per_exception']}s each)")
    print(f"  Only {ai['pct_of_records_needing_ai']}% of records ever reach the model - "
          f"the rest are pure arithmetic.")

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

    print(f"\n(Full detail saved to {report.get('_out_path', 'data/report.json')})")


def main():
    p = argparse.ArgumentParser(
        description="Reconcile a transaction ledger against a bank settlement feed.")
    p.add_argument("--data-dir", default="data",
                   help="Directory holding razorpay_transactions.csv, bank_settlements.csv "
                        "and (optionally) refunds.csv. Defaults to ./data")
    p.add_argument("--out", default=None,
                   help="Where to write report.json. Defaults to <data-dir>/report.json")
    p.add_argument("--html", action="store_true",
                   help="Also render report.html next to the JSON.")
    args = p.parse_args()

    report = generate_report(data_dir=args.data_dir, out_path=args.out)
    print_summary(report)

    if args.html:
        from render_report import render
        html_path = os.path.splitext(report["_out_path"])[0] + ".html"
        with open(html_path, "w") as f:
            f.write(render(report))
        print(f"(Readable HTML report at {html_path})")


if __name__ == "__main__":
    main()
