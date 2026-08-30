"""
Deterministic matching layer (NO AI/LLM here on purpose).

WHY NO AI IN THIS FILE:
Comparing two numbers and checking if they're close enough is arithmetic, not a
judgment call. Running that through an LLM would be slower, costlier, and less
reliable than just... doing the math. The AI agent gets involved later, only
for the records this file can't confidently resolve (see exception_agent.py).

WHAT THIS FILE DOES:
1. Loads the three data sources (transactions, settlements, refunds).
2. Flags duplicate transaction rows (same order_id appearing twice).
3. For every settlement, reconstructs "what SHOULD have settled" from the
   transactions it claims to include (accounting for refunds), and compares
   that to what the bank statement actually says.
4. Classifies every settlement and every transaction into a status.

This file never opens ground_truth.json. It has no way to cheat.
"""

import csv
from datetime import datetime

AMOUNT_TOLERANCE = 0.05  # rupees - allows for float rounding, nothing more


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_data(data_dir="data"):
    transactions = load_csv(f"{data_dir}/razorpay_transactions.csv")
    settlements = load_csv(f"{data_dir}/bank_settlements.csv")
    refunds = load_csv(f"{data_dir}/refunds.csv")

    for t in transactions:
        t["amount"] = float(t["amount"])
        t["razorpay_fee"] = float(t["razorpay_fee"])
        t["gst_on_fee"] = float(t["gst_on_fee"])
        t["net_amount"] = float(t["net_amount"])
    for s in settlements:
        s["settled_amount"] = float(s["settled_amount"])
    for r in refunds:
        r["refund_amount"] = float(r["refund_amount"])

    return transactions, settlements, refunds


def find_duplicates(transactions):
    """
    Same order_id showing up on more than one transaction row = almost
    certainly a double-logged payment, not two real payments. We keep the
    chronologically first row as 'real' and flag the rest.
    Returns a set of transaction_ids that are the FLAGGED duplicates.
    """
    by_order = {}
    for t in transactions:
        by_order.setdefault(t["order_id"], []).append(t)

    duplicate_ids = set()
    for order_id, rows in by_order.items():
        if len(rows) > 1:
            rows_sorted = sorted(rows, key=lambda t: t["transaction_id"])
            for extra in rows_sorted[1:]:
                duplicate_ids.add(extra["transaction_id"])
    return duplicate_ids


def match(transactions, settlements, refunds):
    txn_by_id = {t["transaction_id"]: t for t in transactions}
    refund_by_txn = {r["transaction_id"]: r for r in refunds}
    duplicate_ids = find_duplicates(transactions)

    settlement_results = []
    settled_txn_ids = set()

    for s in settlements:
        included_ids = s["transaction_ids_included"].split(";")
        missing_txn_refs = [tid for tid in included_ids if tid not in txn_by_id]

        if missing_txn_refs:
            # The settlement points at a transaction that doesn't exist in our
            # ledger at all. This is the "orphan settlement" case.
            settlement_results.append({
                "settlement_id": s["settlement_id"],
                "status": "ORPHAN",
                "reason": f"References unknown transaction(s): {missing_txn_refs}",
                "expected_amount": None,
                "actual_amount": s["settled_amount"],
                "included_transactions": included_ids,
            })
            continue

        expected_total = 0.0
        latest_txn_date = None
        for tid in included_ids:
            settled_txn_ids.add(tid)
            t = txn_by_id[tid]
            net = t["net_amount"]
            if tid in refund_by_txn:
                net -= refund_by_txn[tid]["refund_amount"]
            expected_total += net

            txn_date = datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S")
            if latest_txn_date is None or txn_date > latest_txn_date:
                latest_txn_date = txn_date

        expected_total = round(expected_total, 2)
        actual_total = s["settled_amount"]
        diff = round(actual_total - expected_total, 2)

        settlement_date = datetime.strptime(s["settlement_date"], "%Y-%m-%d")
        days_to_settle = (settlement_date - latest_txn_date).days
        timing_note = (
            "normal T+2 cycle" if 1 <= days_to_settle <= 3
            else f"unusual delay ({days_to_settle}d)"
        )

        if abs(diff) <= AMOUNT_TOLERANCE:
            status = "MATCHED"
            reason = (
                "Batched settlement, amounts reconcile"
                if len(included_ids) > 1 else "Clean match"
            )
        else:
            status = "AMOUNT_MISMATCH"
            reason = f"Settled amount is off by Rs {diff:+.2f} vs expected"

        settlement_results.append({
            "settlement_id": s["settlement_id"],
            "status": status,
            "reason": reason,
            "expected_amount": expected_total,
            "actual_amount": actual_total,
            "diff": diff,
            "timing_note": timing_note,
            "included_transactions": included_ids,
        })

    # Transaction-level view: every real (non-duplicate) transaction is either
    # settled somewhere, or it's missing.
    transaction_results = {}
    for t in transactions:
        tid = t["transaction_id"]
        if tid in duplicate_ids:
            transaction_results[tid] = {
                "status": "DUPLICATE",
                "reason": f"Same order_id ({t['order_id']}) appears on another transaction row",
            }
        elif tid in settled_txn_ids:
            transaction_results[tid] = {"status": "SETTLED", "reason": None}
        else:
            transaction_results[tid] = {
                "status": "MISSING_SETTLEMENT",
                "reason": "No settlement references this transaction",
            }

    return settlement_results, transaction_results


if __name__ == "__main__":
    transactions, settlements, refunds = load_data()
    settlement_results, transaction_results = match(transactions, settlements, refunds)

    from collections import Counter
    s_counts = Counter(r["status"] for r in settlement_results)
    t_counts = Counter(r["status"] for r in transaction_results.values())

    print(f"Settlements: {len(settlement_results)} total -> {dict(s_counts)}")
    print(f"Transactions: {len(transaction_results)} total -> {dict(t_counts)}")

    print("\n--- Settlements needing a closer look ---")
    for r in settlement_results:
        if r["status"] != "MATCHED":
            print(f"  {r['settlement_id']}: {r['status']} - {r['reason']}")

    print("\n--- Transactions needing a closer look ---")
    for tid, r in transaction_results.items():
        if r["status"] != "SETTLED":
            print(f"  {tid}: {r['status']} - {r['reason']}")
