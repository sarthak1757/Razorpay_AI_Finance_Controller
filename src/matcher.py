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
from datetime import datetime, timedelta

AMOUNT_TOLERANCE = 0.05  # rupees - allows for float rounding, nothing more

# A discrepancy this size or smaller is in "someone's fee/GST maths was off"
# territory rather than "a whole payment went missing" territory. Used only to
# describe the gap, never to decide what caused it.
SMALL_DISCREPANCY_CEILING = 50.0

# A gap between these fractions of a single transaction's net is refund-shaped.
REFUND_FRACTION_MIN = 0.10
REFUND_FRACTION_MAX = 0.70

SETTLEMENT_SLA_DAYS = 2  # Razorpay's standard T+2 payout cycle


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


def contribution_of(txn, refund_by_txn):
    """What this transaction SHOULD have contributed to its settlement:
    its net amount, less any refund we have a record of."""
    net = txn["net_amount"]
    refund = refund_by_txn.get(txn["transaction_id"])
    if refund:
        net = round(net - refund["refund_amount"], 2)
    return round(net, 2)


def discrepancy_signals(diff, expected_total, batch_txns, refund_by_txn):
    """Purely arithmetic observations about a settlement discrepancy.

    DELIBERATELY DOES NOT NAME A CAUSE. This function answers factual
    questions - "is the gap exactly equal to one transaction's contribution?",
    "is it a refund-sized fraction of one?", "did the bank pay too much or too
    little?" - and stops there. Turning those facts into a root cause is a
    judgment call, and judgment calls are the reasoning layer's job (see
    reasoner.py). Keeping the split here is what makes the AI layer's accuracy
    a real measurement rather than a formality: the model has to do the
    attribution, we only hand it good evidence.
    """
    abs_diff = abs(diff)
    signals = {
        "direction": "bank_paid_more_than_owed" if diff > 0 else "bank_paid_less_than_owed",
        "abs_diff": round(abs_diff, 2),
        "diff_pct_of_expected": round(abs_diff / expected_total * 100, 3) if expected_total else None,
        "is_small_absolute_discrepancy": abs_diff <= SMALL_DISCREPANCY_CEILING,
        "transactions_in_batch": len(batch_txns),
        "exact_match_to_one_transaction": None,
        "refund_sized_fraction_of_one_transaction": None,
        # The full batch, so a reasoner can work the numbers itself instead of
        # trusting our two convenience checks above. Those checks use fixed
        # thresholds, and fixed thresholds are exactly what a real discrepancy
        # eventually falls outside of - a 6% unrecorded refund isn't
        # "refund-sized" by our rule, but it's still obviously 6% of one
        # transaction if you look. Handing over the raw composition is what
        # lets the reasoning layer beat the rule instead of inheriting its
        # blind spots.
        "batch_composition": [
            {
                "transaction_id": t["transaction_id"],
                "net_amount": t["net_amount"],
                "contribution_after_recorded_refunds": contribution_of(t, refund_by_txn),
                "gap_as_pct_of_this_txn_net": round(abs_diff / t["net_amount"] * 100, 1) if t["net_amount"] else None,
            }
            for t in batch_txns
        ],
    }

    # Does the gap equal exactly what one of these transactions should have
    # contributed? (i.e. one payment's worth of money is unaccounted for)
    for t in batch_txns:
        contribution = contribution_of(t, refund_by_txn)
        if abs(abs_diff - contribution) <= AMOUNT_TOLERANCE:
            signals["exact_match_to_one_transaction"] = {
                "transaction_id": t["transaction_id"],
                "contribution": contribution,
            }
            break

    # Failing that - is it a refund-shaped slice of one transaction?
    if not signals["exact_match_to_one_transaction"]:
        for t in batch_txns:
            net = t["net_amount"]
            if not net:
                continue
            fraction = abs_diff / net
            if REFUND_FRACTION_MIN <= fraction <= REFUND_FRACTION_MAX:
                signals["refund_sized_fraction_of_one_transaction"] = {
                    "transaction_id": t["transaction_id"],
                    "net_amount": net,
                    "pct_of_net": round(fraction * 100, 1),
                    "has_recorded_refund": t["transaction_id"] in refund_by_txn,
                }
                break

    return signals


def match(transactions, settlements, refunds):
    txn_by_id = {t["transaction_id"]: t for t in transactions}
    refund_by_txn = {r["transaction_id"]: r for r in refunds}
    duplicate_ids = find_duplicates(transactions)

    # The last date this dataset can tell us anything about. A transaction
    # from the final two days simply hasn't had time to settle at T+2 - that
    # is a fact about the reporting window, not a problem with the payment.
    settlement_dates = [
        datetime.strptime(s["settlement_date"], "%Y-%m-%d") for s in settlements
    ]
    data_window_end = max(settlement_dates) if settlement_dates else None

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
        batch_txns = []
        for tid in included_ids:
            settled_txn_ids.add(tid)
            t = txn_by_id[tid]
            batch_txns.append(t)
            expected_total += contribution_of(t, refund_by_txn)

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
            signals = None
        else:
            status = "AMOUNT_MISMATCH"
            reason = f"Settled amount is off by Rs {diff:+.2f} vs expected"
            signals = discrepancy_signals(diff, expected_total, batch_txns, refund_by_txn)

        result = {
            "settlement_id": s["settlement_id"],
            "status": status,
            "reason": reason,
            "expected_amount": expected_total,
            "actual_amount": actual_total,
            "diff": diff,
            "timing_note": timing_note,
            "included_transactions": included_ids,
        }
        if signals:
            result["signals"] = signals
        settlement_results.append(result)

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
            # Facts only: when it happened, when T+2 would have made it due,
            # and whether that deadline has passed within the data we have.
            # Whether "overdue" means lost or just in-flight is a judgment
            # call, left to the reasoning layer.
            txn_date = datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S")
            due_by = txn_date + timedelta(days=SETTLEMENT_SLA_DAYS)
            signals = {
                "transaction_timestamp": t["timestamp"],
                "settlement_due_by": due_by.strftime("%Y-%m-%d"),
                "data_window_end": data_window_end.strftime("%Y-%m-%d") if data_window_end else None,
                "net_amount": t["net_amount"],
            }
            if data_window_end:
                days_overdue = (data_window_end - due_by).days
                signals["days_past_due_at_window_end"] = days_overdue
                signals["due_date_passed_within_data_window"] = days_overdue >= 0

            transaction_results[tid] = {
                "status": "MISSING_SETTLEMENT",
                "reason": "No settlement references this transaction",
                "signals": signals,
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
