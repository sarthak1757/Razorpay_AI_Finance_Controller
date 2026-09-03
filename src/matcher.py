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
import os
import re
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


class DatasetError(Exception):
    """A dataset we cannot read. Raised with a message that says which file,
    which column and which row - because the alternative is a KeyError or a
    ValueError surfacing three call frames deep, which tells the person
    pointing this at their own export precisely nothing."""


# Columns each file must have for the arithmetic to be possible at all.
# Anything else in the file is carried through untouched.
REQUIRED_COLUMNS = {
    "razorpay_transactions.csv": [
        "transaction_id", "order_id", "amount", "razorpay_fee",
        "gst_on_fee", "net_amount", "timestamp",
    ],
    "bank_settlements.csv": [
        "settlement_id", "settled_amount", "settlement_date",
        "transaction_ids_included",
    ],
    "refunds.csv": ["transaction_id", "refund_amount"],
}


def load_csv(path, required, optional=False):
    """Read one CSV and check its shape before anything downstream trusts it.

    Fails loudly and specifically, or not at all. A dataset that is missing a
    column is a dataset we cannot reconcile, and saying so up front is far
    better than half-processing it and reporting a confident wrong number.
    """
    if not os.path.exists(path):
        if optional:
            return []
        raise DatasetError(f"Required file not found: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []

    missing = [c for c in required if c not in header]
    if missing:
        raise DatasetError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(header) or '(no header row)'}"
        )
    return rows


def to_amount(value, path, column, row_number):
    """Parse a money column the way real exports actually write money.

    Bank and gateway CSVs are not tidy: amounts arrive as "1,234.56", as
    "Rs 1,234.56", wrapped in quotes, or as "(45.00)" for a negative in
    accounting notation. A bare float() rejects every one of those, and the
    person who exported the file did nothing wrong. Blank means zero.
    """
    if value is None:
        raise DatasetError(f"{path} row {row_number}: column '{column}' is absent")

    text = str(value).strip()
    if not text:
        return 0.0

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    # Strip currency symbols/codes and thousands separators, keep sign and point.
    cleaned = re.sub(r"[^0-9.\-+]", "", text)
    try:
        amount = float(cleaned)
    except ValueError:
        raise DatasetError(
            f"{path} row {row_number}: column '{column}' is not a number: {value!r}"
        )
    return -amount if negative else amount


def load_data(data_dir="data"):
    """Load a dataset from any directory laid out with the three CSVs.

    refunds.csv is optional: a merchant with no refunds in the period has no
    refunds file, and that is a valid dataset rather than a crash. The other
    two are mandatory - there is nothing to reconcile without both sides.
    """
    txn_path = os.path.join(data_dir, "razorpay_transactions.csv")
    stl_path = os.path.join(data_dir, "bank_settlements.csv")
    ref_path = os.path.join(data_dir, "refunds.csv")

    transactions = load_csv(txn_path, REQUIRED_COLUMNS["razorpay_transactions.csv"])
    settlements = load_csv(stl_path, REQUIRED_COLUMNS["bank_settlements.csv"])
    refunds = load_csv(ref_path, REQUIRED_COLUMNS["refunds.csv"], optional=True)

    for i, t in enumerate(transactions, start=2):  # row 1 is the header
        for col in ("amount", "razorpay_fee", "gst_on_fee", "net_amount"):
            t[col] = to_amount(t[col], txn_path, col, i)
    for i, s in enumerate(settlements, start=2):
        s["settled_amount"] = to_amount(s["settled_amount"], stl_path, "settled_amount", i)
    for i, r in enumerate(refunds, start=2):
        r["refund_amount"] = to_amount(r["refund_amount"], ref_path, "refund_amount", i)

    if not transactions:
        raise DatasetError(f"{txn_path} has no rows - nothing to reconcile.")
    if not settlements:
        raise DatasetError(f"{stl_path} has no rows - nothing to reconcile against.")

    return transactions, settlements, refunds


# Timestamp shapes we accept. A gateway export, a bank statement and a
# hand-edited spreadsheet will each write the same instant differently, and
# none of them is wrong. Tried in order; the first that parses wins.
TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
]


def to_datetime(value, path_hint, column, row_hint=None):
    """Parse a date/timestamp column, tolerating the usual export dialects.

    Note the ordering hazard this deliberately does NOT paper over: 03/04/2026
    is 3 April in a day-first export and 4 March in a month-first one, and no
    parser can tell which from the string alone. We try day-first before
    month-first because this is an Indian payments dataset, and we say so here
    rather than silently guessing - if your export is US-formatted, normalise
    it to ISO (YYYY-MM-DD) before feeding it in.
    """
    text = str(value or "").strip()
    if not text:
        raise DatasetError(
            f"{path_hint}: column '{column}' is empty"
            + (f" on row {row_hint}" if row_hint else "")
        )

    def try_formats(candidate):
        for fmt in TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
        return None

    parsed = try_formats(text)
    if parsed:
        return parsed

    # Only now consider a trailing timezone designator. Stripping it up front
    # is actively harmful: the tail of a DD-MM-YYYY date ("01-08-2026") is
    # indistinguishable from a "-20:26" UTC offset by shape alone, so an
    # eager strip turns a valid date into "01-08" and fails it. Trying the
    # plain formats first means a real date is never mangled, and a genuine
    # offset still gets its second chance here. We reconcile on calendar
    # days, so dropping the offset changes nothing about which day it is.
    stripped = re.sub(r"(Z|[+-]\d{2}:?\d{2})$", "", text).strip()
    if stripped != text:
        parsed = try_formats(stripped)
        if parsed:
            return parsed

    raise DatasetError(
        f"{path_hint}: could not read '{column}' value {value!r}"
        + (f" on row {row_hint}" if row_hint else "")
        + f". Supported formats: {', '.join(TIMESTAMP_FORMATS)}"
    )


def split_ids(value):
    """Split a settlement's transaction ID list.

    Our generator writes semicolons, but a real feed may use commas, pipes or
    whitespace, and there is no ambiguity in accepting all of them - none of
    these characters occur inside a transaction ID.
    """
    return [part for part in re.split(r"[;,|\s]+", str(value or "").strip()) if part]


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
        # The total fee + GST actually charged across this batch. This is a
        # hard ceiling on how wrong a fee/GST calculation could possibly be:
        # even if the entire fee were computed incorrectly, the gap could not
        # exceed the whole fee. A discrepancy LARGER than this is therefore
        # not a fee error, whatever else it might be.
        #
        # Still a fact, not a cause - it states what was charged and leaves
        # the inference to the reasoning layer. But it is the fact that makes
        # "none of the known causes fit this" a defensible conclusion rather
        # than a shrug, which is what the unknown_anomaly label needs.
        "total_fee_charged_on_batch": round(
            sum(t["razorpay_fee"] + t["gst_on_fee"] for t in batch_txns), 2),
        "gap_exceeds_total_fee_charged": abs_diff > round(
            sum(t["razorpay_fee"] + t["gst_on_fee"] for t in batch_txns), 2),
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
        to_datetime(s["settlement_date"], "bank_settlements.csv", "settlement_date",
                    s.get("settlement_id"))
        for s in settlements
    ]
    data_window_end = max(settlement_dates) if settlement_dates else None

    settlement_results = []
    settled_txn_ids = set()

    for s in settlements:
        included_ids = split_ids(s["transaction_ids_included"])
        if not included_ids:
            # A payout that names no transactions at all. We cannot reconstruct
            # what it should have been, so we surface it rather than silently
            # counting it as clean - an unexplained credit is still an exception.
            settlement_results.append({
                "settlement_id": s["settlement_id"],
                "status": "ORPHAN",
                "reason": "Settlement lists no transaction IDs",
                "expected_amount": None,
                "actual_amount": s["settled_amount"],
                "included_transactions": [],
            })
            continue

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

            txn_date = to_datetime(t["timestamp"], "razorpay_transactions.csv",
                                   "timestamp", tid)
            if latest_txn_date is None or txn_date > latest_txn_date:
                latest_txn_date = txn_date

        expected_total = round(expected_total, 2)
        actual_total = s["settled_amount"]
        diff = round(actual_total - expected_total, 2)

        settlement_date = to_datetime(s["settlement_date"], "bank_settlements.csv",
                                      "settlement_date", s["settlement_id"])
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
            txn_date = to_datetime(t["timestamp"], "razorpay_transactions.csv",
                                   "timestamp", tid)
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
