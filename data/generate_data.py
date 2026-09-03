"""
Synthetic data generator for the Track 4 (AI Finance Controller) reconciliation agent.

WHY THIS SHAPE:
Real Razorpay <-> bank reconciliation has two sources that don't line up 1:1:
  - Razorpay's own transaction ledger (one row per payment)
  - The bank's settlement statement (one row per settlement BATCH, since Razorpay
    nets out fees and pays out multiple transactions in one bank credit, usually T+2)

So this generator produces two CSVs that are DELIBERATELY not a clean 1:1 match,
plus a ground_truth.json that records exactly what we injected and why.

WHY MULTIPLE CAUSES PER EXCEPTION TYPE:
An earlier version of this generator injected exactly one cause of amount
mismatch (a fee error) and one cause of missing settlement. That makes any
"did the AI find the right root cause?" metric meaningless - the answer would
be derivable from the exception TYPE alone, so a model that never reasons at
all scores 100%. Real reconciliation isn't like that: a settlement that came
up short could be a fee error, an unrecorded refund, a transaction that got
dropped from the batch, and a settlement that's too LARGE is a different
problem again. Telling those apart is the actual job.

So each exception type now has several distinguishable causes, and the true
cause of every single injected exception is recorded in ground_truth.json.
The signal needed to tell them apart is present in the raw numbers (a dropped
transaction makes the gap exactly equal that transaction's net; an unrecorded
refund makes it a fraction of one; a fee error is small and unrelated to any
transaction's structure), so this is a real inference task, not a guess.

IMPORTANT: ground_truth.json is for US to grade against later - both the
matcher ("did it find the real exceptions, and did it wrongly flag clean
cases?") and the AI reasoning layer ("did it name the right cause?"). Neither
matcher.py nor reasoner.py ever reads this file. That's what makes "honest
measured accuracy" possible instead of just a vibe.

Usage:
    python3 data/generate_data.py                      # canonical dataset
    python3 data/generate_data.py --seed 7 --n 300     # a different draw
    python3 data/generate_data.py --seed 7 --out-dir /tmp/ds7
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta

# The cause taxonomy is defined once, in src/causes.py, and imported by the
# generator, the reasoner and the grader alike - see that file for why.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from causes import (  # noqa: E402
    CAUSE_FEE_OR_GST,
    CAUSE_UNRECORDED_REFUND,
    CAUSE_DROPPED_TXN,
    CAUSE_OVERPAYMENT,
    CAUSE_ORPHAN,
    CAUSE_STUCK,
    CAUSE_NOT_YET_DUE,
    CAUSE_UNKNOWN,
    MISMATCH_CAUSES,
    ALL_CAUSES,
)

START_DATE = datetime(2026, 8, 1)
SETTLEMENT_SLA_DAYS = 2  # Razorpay's standard T+2 payout cycle
PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet"]

RAZORPAY_FEE_PCT = 0.02      # 2% platform fee
GST_ON_FEE_PCT = 0.18        # 18% GST charged on top of the fee

def make_transaction(rng, i, txn_date):
    amount = round(rng.uniform(150, 25000), 2)
    fee = round(amount * RAZORPAY_FEE_PCT, 2)
    gst_on_fee = round(fee * GST_ON_FEE_PCT, 2)
    net_amount = round(amount - fee - gst_on_fee, 2)
    return {
        "transaction_id": f"txn_{i:04d}",
        "order_id": f"order_{i:04d}",
        "amount": amount,
        "currency": "INR",
        "payment_method": rng.choice(PAYMENT_METHODS),
        "status": "captured",
        "timestamp": txn_date.strftime("%Y-%m-%d %H:%M:%S"),
        "razorpay_fee": fee,
        "gst_on_fee": gst_on_fee,
        "net_amount": net_amount,
    }


def generate(seed=42, n_transactions=60, out_dir="data", window_days=7, verbose=True,
             vary=False, novel_rate=0.0):
    """Generate one complete dataset + its ground truth.

    Uses a local RNG instance rather than seeding the global `random` module,
    so generating several datasets in one process (the validation harness
    does exactly that) can't have one run's draws bleed into the next.

    Exception counts scale with n_transactions so that a 600-transaction
    dataset has proportionally the same exception density as a 60-transaction
    one - otherwise cross-scale comparisons would be apples to oranges.

    WHAT `vary` IS FOR:
    Without it, changing the seed changes every amount, date and victim, but
    NOT the recipe: every dataset still has ~12% mismatched batches, a
    duplicate every 20 transactions, and the same 1-3 batch-size mix. Running
    a hundred of those measures robustness to different *numbers*, which is
    worth something, but it does not measure robustness to a different *kind*
    of merchant - and quoting "120 datasets" while every one shares a recipe
    would overstate what was actually tested.

    With `vary=True` the regime itself is drawn from the seed: how often
    settlements go wrong (4%-30%), how big the payout batches are (from all
    singletons to batches of six), how long the reporting window is, how
    common refunds, duplicates and stuck payouts are. A high-volume merchant
    with clean books and a small one with messy books are different problems,
    and the matcher has to hold on both.
    """
    rng = random.Random(seed)

    # The regime gets its OWN generator, deliberately. Sharing one would mean
    # that changing the shape of a dataset also reshuffled every amount inside
    # it, so a vary=True run could never be compared against the vary=False
    # run at the same seed. Two independent knobs, two independent streams.
    if vary:
        vrng = random.Random(seed * 7919 + 13)
        window_days = vrng.choice([5, 7, 10, 14, 21])
        mismatch_rate = vrng.uniform(0.04, 0.30)
        stuck_divisor = vrng.choice([15, 20, 30, 50, 80])
        refund_divisor = vrng.choice([8, 12, 15, 25, 40])
        duplicate_divisor = vrng.choice([10, 20, 35, 60])
        batch_size_weights = vrng.choice([
            [1],                # never batches - one payout per payment
            [1, 1, 2],          # mostly singletons
            [1, 1, 2, 2, 3],    # the default mix
            [1, 2, 3, 4],       # moderately batched
            [2, 3, 4, 5, 6],    # heavily batched, no singletons at all
        ])
    else:
        mismatch_rate = 0.12
        stuck_divisor, refund_divisor, duplicate_divisor = 30, 15, 20
        batch_size_weights = [1, 1, 2, 2, 3]

    regime = {
        "vary": vary,
        "window_days": window_days,
        "mismatch_rate": round(mismatch_rate, 4),
        "stuck_divisor": stuck_divisor,
        "refund_divisor": refund_divisor,
        "duplicate_divisor": duplicate_divisor,
        "batch_size_weights": batch_size_weights,
        "novel_rate": round(novel_rate, 4),
    }

    # window_end is the "report pull date" - the last date this dataset can
    # tell us anything about. A payment from the final two days simply has no
    # settlement row yet, because T+2 hasn't elapsed by the time the report
    # was pulled. That is a property of the reporting window, not a fault.
    window_end = START_DATE + timedelta(days=window_days)
    settleable_end = window_end - timedelta(days=SETTLEMENT_SLA_DAYS)

    n_not_yet_due = max(1, n_transactions // 30)

    # Transactions are placed in two groups so that the "not yet due" label is
    # always TRUE BY CONSTRUCTION rather than asserted: everything in the
    # settleable window has T+2 <= window_end, everything in the tail does not.
    # (An earlier version picked "recent" transactions by eye and produced
    # labels the arithmetic contradicted - grading a model against those would
    # have punished it for being correct.)
    transactions = []
    for i in range(1, n_transactions + 1):
        if i <= n_transactions - n_not_yet_due:
            span_hours = max(1, int((settleable_end - START_DATE).total_seconds() // 3600))
            txn_date = START_DATE + timedelta(hours=rng.randint(0, span_hours - 1))
        else:
            tail_hours = int((window_end - settleable_end).total_seconds() // 3600)
            txn_date = settleable_end + timedelta(hours=rng.randint(1, tail_hours - 1))
        transactions.append(make_transaction(rng, i, txn_date))
    transactions.sort(key=lambda t: t["timestamp"])

    def txn_dt(t):
        return datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S")

    ground_truth = {
        "meta": {
            "seed": seed,
            "n_transactions": n_transactions,
            "window_days": window_days,
            "data_window_end": window_end.strftime("%Y-%m-%d"),
            "regime": regime,
        },
        "clean_settlements": [],      # settlement_ids that should reconcile exactly
        "batched_into": {},           # txn_id -> settlement_id
        "legit_partial_refund": [],   # txn_ids w/ a PROPERLY recorded refund - must NOT be flagged
        "duplicate_entry": [],        # duplicate txn_ids that shouldn't exist
        # The graded part: every injected exception and its TRUE cause.
        "true_causes": {},            # settlement_id | txn_id -> cause label
        # For novel cases the graded answer is "unknown_anomaly"; THIS records
        # what was actually done, for diagnostics only. Never graded against -
        # the system was never told these mechanisms exist.
        "novel_mechanisms": {},
    }

    # --- pick exception victims -------------------------------------------
    # Scaled proportionally; at the default n=60 this reproduces the original
    # mix (4 missing, 4 partial refunds, 3 duplicates, ~4 mismatches, 1 orphan).
    n_stuck = max(1, n_transactions // stuck_divisor)
    n_partial_refund = max(1, n_transactions // refund_divisor)
    n_duplicate = max(1, n_transactions // duplicate_divisor)

    # Everything past the settleable cutoff is not-yet-due by construction:
    # its T+2 falls after the report pull date, so no settlement exists yet.
    not_yet_due_ids = {
        t["transaction_id"] for t in transactions if txn_dt(t) > settleable_end
    }
    older = [t for t in transactions if txn_dt(t) <= settleable_end]
    rng.shuffle(older)

    # "Stuck" ones are deliberately dropped from settlement despite being old
    # enough that T+2 has comfortably passed - a genuinely lost payout.
    stuck_ids = {t["transaction_id"] for t in older[:n_stuck]}

    remaining_pool = [
        t for t in transactions
        if t["transaction_id"] not in not_yet_due_ids
        and t["transaction_id"] not in stuck_ids
    ]
    rng.shuffle(remaining_pool)
    partial_refund_ids = {t["transaction_id"] for t in remaining_pool[:n_partial_refund]}
    duplicate_source_ids = {
        t["transaction_id"]
        for t in remaining_pool[n_partial_refund:n_partial_refund + n_duplicate]
    }

    for tid in sorted(stuck_ids):
        ground_truth["true_causes"][tid] = CAUSE_STUCK
    for tid in sorted(not_yet_due_ids):
        ground_truth["true_causes"][tid] = CAUSE_NOT_YET_DUE
    ground_truth["legit_partial_refund"] = sorted(partial_refund_ids)

    # --- batch the rest into settlements -----------------------------------
    to_settle = [
        t for t in transactions
        if t["transaction_id"] not in stuck_ids
        and t["transaction_id"] not in not_yet_due_ids
    ]
    to_settle.sort(key=lambda t: t["timestamp"])

    settlements = []
    refunds = []
    settlement_counter = 1
    batches = []  # (settlement_dict, [txn dicts], expected_total)

    i = 0
    while i < len(to_settle):
        batch_size = rng.choice(batch_size_weights)
        batch = to_settle[i:i + batch_size]
        i += batch_size
        if not batch:
            continue

        settlement_id = f"stl_{settlement_counter:04d}"
        settlement_counter += 1

        latest = max(txn_dt(t) for t in batch)
        settlement_date = latest + timedelta(days=2)

        expected_total = 0.0
        for t in batch:
            net = t["net_amount"]
            if t["transaction_id"] in partial_refund_ids:
                # A REAL refund, properly recorded in refunds.csv. The matcher
                # should net this out and NOT flag the settlement. These exist
                # specifically as false-positive bait.
                refund_amount = round(net * rng.uniform(0.2, 0.5), 2)
                net = round(net - refund_amount, 2)
                refunds.append({
                    "refund_id": f"rfnd_{t['transaction_id']}",
                    "transaction_id": t["transaction_id"],
                    "refund_amount": refund_amount,
                    "refund_date": settlement_date.strftime("%Y-%m-%d"),
                })
            expected_total += net
            if len(batch) > 1:
                ground_truth["batched_into"][t["transaction_id"]] = settlement_id

        expected_total = round(expected_total, 2)
        settlement = {
            "settlement_id": settlement_id,
            "utr_number": f"UTR{rng.randint(100000000000, 999999999999)}",
            "settlement_date": settlement_date.strftime("%Y-%m-%d"),
            "settled_amount": expected_total,
            "transaction_ids_included": ";".join(t["transaction_id"] for t in batch),
        }
        settlements.append(settlement)
        batches.append((settlement, batch, expected_total))

    # --- inject amount mismatches, round-robin across the four causes ------
    # Round-robin (rather than random choice) guarantees every cause is
    # represented even on a small dataset, so the AI-accuracy metric is never
    # computed over a set that happens to contain only one kind of problem.
    n_mismatch = max(len(MISMATCH_CAUSES), round(len(batches) * mismatch_rate))
    eligible = [b for b in batches if b[1]]
    rng.shuffle(eligible)

    for idx, (settlement, batch, expected_total) in enumerate(eligible[:n_mismatch]):
        cause = MISMATCH_CAUSES[idx % len(MISMATCH_CAUSES)]
        sid = settlement["settlement_id"]

        if cause == CAUSE_FEE_OR_GST:
            # A fee MIS-RATE, not a flat error: the fee was computed at the
            # wrong percentage, so the size of the mistake scales with the
            # value of the batch. This matters for honesty of the benchmark -
            # a flat Rs 5-40 error is always tiny and therefore always
            # trivially separable from a refund-sized gap, which would make
            # root-cause attribution a solved problem by construction. Real
            # fee errors scale, and on a large batch they land in the same
            # numeric range as a refund on a small transaction. That overlap
            # is where attribution actually requires weighing evidence.
            mis_rate = rng.uniform(0.002, 0.01)
            error = round(expected_total * mis_rate, 2)
            settlement["settled_amount"] = round(expected_total - error, 2)

        elif cause == CAUSE_UNRECORDED_REFUND:
            # Short by a refund-sized fraction of ONE transaction's net, with
            # no corresponding row in refunds.csv. Wide fraction range for the
            # same reason as above: a 6% refund on a big transaction and a fee
            # mis-rate on the batch are genuinely hard to tell apart.
            victim = rng.choice(batch)
            hidden = round(victim["net_amount"] * rng.uniform(0.05, 0.6), 2)
            settlement["settled_amount"] = round(expected_total - hidden, 2)

        elif cause == CAUSE_DROPPED_TXN:
            # Settlement claims to include a transaction it never actually paid:
            # short by EXACTLY that transaction's contribution.
            if len(batch) < 2:
                # Needs >=2 transactions to be distinguishable from "nothing
                # settled at all"; fall back to a fee error on singletons.
                #
                # The error must be a MIS-RATE, not a flat 5-40 rupees. A flat
                # error on a small payment can exceed the entire fee that was
                # charged on it - and a gap bigger than the whole fee cannot
                # be a fee miscalculation, whatever we label it. That made the
                # ground truth contradict its own arithmetic, and any reasoner
                # correctly refusing to call it a fee error got marked wrong
                # for being right. Same failure mode as the "not_yet_due"
                # labels that an earlier version asserted instead of
                # constructing - see the note above the transaction split.
                mis_rate = rng.uniform(0.002, 0.01)
                error = round(expected_total * mis_rate, 2)
                settlement["settled_amount"] = round(expected_total - error, 2)
                cause = CAUSE_FEE_OR_GST
            else:
                victim = rng.choice(batch)
                contribution = victim["net_amount"]
                refund_row = next(
                    (r for r in refunds if r["transaction_id"] == victim["transaction_id"]), None
                )
                if refund_row:
                    contribution = round(contribution - refund_row["refund_amount"], 2)
                settlement["settled_amount"] = round(expected_total - contribution, 2)

        elif cause == CAUSE_OVERPAYMENT:
            # Bank credited MORE than owed - one transaction paid out twice.
            victim = rng.choice(batch)
            contribution = victim["net_amount"]
            refund_row = next(
                (r for r in refunds if r["transaction_id"] == victim["transaction_id"]), None
            )
            if refund_row:
                contribution = round(contribution - refund_row["refund_amount"], 2)
            settlement["settled_amount"] = round(expected_total + contribution, 2)

        ground_truth["true_causes"][sid] = cause

    # --- inject NOVEL mechanisms, deliberately outside the taxonomy --------
    # Everything above is a cause the system has been told about. These are
    # not: they are mechanisms a real ledger contains and this taxonomy never
    # enumerated. The correct answer for every one of them is therefore
    # CAUSE_UNKNOWN - "the numbers are wrong in a way I don't recognise" -
    # NOT the known label they least resemble.
    #
    # This is the only way to measure the thing a closed taxonomy is bad at.
    # Grading a classifier solely on causes you told it about cannot reveal
    # that it files every surprise under "fee error" with high confidence,
    # because you never showed it a surprise.
    #
    # Each mechanism is chosen to be genuinely distinguishable from the four
    # known ones IF you reason about the numbers, and genuinely NOT matched by
    # any of the matcher's fixed-threshold signals. The real mechanism is
    # recorded separately for diagnostics; it is never the graded answer.
    if novel_rate > 0:
        used = {s["settlement_id"] for s, _, _ in eligible[:n_mismatch]}
        novel_pool = [b for b in eligible if b[0]["settlement_id"] not in used]
        n_novel = round(len(batches) * novel_rate)

        for settlement, batch, expected_total in novel_pool[:n_novel]:
            sid = settlement["settlement_id"]
            refunded = [
                (t, r) for t in batch
                for r in refunds if r["transaction_id"] == t["transaction_id"]
            ]
            choices = ["chargeback_with_penalty", "partial_round_payout"]
            if refunded:
                choices.append("double_deducted_refund")
            mechanism = rng.choice(choices)

            if mechanism == "chargeback_with_penalty":
                # A disputed payment is clawed back AND the bank charges a flat
                # penalty on top. Looks superficially like a dropped
                # transaction, but the gap is that transaction's contribution
                # PLUS a few hundred rupees - so it matches nothing exactly,
                # and it is far too large to be a fee error.
                victim = rng.choice(batch)
                penalty = round(rng.uniform(250, 1500), 2)
                settlement["settled_amount"] = round(
                    expected_total - victim["net_amount"] - penalty, 2)

            elif mechanism == "partial_round_payout":
                # The bank pays a round figure and holds the remainder for the
                # next cycle. The tell is that settled_amount is exactly
                # divisible by 10,000 - a pattern no known cause produces.
                floored = (int(expected_total) // 10000) * 10000
                if floored <= 0 or expected_total - floored < 100:
                    # Too small to round down meaningfully - fall back to the
                    # chargeback so the dataset still gets a novel case here.
                    victim = rng.choice(batch)
                    penalty = round(rng.uniform(250, 1500), 2)
                    settlement["settled_amount"] = round(
                        expected_total - victim["net_amount"] - penalty, 2)
                    mechanism = "chargeback_with_penalty"
                else:
                    settlement["settled_amount"] = float(floored)

            else:  # double_deducted_refund
                # A refund that IS correctly recorded gets subtracted a second
                # time. The gap is refund-shaped, which is the trap - but the
                # refund is present in refunds.csv, so "unrecorded refund" is
                # precisely the wrong answer.
                _, refund_row = rng.choice(refunded)
                settlement["settled_amount"] = round(
                    expected_total - refund_row["refund_amount"], 2)

            ground_truth["true_causes"][sid] = CAUSE_UNKNOWN
            ground_truth["novel_mechanisms"][sid] = mechanism

    ground_truth["clean_settlements"] = sorted(
        s["settlement_id"] for s in settlements
        if s["settlement_id"] not in ground_truth["true_causes"]
    )

    # --- duplicate transaction rows (data quality issue on Razorpay's side) --
    for dup_source in sorted(duplicate_source_ids):
        original = next(t for t in transactions if t["transaction_id"] == dup_source)
        dup = dict(original)
        dup["transaction_id"] = dup_source + "_dup"
        transactions.append(dup)
        ground_truth["duplicate_entry"].append(dup["transaction_id"])

    # --- one orphan settlement (references a txn that never existed) --------
    orphan_id = f"stl_{settlement_counter:04d}"
    settlement_counter += 1
    settlements.append({
        "settlement_id": orphan_id,
        "utr_number": f"UTR{rng.randint(100000000000, 999999999999)}",
        "settlement_date": window_end.strftime("%Y-%m-%d"),
        "settled_amount": round(rng.uniform(500, 5000), 2),
        "transaction_ids_included": "txn_9999",
    })
    ground_truth["true_causes"][orphan_id] = CAUSE_ORPHAN

    transactions.sort(key=lambda t: t["transaction_id"])

    # --- write outputs ------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "razorpay_transactions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(transactions[0].keys()))
        w.writeheader()
        w.writerows(transactions)

    with open(os.path.join(out_dir, "bank_settlements.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(settlements[0].keys()))
        w.writeheader()
        w.writerows(settlements)

    with open(os.path.join(out_dir, "refunds.csv"), "w", newline="") as f:
        if refunds:
            w = csv.DictWriter(f, fieldnames=list(refunds[0].keys()))
            w.writeheader()
            w.writerows(refunds)
        else:
            f.write("refund_id,transaction_id,refund_amount,refund_date\n")

    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    cause_counts = {}
    for cause in ground_truth["true_causes"].values():
        cause_counts[cause] = cause_counts.get(cause, 0) + 1

    if verbose:
        print(f"[seed={seed}] {len(transactions)} transactions "
              f"({len(duplicate_source_ids)} duplicated), {len(settlements)} settlements, "
              f"{len(refunds)} recorded refunds -> {out_dir}")
        print(f"  injected exceptions ({len(ground_truth['true_causes'])} total):")
        for cause in ALL_CAUSES:
            if cause in cause_counts:
                print(f"    {cause:28s} {cause_counts[cause]}")

    return {
        "transactions": transactions,
        "settlements": settlements,
        "refunds": refunds,
        "ground_truth": ground_truth,
        "cause_counts": cause_counts,
    }


def main():
    p = argparse.ArgumentParser(description="Generate a synthetic reconciliation dataset.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=60, dest="n_transactions")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--novel-rate", type=float, default=0.0,
                   help="Fraction of settlements given a mechanism that is NOT in the "
                        "taxonomy (chargeback+penalty, round-number partial payout, "
                        "double-deducted refund). Their true cause is recorded as "
                        "'unknown_anomaly' - the correct answer is to admit the numbers "
                        "match nothing known, not to guess the nearest label.")
    p.add_argument("--vary", action="store_true",
                   help="Draw the dataset's REGIME from the seed too - exception rate, "
                        "batch sizes, refund/duplicate frequency, window length - not "
                        "just its amounts. Use this to test across different kinds of "
                        "merchant rather than different draws of the same kind.")
    args = p.parse_args()
    generate(
        seed=args.seed,
        n_transactions=args.n_transactions,
        out_dir=args.out_dir,
        window_days=args.window_days,
        vary=args.vary,
        novel_rate=args.novel_rate,
    )


if __name__ == "__main__":
    main()
