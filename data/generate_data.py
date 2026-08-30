"""
Synthetic data generator for the Track 4 (AI Finance Controller) reconciliation agent.

WHY THIS SHAPE:
Real Razorpay <-> bank reconciliation has two sources that don't line up 1:1:
  - Razorpay's own transaction ledger (one row per payment)
  - The bank's settlement statement (one row per settlement BATCH, since Razorpay
    nets out fees and pays out multiple transactions in one bank credit, usually T+2)

So this generator produces two CSVs that are DELIBERATELY not a clean 1:1 match,
plus a ground_truth.json that records exactly what we injected and why.

IMPORTANT: ground_truth.json is for US to grade the matcher against later ("did it
find the real exceptions, and did it wrongly flag clean cases?"). The matching
agent we build next will NEVER see this file - that would be cheating the metric.
This is what makes "honest measured accuracy" possible instead of just a vibe.
"""

import csv
import json
import random
from datetime import datetime, timedelta

random.seed(42)  # reproducible run - same data every time we regenerate

N_TRANSACTIONS = 60
START_DATE = datetime(2026, 8, 1)
PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet"]

RAZORPAY_FEE_PCT = 0.02      # 2% platform fee
GST_ON_FEE_PCT = 0.18        # 18% GST charged on top of the fee


def make_transaction(i, txn_date):
    amount = round(random.uniform(150, 25000), 2)
    fee = round(amount * RAZORPAY_FEE_PCT, 2)
    gst_on_fee = round(fee * GST_ON_FEE_PCT, 2)
    net_amount = round(amount - fee - gst_on_fee, 2)
    return {
        "transaction_id": f"txn_{i:04d}",
        "order_id": f"order_{i:04d}",
        "amount": amount,
        "currency": "INR",
        "payment_method": random.choice(PAYMENT_METHODS),
        "status": "captured",
        "timestamp": txn_date.strftime("%Y-%m-%d %H:%M:%S"),
        "razorpay_fee": fee,
        "gst_on_fee": gst_on_fee,
        "net_amount": net_amount,
    }


def main():
    transactions = []
    for i in range(1, N_TRANSACTIONS + 1):
        txn_date = START_DATE + timedelta(
            days=random.randint(0, 6), hours=random.randint(0, 23)
        )
        transactions.append(make_transaction(i, txn_date))

    transactions.sort(key=lambda t: t["timestamp"])

    ground_truth = {
        "clean_matches": [],
        "timing_offset_only": [],
        "batched_into": {},       # txn_id -> settlement_id (many txns -> one settlement)
        "partial_refund": [],     # txn_id
        "fee_mismatch": [],       # settlement_id
        "duplicate_entry": [],    # the duplicate txn_id that shouldn't exist
        "missing_settlement": [], # txn_id that never got settled
        "orphan_settlement": [],  # settlement_id referencing a txn that doesn't exist
    }

    settlements = []
    settlement_counter = 1
    refunds = []

    # Decide which transactions get which treatment.
    ids = [t["transaction_id"] for t in transactions]
    random.shuffle(ids)

    n_missing = 4
    n_partial_refund = 4
    n_duplicate = 3
    missing_ids = set(ids[:n_missing])
    partial_refund_ids = set(ids[n_missing:n_missing + n_partial_refund])
    duplicate_source_ids = set(
        ids[n_missing + n_partial_refund: n_missing + n_partial_refund + n_duplicate]
    )

    ground_truth["missing_settlement"] = sorted(missing_ids)
    ground_truth["partial_refund"] = sorted(partial_refund_ids)

    remaining = [
        t for t in transactions
        if t["transaction_id"] not in missing_ids
    ]

    # Batch remaining transactions into settlements of 1-3 transactions each,
    # settling T+2 from the LATEST transaction in the batch.
    i = 0
    while i < len(remaining):
        batch_size = random.choice([1, 1, 2, 2, 3])
        batch = remaining[i:i + batch_size]
        i += batch_size
        if not batch:
            continue

        settlement_id = f"stl_{settlement_counter:04d}"
        settlement_counter += 1

        latest_txn_date = max(
            datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S") for t in batch
        )
        settlement_date = latest_txn_date + timedelta(days=2)

        settled_amount = 0.0
        for t in batch:
            net = t["net_amount"]
            if t["transaction_id"] in partial_refund_ids:
                refund_amount = round(net * random.uniform(0.2, 0.5), 2)
                net = round(net - refund_amount, 2)
                refunds.append({
                    "refund_id": f"rfnd_{t['transaction_id']}",
                    "transaction_id": t["transaction_id"],
                    "refund_amount": refund_amount,
                    "refund_date": settlement_date.strftime("%Y-%m-%d"),
                })
            settled_amount += net
            if len(batch) > 1:
                ground_truth["batched_into"][t["transaction_id"]] = settlement_id
            else:
                ground_truth["clean_matches" if t["transaction_id"] not in partial_refund_ids else "timing_offset_only"].append(t["transaction_id"])

        settled_amount = round(settled_amount, 2)

        # Occasionally inject a fee-miscalculation error into the settlement total
        if random.random() < 0.08:
            error = round(random.uniform(5, 40), 2)
            settled_amount = round(settled_amount - error, 2)
            ground_truth["fee_mismatch"].append(settlement_id)

        settlements.append({
            "settlement_id": settlement_id,
            "utr_number": f"UTR{random.randint(100000000000, 999999999999)}",
            "settlement_date": settlement_date.strftime("%Y-%m-%d"),
            "settled_amount": settled_amount,
            "transaction_ids_included": ";".join(t["transaction_id"] for t in batch),
        })

    # Inject duplicate transaction rows (data quality issue on Razorpay's own side)
    for dup_source in duplicate_source_ids:
        original = next(t for t in transactions if t["transaction_id"] == dup_source)
        dup = dict(original)
        dup["transaction_id"] = dup_source + "_dup"
        transactions.append(dup)
        ground_truth["duplicate_entry"].append(dup["transaction_id"])

    # Inject one orphan settlement (references a transaction_id that never existed)
    orphan_id = f"stl_{settlement_counter:04d}"
    settlement_counter += 1
    settlements.append({
        "settlement_id": orphan_id,
        "utr_number": f"UTR{random.randint(100000000000, 999999999999)}",
        "settlement_date": (START_DATE + timedelta(days=8)).strftime("%Y-%m-%d"),
        "settled_amount": round(random.uniform(500, 5000), 2),
        "transaction_ids_included": "txn_9999",
    })
    ground_truth["orphan_settlement"].append(orphan_id)

    transactions.sort(key=lambda t: t["transaction_id"])

    # --- write outputs ---
    with open("data/razorpay_transactions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(transactions[0].keys()))
        w.writeheader()
        w.writerows(transactions)

    with open("data/bank_settlements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(settlements[0].keys()))
        w.writeheader()
        w.writerows(settlements)

    with open("data/refunds.csv", "w", newline="") as f:
        if refunds:
            w = csv.DictWriter(f, fieldnames=list(refunds[0].keys()))
            w.writeheader()
            w.writerows(refunds)
        else:
            f.write("refund_id,transaction_id,refund_amount,refund_date\n")

    with open("data/ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {len(transactions)} transactions ({len(duplicate_source_ids)} duplicated)")
    print(f"Generated {len(settlements)} settlements (1 orphan)")
    print(f"Generated {len(refunds)} refunds")
    print(f"Seeded exceptions: {n_missing} missing, {len(partial_refund_ids)} partial refunds, "
          f"{len(ground_truth['fee_mismatch'])} fee mismatches, {len(duplicate_source_ids)} duplicates, 1 orphan")


if __name__ == "__main__":
    main()
