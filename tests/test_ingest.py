"""
Proves the ingest layer accepts datasets that aren't ours.

WHY THIS FILE EXISTS:
Every accuracy number in this project is measured on data our own generator
produced. That is the right way to measure attribution accuracy - you cannot
grade a root cause you don't know - but it leaves an obvious question open:
does this thing only work on data shaped exactly the way we happened to write
it? A reconciliation tool that needs its input pre-normalised into one exact
dialect isn't a reconciliation tool, it's a demo.

So these cases feed the loader the dialects real exports actually use -
"1,234.56", "Rs 1,234.56", "(45.00)" for accounting negatives, ISO-T
timestamps, DD/MM/YYYY dates, comma- and pipe-separated ID lists, extra
columns nobody asked for, a missing refunds file - and assert the arithmetic
comes out identical to the canonical form. Same numbers in, same numbers out,
regardless of how the exporter chose to spell them.

They also assert the failure modes: a dataset we genuinely cannot reconcile
must raise DatasetError naming the file and column, not surface a KeyError
three frames deep or - far worse - silently reconcile a subset and report a
confident wrong total.

Run:  python3 tests/test_ingest.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from matcher import (  # noqa: E402
    DatasetError,
    load_data,
    match,
    split_ids,
    to_amount,
    to_datetime,
)

PASSED = []
FAILED = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        FAILED.append((name, f"assertion failed: {e}"))
    except Exception as e:
        FAILED.append((name, f"{type(e).__name__}: {e}"))
    else:
        PASSED.append(name)


def write_dataset(directory, transactions, settlements, refunds=None):
    """Write CSVs verbatim - each row is a raw line, so a test can express an
    exact dialect (quoting, spacing, symbols) rather than whatever csv.writer
    would normalise it into."""
    os.makedirs(directory, exist_ok=True)
    for filename, lines in (
        ("razorpay_transactions.csv", transactions),
        ("bank_settlements.csv", settlements),
        ("refunds.csv", refunds),
    ):
        if lines is None:
            continue
        with open(os.path.join(directory, filename), "w", newline="") as f:
            f.write("\n".join(lines) + "\n")


# A clean two-transaction batch that reconciles exactly, written the way our
# own generator writes it. Every dialect test below is this same dataset,
# respelled - so any difference in the result is the loader's fault.
CANONICAL_TXNS = [
    "transaction_id,order_id,amount,razorpay_fee,gst_on_fee,net_amount,timestamp,payment_method",
    "txn_0001,order_0001,1000.00,20.00,3.60,976.40,2026-08-01 10:00:00,upi",
    "txn_0002,order_0002,2000.00,40.00,7.20,1952.80,2026-08-01 11:00:00,card",
]
CANONICAL_STLS = [
    "settlement_id,settled_amount,settlement_date,transaction_ids_included",
    "stl_0001,2929.20,2026-08-03,txn_0001;txn_0002",
]


def canonical_baseline(tmp):
    directory = os.path.join(tmp, "canonical")
    write_dataset(directory, CANONICAL_TXNS, CANONICAL_STLS)
    t, s, r = load_data(directory)
    sr, _ = match(t, s, r)
    return sr[0]


# --------------------------------------------------------------------------
# Money columns
# --------------------------------------------------------------------------

def test_amount_dialects():
    assert to_amount("1000.00", "f", "c", 1) == 1000.0
    assert to_amount("1,234.56", "f", "c", 1) == 1234.56, "thousands separator"
    assert to_amount("Rs 1,234.56", "f", "c", 1) == 1234.56, "currency prefix"
    assert to_amount("₹1234.56", "f", "c", 1) == 1234.56, "currency symbol"
    assert to_amount("  42.50  ", "f", "c", 1) == 42.5, "surrounding whitespace"
    assert to_amount("(45.00)", "f", "c", 1) == -45.0, "accounting negative"
    assert to_amount("-45.00", "f", "c", 1) == -45.0, "signed negative"
    assert to_amount("", "f", "c", 1) == 0.0, "blank means zero"
    assert to_amount("   ", "f", "c", 1) == 0.0, "whitespace-only means zero"


def test_amount_rejects_garbage():
    for bad in ("not-a-number", "N/A", "--"):
        try:
            to_amount(bad, "some.csv", "amount", 7)
        except DatasetError as e:
            assert "some.csv" in str(e) and "amount" in str(e) and "7" in str(e), \
                f"error must name file, column and row; got: {e}"
        else:
            raise AssertionError(f"{bad!r} should have raised DatasetError")


# --------------------------------------------------------------------------
# Timestamp columns
# --------------------------------------------------------------------------

def test_timestamp_dialects():
    expected = "2026-08-01"
    for value in (
        "2026-08-01 10:00:00",
        "2026-08-01T10:00:00",
        "2026-08-01T10:00:00Z",
        "2026-08-01T10:00:00+05:30",
        "2026-08-01 10:00",
        "2026-08-01",
        "01/08/2026 10:00:00",
        "01-08-2026",
    ):
        got = to_datetime(value, "f", "timestamp").strftime("%Y-%m-%d")
        assert got == expected, f"{value!r} parsed as {got}, expected {expected}"


def test_timestamp_rejects_garbage():
    try:
        to_datetime("last tuesday", "some.csv", "timestamp", "txn_0001")
    except DatasetError as e:
        assert "timestamp" in str(e) and "txn_0001" in str(e)
    else:
        raise AssertionError("unparseable date should have raised DatasetError")


# --------------------------------------------------------------------------
# ID list separators
# --------------------------------------------------------------------------

def test_id_separators():
    expected = ["txn_0001", "txn_0002"]
    for value in (
        "txn_0001;txn_0002",
        "txn_0001,txn_0002",
        "txn_0001, txn_0002",
        "txn_0001|txn_0002",
        "txn_0001 txn_0002",
        " txn_0001 ; txn_0002 ",
    ):
        assert split_ids(value) == expected, f"failed on {value!r}"
    assert split_ids("") == [], "empty list is empty, not ['']"
    assert split_ids(None) == []


# --------------------------------------------------------------------------
# End-to-end: a messy export must reconcile identically to a clean one
# --------------------------------------------------------------------------

def test_messy_export_matches_canonical(tmp):
    """The same two payments, exported by a system with different habits:
    currency-formatted amounts, ISO-T timestamps with a timezone, DD/MM/YYYY
    settlement date, comma-separated IDs, columns in a different order with
    extra ones we don't use, and no refunds file at all."""
    baseline = canonical_baseline(tmp)

    directory = os.path.join(tmp, "messy")
    write_dataset(
        directory,
        [
            "timestamp,transaction_id,merchant_ref,order_id,amount,razorpay_fee,gst_on_fee,net_amount,notes",
            '2026-08-01T10:00:00+05:30,txn_0001,MER-1,order_0001,"Rs 1,000.00",20.00,3.60,976.40,first',
            '2026-08-01T11:00:00+05:30,txn_0002,MER-2,order_0002,"Rs 2,000.00",40.00,7.20,"1,952.80",second',
        ],
        [
            "settlement_id,utr,settled_amount,settlement_date,transaction_ids_included",
            'stl_0001,UTR123,"2,929.20",03/08/2026,"txn_0001, txn_0002"',
        ],
        refunds=None,  # no refunds file at all
    )

    t, s, r = load_data(directory)
    assert r == [], "a missing refunds.csv should load as no refunds, not crash"
    sr, tr = match(t, s, r)
    result = sr[0]

    assert result["status"] == baseline["status"] == "MATCHED", \
        f"messy export gave {result['status']}, canonical gave {baseline['status']}"
    assert result["expected_amount"] == baseline["expected_amount"], \
        f"expected {baseline['expected_amount']}, got {result['expected_amount']}"
    assert result["actual_amount"] == baseline["actual_amount"]
    assert result["diff"] == baseline["diff"] == 0.0
    assert all(v["status"] == "SETTLED" for v in tr.values())


def test_real_discrepancy_survives_messy_formatting(tmp):
    """Robust parsing must not become forgiving arithmetic. A settlement that
    is genuinely short is still short after going through the messy path."""
    directory = os.path.join(tmp, "messy_short")
    write_dataset(
        directory,
        CANONICAL_TXNS,
        [
            "settlement_id,settled_amount,settlement_date,transaction_ids_included",
            'stl_0001,"Rs 1,952.80",03/08/2026,"txn_0001, txn_0002"',
        ],
    )
    t, s, r = load_data(directory)
    sr, _ = match(t, s, r)
    assert sr[0]["status"] == "AMOUNT_MISMATCH", "a short payout must still be flagged"
    assert sr[0]["diff"] == -976.40, f"expected -976.40, got {sr[0]['diff']}"
    # And the evidence for the reasoner must still be there and still correct.
    exact = sr[0]["signals"]["exact_match_to_one_transaction"]
    assert exact and exact["transaction_id"] == "txn_0001", \
        "the gap equals txn_0001's contribution exactly - that signal must survive"


def test_refunds_applied_through_messy_path(tmp):
    directory = os.path.join(tmp, "messy_refund")
    write_dataset(
        directory,
        CANONICAL_TXNS,
        [
            "settlement_id,settled_amount,settlement_date,transaction_ids_included",
            "stl_0001,2829.20,2026-08-03,txn_0001;txn_0002",
        ],
        refunds=[
            "refund_id,transaction_id,refund_amount,refund_date",
            'ref_0001,txn_0001,"Rs 100.00",2026-08-02',
        ],
    )
    t, s, r = load_data(directory)
    sr, _ = match(t, s, r)
    assert sr[0]["status"] == "MATCHED", \
        f"a recorded refund should reconcile, got {sr[0]['status']} ({sr[0].get('diff')})"


# --------------------------------------------------------------------------
# Failure modes: unusable datasets must say why
# --------------------------------------------------------------------------

def test_missing_column_names_itself(tmp):
    directory = os.path.join(tmp, "no_net")
    write_dataset(
        directory,
        [
            "transaction_id,order_id,amount,razorpay_fee,gst_on_fee,timestamp",
            "txn_0001,order_0001,1000.00,20.00,3.60,2026-08-01 10:00:00",
        ],
        CANONICAL_STLS,
    )
    try:
        load_data(directory)
    except DatasetError as e:
        assert "net_amount" in str(e), f"error must name the missing column; got: {e}"
    else:
        raise AssertionError("a missing required column should raise DatasetError")


def test_missing_file_names_itself(tmp):
    directory = os.path.join(tmp, "no_settlements")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "razorpay_transactions.csv"), "w") as f:
        f.write("\n".join(CANONICAL_TXNS) + "\n")
    try:
        load_data(directory)
    except DatasetError as e:
        assert "bank_settlements.csv" in str(e)
    else:
        raise AssertionError("a missing required file should raise DatasetError")


def test_empty_dataset_refuses_rather_than_reporting_100_pct(tmp):
    """An empty ledger reconciles perfectly against an empty bank feed, which
    would report a flawless 100% match rate over nothing at all. Refusing is
    the honest behaviour."""
    directory = os.path.join(tmp, "empty")
    write_dataset(directory, [CANONICAL_TXNS[0]], [CANONICAL_STLS[0]])
    try:
        load_data(directory)
    except DatasetError as e:
        assert "no rows" in str(e)
    else:
        raise AssertionError("an empty dataset should raise rather than report 100%")


def test_settlement_with_no_ids_is_flagged(tmp):
    """An unexplained bank credit naming no transactions must surface as an
    exception, never be counted as a clean match."""
    directory = os.path.join(tmp, "no_ids")
    write_dataset(
        directory,
        CANONICAL_TXNS,
        [
            "settlement_id,settled_amount,settlement_date,transaction_ids_included",
            "stl_0001,2929.20,2026-08-03,",
        ],
    )
    t, s, r = load_data(directory)
    sr, _ = match(t, s, r)
    assert sr[0]["status"] == "ORPHAN", \
        f"a settlement naming no transactions must be flagged, got {sr[0]['status']}"


def main():
    tmp = tempfile.mkdtemp(prefix="recon_ingest_tests_")
    try:
        check("amount dialects (commas, symbols, accounting negatives)", test_amount_dialects)
        check("amount garbage is rejected with a locating message", test_amount_rejects_garbage)
        check("timestamp dialects (ISO-T, timezone, DD/MM/YYYY)", test_timestamp_dialects)
        check("timestamp garbage is rejected with a locating message", test_timestamp_rejects_garbage)
        check("ID list separators (; , | space)", test_id_separators)
        check("messy export reconciles identically to canonical", lambda: test_messy_export_matches_canonical(tmp))
        check("a real discrepancy survives messy formatting", lambda: test_real_discrepancy_survives_messy_formatting(tmp))
        check("recorded refunds apply through the messy path", lambda: test_refunds_applied_through_messy_path(tmp))
        check("missing column names itself", lambda: test_missing_column_names_itself(tmp))
        check("missing file names itself", lambda: test_missing_file_names_itself(tmp))
        check("empty dataset refuses rather than reporting 100%", lambda: test_empty_dataset_refuses_rather_than_reporting_100_pct(tmp))
        check("settlement naming no transactions is flagged", lambda: test_settlement_with_no_ids_is_flagged(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for name in PASSED:
        print(f"  PASS  {name}")
    for name, why in FAILED:
        print(f"  FAIL  {name}\n          {why}")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
