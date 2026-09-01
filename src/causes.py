"""
The root-cause taxonomy - one definition, imported by everything.

WHY THIS FILE EXISTS:
Three separate places need to agree on what the possible causes are: the data
generator (which injects them), the reasoning layer (which has to pick one),
and the grader (which compares the two). If those lists ever drift apart, the
accuracy metric silently becomes nonsense - the model gets marked wrong for
saying "fee_error" when ground truth spelled it "fee_or_gst_error". So the
list lives here and nowhere else.

IS TELLING THE MODEL THE LIST CHEATING?
No. Knowing the set of possible diagnoses is not knowing the diagnosis - a
doctor knowing the list of diseases isn't cheating at diagnosis. The model
still has to work out WHICH one applies from the numbers. Constraining the
output to a fixed set is also what makes the metric computable at all: free
text ("transaction_fee_difference" vs "fee_calculation_error" vs "processing
fee issue") can't be scored without a human judging synonyms, and a metric a
human has to referee isn't a metric.
"""

# --- Causes behind a settlement whose amount doesn't reconcile -----------
CAUSE_FEE_OR_GST = "fee_or_gst_error"
CAUSE_UNRECORDED_REFUND = "unrecorded_refund"
CAUSE_DROPPED_TXN = "missing_txn_in_batch"
CAUSE_OVERPAYMENT = "overpayment_duplicate_payout"

# --- Cause behind a settlement referencing a non-existent transaction ----
CAUSE_ORPHAN = "orphan_reference"

# --- Causes behind a transaction with no settlement ----------------------
CAUSE_STUCK = "unsettled_stuck"
CAUSE_NOT_YET_DUE = "not_yet_due"

MISMATCH_CAUSES = [
    CAUSE_FEE_OR_GST,
    CAUSE_UNRECORDED_REFUND,
    CAUSE_DROPPED_TXN,
    CAUSE_OVERPAYMENT,
]

MISSING_CAUSES = [CAUSE_STUCK, CAUSE_NOT_YET_DUE]

ALL_CAUSES = MISMATCH_CAUSES + [CAUSE_ORPHAN] + MISSING_CAUSES

# Which causes are plausible for which exception kind. The model is only ever
# asked to choose within the relevant set - asking it to consider "not_yet_due"
# for an amount mismatch would be noise, not rigour.
CAUSES_BY_KIND = {
    "AMOUNT_MISMATCH": MISMATCH_CAUSES,
    "ORPHAN": [CAUSE_ORPHAN],
    "MISSING_SETTLEMENT": MISSING_CAUSES,
}

# Human-readable descriptions, used in the reasoner's prompt so the model knows
# what each label MEANS rather than having to infer it from the string.
CAUSE_DESCRIPTIONS = {
    CAUSE_FEE_OR_GST:
        "The platform fee or GST was miscalculated. Typically a small gap, "
        "not tied to any single transaction's amount.",
    CAUSE_UNRECORDED_REFUND:
        "A refund was issued to the customer but never recorded in the refunds "
        "data, so the bank settled less than the ledger expected. The gap is a "
        "fraction of one transaction's net amount.",
    CAUSE_DROPPED_TXN:
        "The settlement claims to cover a transaction it did not actually pay "
        "out. The gap equals exactly one included transaction's full contribution.",
    CAUSE_OVERPAYMENT:
        "The bank paid MORE than was owed - typically one transaction paid out "
        "twice. The settled amount exceeds the expected amount.",
    CAUSE_ORPHAN:
        "The settlement references a transaction ID that does not exist in the "
        "ledger at all - a data sync gap or a bad reference.",
    CAUSE_STUCK:
        "The payment is genuinely overdue: its T+2 settlement date passed within "
        "the reporting window and no settlement ever arrived.",
    CAUSE_NOT_YET_DUE:
        "Nothing is wrong. The payment is too recent - its T+2 settlement date "
        "falls after the end of this reporting window, so no settlement exists yet.",
}

# Causes where the honest answer is "no action needed" - surfacing these as
# problems would be crying wolf.
BENIGN_CAUSES = {CAUSE_NOT_YET_DUE}
