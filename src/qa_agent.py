"""
Settlement Q&A agent - one of Track 4's named example directions.

Instead of only producing a batch report, this lets you ask a plain-English
question about the reconciliation and get an answer grounded in the actual
data - "why did I get less money for settlement X", "what happened to
txn_0004", "how many settlements are still unresolved".

DESIGN NOTE: at this batch size (96 records) we ground the model by handing
it the structured result set directly in the prompt, rather than building a
vector-search/RAG layer. That's a deliberate choice, not a shortcut - at
this scale it's more reliable (no retrieval to get wrong) and simpler to
audit. A larger production dataset (thousands+ of records) is where real
retrieval would earn its complexity; forcing it here would just be an extra
thing to get wrong for no benefit at this size.

CONTEXT SIZING: "full context" doesn't mean "every raw field for every
record". Handing the model all 63 transactions' full detail (amount, fee,
GST, net_amount...) to answer a question about ONE broken settlement blew
past Groq's free-tier per-request token limit (8000) for no reason - the
model doesn't need txn_0001's fee breakdown to explain stl_0008. So the
context has two tiers: a compact status roster for every settlement/
transaction (cheap - id, status, one-line reason) so broad questions like
"how many are still unresolved" still work, plus full raw records (amount,
fee, GST, net_amount, refunds) ONLY for whatever is actually flagged as an
exception - the handful of things a merchant would ask "why" about.

Usage:
    python3 src/qa_agent.py                     -> interactive loop
    python3 src/qa_agent.py "why did stl_0008 mismatch?"   -> single question
"""

import sys
import json

from matcher import load_data, match
from reasoner import get_reasoner

EXCEPTION_SETTLEMENT_STATUSES = {"AMOUNT_MISMATCH", "ORPHAN"}
EXCEPTION_TXN_STATUSES = {"MISSING_SETTLEMENT", "DUPLICATE"}


def build_context(transactions, settlements, refunds, settlement_results, transaction_results):
    """Two-tier context, sized to actually fit in a free-tier request.

    Tier 1 - status_roster: every settlement/transaction, but only its id,
    status, and short reason. Enough to answer "what happened to txn_0004"
    or "how many settlements are unresolved" without needing raw numbers.

    Tier 2 - exception_detail: full raw records (amount, razorpay_fee,
    gst_on_fee, net_amount) plus the matcher's full analysis (expected vs
    actual, diff, timing), but ONLY for records actually flagged as an
    exception. This is where "why is stl_0008 short by Rs 12.46" gets
    answered - the raw fee/net_amount for the transaction(s) inside that
    settlement are right there. Clean, already-matched records don't need
    this level of detail, so they're left out of tier 2 entirely.
    """
    txn_by_id = {t["transaction_id"]: t for t in transactions}
    settlement_by_id = {s["settlement_id"]: s for s in settlements}

    exception_settlements = [s for s in settlement_results if s["status"] in EXCEPTION_SETTLEMENT_STATUSES]
    exception_txn_ids = {tid for tid, r in transaction_results.items() if r["status"] in EXCEPTION_TXN_STATUSES}
    for s in exception_settlements:
        for tid in s["included_transactions"]:
            if tid in txn_by_id:
                exception_txn_ids.add(tid)

    status_roster = {
        "settlements": [
            {"settlement_id": s["settlement_id"], "status": s["status"], "reason": s["reason"]}
            for s in settlement_results
        ],
        "transactions": [
            {"transaction_id": tid, "status": r["status"], "reason": r["reason"]}
            for tid, r in transaction_results.items()
        ],
    }

    exception_detail = {
        "settlement_analysis": exception_settlements,
        "raw_transactions": [txn_by_id[tid] for tid in sorted(exception_txn_ids)],
        "raw_settlements": [settlement_by_id[s["settlement_id"]] for s in exception_settlements],
        "raw_refunds": [r for r in refunds if r["transaction_id"] in exception_txn_ids],
    }

    return json.dumps({
        "status_roster": status_roster,
        "exception_detail": exception_detail,
    }, separators=(",", ":"), default=str)


def answer(question: str, context: str, reasoner) -> str:
    prompt = f"""You are a settlement Q&A assistant for a merchant using Razorpay.
You have two layers of data below: a "status_roster" covering every
settlement and transaction at a glance (id, status, short reason), and an
"exception_detail" section with full raw numbers (amount, razorpay_fee,
gst_on_fee, net_amount, refunds) and full analysis (expected vs actual,
diff, timing) - but only for the records that were actually flagged as an
exception. If a question is about something in exception_detail, use the
raw fee/net_amount there to trace exactly where the gap comes from. If it's
about something that only appears in the roster (a clean, matched record),
you have its status but not its raw numbers - say so rather than guessing
a number you don't have.

RECONCILIATION DATA:
{context}

MERCHANT QUESTION: {question}

Answer in plain English, 2-4 sentences, using ONLY the data above. If the
data doesn't contain enough to answer confidently, say so plainly instead
of guessing. Reference specific IDs and amounts where relevant."""

    return reasoner.chat(prompt)


def main():
    # Hand-rolled rather than argparse so a bare question still works
    # unquoted: `qa_agent.py why did stl_0008 mismatch?`
    argv = sys.argv[1:]
    data_dir = "data"
    if "--data-dir" in argv:
        i = argv.index("--data-dir")
        data_dir = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    transactions, settlements, refunds = load_data(data_dir)
    settlement_results, transaction_results = match(transactions, settlements, refunds)
    context = build_context(transactions, settlements, refunds, settlement_results, transaction_results)
    reasoner = get_reasoner()

    print(f"Settlement Q&A agent ready (reasoner: {reasoner.name}, data: {data_dir})")

    if argv:
        question = " ".join(argv)
        print(f"\nQ: {question}")
        print(f"A: {answer(question, context, reasoner)}")
        return

    print("Ask a question about the reconciliation (or 'quit' to exit):\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit"):
            break
        print(f"A: {answer(question, context, reasoner)}\n")


if __name__ == "__main__":
    main()
