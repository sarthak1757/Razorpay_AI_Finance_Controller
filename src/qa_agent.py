"""
Settlement Q&A agent - one of Track 4's named example directions.

Instead of only producing a batch report, this lets you ask a plain-English
question about the reconciliation and get an answer grounded in the actual
data - "why did I get less money for settlement X", "what happened to
txn_0004", "how many settlements are still unresolved".

DESIGN NOTE: at this batch size (96 records) we ground the model by handing
it the full structured result set directly in the prompt, rather than
building a vector-search/RAG layer. That's a deliberate choice, not a
shortcut - at this scale it's more reliable (no retrieval to get wrong) and
simpler to audit. A larger production dataset (thousands+ of records) is
where real retrieval would earn its complexity; forcing it here would just
be an extra thing to get wrong for no benefit at this size.

Usage:
    python3 src/qa_agent.py                     -> interactive loop
    python3 src/qa_agent.py "why did stl_0008 mismatch?"   -> single question
"""

import sys
import json

from matcher import load_data, match
from reasoner import get_reasoner


def build_context(transactions, settlements, refunds, settlement_results, transaction_results):
    """Compact, full-detail context - small enough at this batch size to
    just hand the model everything rather than retrieve a subset.

    Includes BOTH layers: the matcher's derived analysis (status, diff,
    reason - what's wrong) and the raw source records (amount, fee, gst,
    net_amount - why it's wrong). Without the raw records the model can see
    that stl_0008 is off by Rs 12.46 but can't trace which fee or refund
    caused it - it can only guess. With them, it can actually look up
    txn_0060's razorpay_fee/gst_on_fee/net_amount and reason about the gap."""
    return json.dumps({
        "raw_transactions": transactions,
        "raw_settlements": settlements,
        "raw_refunds": refunds,
        "settlement_analysis": settlement_results,
        "transaction_analysis": transaction_results,
    }, indent=2, default=str)


def answer(question: str, context: str, reasoner) -> str:
    prompt = f"""You are a settlement Q&A assistant for a merchant using Razorpay.
You have two layers of data below: the raw source records (individual
transactions with their amount/fee/GST/net_amount, bank settlements, and
refunds) and the deterministic matcher's analysis (which settlements/
transactions were flagged, and why). Use the raw records to trace exact
numbers - e.g. if a settlement is flagged as short by some amount, look up
the underlying transaction's razorpay_fee, gst_on_fee, and any refund to
see if they explain the gap.

RECONCILIATION DATA:
{context}

MERCHANT QUESTION: {question}

Answer in plain English, 2-4 sentences, using ONLY the data above. If the
data doesn't contain enough to answer confidently, say so plainly instead
of guessing. Reference specific IDs and amounts where relevant."""

    return reasoner.chat(prompt)


def main():
    transactions, settlements, refunds = load_data()
    settlement_results, transaction_results = match(transactions, settlements, refunds)
    context = build_context(transactions, settlements, refunds, settlement_results, transaction_results)
    reasoner = get_reasoner()

    print(f"Settlement Q&A agent ready (reasoner: {reasoner.name})")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
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
