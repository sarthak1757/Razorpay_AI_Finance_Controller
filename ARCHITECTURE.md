# Architecture

Razorpay AI Buildathon — Track 4, AI Finance Controller.

This document explains how the system is built and, more importantly, *why*
it is built this way. Most of the interesting decisions here are about
measurement rather than about code, because the track's bar is not "does it
work on this one example" — it is:

> Throughput plus measured accuracy plus an honest exception list.
> One cherry-picked match proves nothing.

---

## 1. The problem

A merchant has two records of the same money, and they do not line up:

| Razorpay's transaction ledger | The bank's settlement feed |
|---|---|
| One row per **payment** | One row per **payout batch** |
| Gross amount, fee, GST, net | One net credit, no breakdown |
| Timestamped at capture | Dated at T+2 |

They are not supposed to line up 1:1. Razorpay nets out its fee and GST,
groups several payments into one bank credit, and pays out roughly two days
later. So "these two numbers differ" is the *normal* case, and the actual job
is separating the differences that are explainable from the ones that are not.

Three kinds of things go wrong, and they need different treatment:

1. **Explainable by arithmetic** — fees, GST, batching, recorded refunds,
   T+2 timing. No judgment needed; just do the maths.
2. **Genuinely ambiguous** — a settlement is ₹1,465 short. Is that a fee
   miscalculation, a refund nobody recorded, or a payment silently dropped
   from the batch? The numbers contain the answer, but reading them is
   inference, not arithmetic.
3. **Not actually wrong** — a payment from yesterday has no settlement
   because T+2 has not elapsed. Reporting this as a problem is crying wolf.

---

## 2. The shape of the system

```
                    ┌──────────────────────────────────────┐
   3 CSVs  ────────▶│  matcher.py                          │
  (any dataset)     │  deterministic arithmetic — NO AI    │
                    │                                      │
                    │  • nets fees, GST, recorded refunds  │
                    │  • reconstructs each payout batch    │
                    │  • flags duplicates by order_id      │
                    │  • checks T+2 against the window     │
                    └────────────┬─────────────────────────┘
                                 │
              ~90% resolved      │      ~10% flagged, with
              here, for free ◀───┤───▶  arithmetic EVIDENCE
                                 │      attached
                                 ▼
                    ┌──────────────────────────────────────┐
                    │  exception_agent.py                  │
                    │  packages each exception + signals    │
                    └────────────┬─────────────────────────┘
                                 │
                    ┌────────────▼─────────────────────────┐
                    │  reasoner.py                         │
                    │  ┌────────────────┐ ┌──────────────┐ │
                    │  │ LLM backend    │ │ rule baseline│ │
                    │  │ (Groq/Claude/  │ │  = CONTROL   │ │
                    │  │  Gemini)       │ │              │ │
                    │  └────────────────┘ └──────────────┘ │
                    │  names ONE root cause + confidence   │
                    └────────────┬─────────────────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
           report.py        qa_agent.py      grade.py
        throughput,      plain-English    ground truth —
        exception list   Q&A over the     opened ONLY here,
        report.html      reconciliation   and only afterwards
```

### Why two layers instead of one

The obvious "AI finance controller" is to hand every record to a model and
ask it to reconcile. That would be slower, more expensive, and *less
accurate* than arithmetic, because checking whether `976.40 + 1952.80`
equals `2929.20` is not a judgment call. An LLM can get that wrong. Python
cannot.

So the split is drawn on a specific line: **arithmetic below, judgment
above.** The matcher resolves everything a calculator can resolve. Only what
survives — around 10% of records — reaches a model. That is the design
decision the whole architecture exists to express, and it is why throughput
is reported as two separate phases (§5).

---

## 3. The evidence boundary

This is the single most important design decision in the project, and the
easiest one to get subtly wrong.

`matcher.py` computes `discrepancy_signals()` for every mismatch: factual,
arithmetic observations about the gap.

- Does the gap **exactly equal** one transaction's contribution?
- Is it a **refund-sized fraction** of one transaction's net?
- Did the bank pay **more** or **less** than owed?
- What is the **full composition** of the batch, transaction by transaction?

What it deliberately does **not** do is name a cause.

The temptation is obvious — the matcher already knows the gap equals
`txn_0192`'s net, so why not have it write `"cause": "missing_txn_in_batch"`
and be done? Because the moment it does, the AI layer has nothing left to
decide, and "our LLM attributes root causes with X% accuracy" becomes a
measurement of a string copy. The number would still be printable. It would
just no longer mean anything.

So the matcher hands over evidence, and the reasoner does the attribution.
That boundary is what makes the accuracy figure in §5 a real measurement.

Note that `batch_composition` hands over the raw per-transaction numbers, not
just the two convenience checks. Those checks use fixed thresholds (a gap
between 10% and 70% of a net is "refund-shaped"), and fixed thresholds are
exactly what a real discrepancy eventually falls outside of. A 6% unrecorded
refund is not refund-shaped by that rule — but it is still obviously 6% of
one transaction if you look at the numbers. Handing over the raw composition
is what lets the reasoning layer *beat* the rule instead of inheriting its
blind spots.

---

## 4. Accepting datasets that aren't ours

Every accuracy number here is measured on data our own generator produced.
That is unavoidable — you cannot grade a root cause you do not know — but it
raises a fair question: does this only work on data shaped exactly the way we
happened to write it?

Three separate things address that.

**Ingest tolerance.** `matcher.py` reads the dialects real exports actually
use: `1,234.56`, `Rs 1,234.56`, `(45.00)` for an accounting negative, ISO-T
timestamps with a timezone, `DD/MM/YYYY`, ID lists separated by semicolons,
commas, pipes or spaces, extra columns nobody asked for, and a missing
`refunds.csv` for a merchant who had none. `tests/test_ingest.py` asserts a
messy export reconciles to *identical numbers* as the canonical one — and,
just as importantly, that a genuinely short settlement is still flagged short
after going through the messy path. Robust parsing must not become forgiving
arithmetic.

**Loud failure.** A dataset we cannot reconcile raises `DatasetError` naming
the file, column and row. An empty ledger is refused rather than reported as
a flawless 100% match rate over nothing at all. A payout naming no
transactions is surfaced as an exception, never counted as clean.

**Regime variation, not just seed variation.** This one matters most. Running
a hundred datasets at a hundred seeds changes every amount, date and victim —
but leaves the *recipe* identical: still ~12% bad settlements, still a
duplicate every 20 transactions, still the same batch-size mix. That measures
robustness to different numbers, which is worth something, but quoting "120
datasets" while every one shares a recipe overstates what was tested.

`--vary` draws the regime itself from the seed: exception rate from 4% to
30%, batch sizes from all-singletons to batches of six, reporting windows
from 5 to 21 days, and refund/duplicate/stuck frequencies across a wide
range. A high-volume merchant with clean books and a small one with messy
books are different problems. The results in §5 cover both.

---

## 5. How accuracy is measured

`grade.py` is the honesty layer. It is the **only** file that opens
`ground_truth.json`, and it opens it only after the matcher and the reasoner
have committed to their answers. Neither of them can see it, so neither can
cheat.

Two different numbers are computed, because they answer different questions.

### 5a. Structural accuracy — did the matcher find the right things?

Reported as **precision and recall**, not a single "match rate", because the
two failures cost different things: missing a real exception loses the
merchant money, while flagging a clean settlement wastes an analyst's
afternoon. Every dataset contains legitimate, properly-recorded refunds that
exist specifically as **false-positive bait** — a matcher that ignored
`refunds.csv` would flag them and its precision would drop.

### 5b. Attribution accuracy — did the reasoner name the *true* cause?

An LLM that writes a fluent, confident, wrong explanation looks identical in
a demo to one that is right. The only way to tell them apart is to check its
answer against a cause you know for a fact, because you injected it.

Two things make this number honest:

**It is graded against a control.** A rule-based classifier reads the same
signals and picks a cause using a handful of if-statements. "Our LLM scored
X%" means nothing until you know what a few if-statements score on the same
task. The baseline is never tuned using ground truth — that would make it
cheat and destroy the comparison.

**Non-answers are not wrong answers.** An empty response, unparseable output
or an API error is our bug, not the model's mistake. They are counted and
excluded from the accuracy denominator, and reported separately. This is not
hypothetical: an unset `max_tokens` once made the model return empty content
on ~38% of a run, and scoring those as wrong would have published a 30.8%
figure that measured our own bug and called it a finding about AI accuracy.

**Orphans are excluded.** There is exactly one possible cause for a
settlement referencing a nonexistent transaction, so getting it right
demonstrates nothing and would inflate the score for free.

### 5c. The adversarial sample — the comparison that discriminates

Grading the LLM on a random sample mostly measures how often it agrees with
us about easy cases. Our first honest run scored 40/40 — and so did the rule
baseline. Two perfect scores on an easy exam say nothing about which grader
is better.

So `--hard-cases` sweeps many datasets with the **free** rule baseline, keeps
only the cases it gets *wrong*, and hands those to the model. The baseline's
accuracy on that sample is 0% by construction. The question stops being "can
the LLM do what rules already do" and becomes "when a hand-tuned threshold
breaks, does reasoning recover it?"

It is also cheap: the sweep costs nothing, and the model is only paid for on
the handful of cases that actually discriminate.

**And that sample needs a control of its own.** This is the subtlest trap in
the whole project, and we walked into it. The adversarial sample is not drawn
at random — it is *selected* by "the baseline was wrong", and the baseline is
wrong asymmetrically (it over-predicts `fee_or_gst_error`). That selection
skews the true-cause mix hard, 19:3 in our run. A classifier that ignores
every piece of evidence and always emits the commonest label scores **86.4%**
on that sample. The LLM scored **50%**.

So "the LLM recovered half the cases the rules missed" — which sounds like a
result — is a number *worse than guessing*, and would have been reported as a
success had the control not been computed. `grade.py` now computes the
majority-class control automatically and prints a warning when the model
fails to beat it, so this comparison cannot be quietly skipped again.

The constant guess is not a legitimate rival system: it exploits how the
sample was selected and would be useless on the real distribution. But it is
a necessary sanity check on the *metric*. Failing it means the evaluation has
not demonstrated the model adds value — which is the honest current status of
the AI attribution layer, recorded in `data/validation_hard.json`.

### 5d. Throughput is two numbers, not one

An earlier version reported one blended records/sec — total wall clock over
all records. That was misleading in both directions: it made the
deterministic matcher look thousands of times slower than it is, while
implying every record pays LLM latency. Neither is true.

Phase 1 scales with **record count**. Phase 2 scales with **exception
count**. Reporting them separately is what makes the architecture legible;
blending them hides the entire design decision and invites a fair question
("why is this so slow?") whose honest answer is "it isn't".

---

## 6. Design invariants

These are load-bearing. Breaking any one of them turns a real measurement
into a decorative one.

1. **`matcher.py` never imports `reasoner.py` and never opens
   `ground_truth.json`.** It is pure arithmetic and must stay that way.
2. **`matcher.py` emits evidence, never a cause.** See §3.
3. **`ground_truth.json` is opened only by `grade.py`,** and only after the
   matcher and reasoner have committed.
4. **`causes.py` is the single definition of the taxonomy,** imported by the
   generator, the reasoner and the grader. If those three drift, the metric
   silently becomes nonsense.
5. **The rule baseline is a control.** It must not be tuned using ground
   truth.
6. **A non-answer is not a wrong answer.** See §5b.
7. **Throughput is reported as two phases.** See §5d.
8. **Every accuracy figure is quoted next to what a dumb strategy scores on
   the same sample.** The rule baseline is that control for the general case;
   the majority-class guess is that control for the adversarial sample. A
   percentage with no control beside it is not a result. See §5c for the run
   where this changed the conclusion entirely.

---

## 7. Module reference

| File | Role |
|---|---|
| `data/generate_data.py` | Synthetic data + hidden ground truth. `--seed`, `--n`, `--out-dir`, `--vary`. Injects 4 mismatch causes + 2 missing-settlement causes + orphans + duplicates. |
| `src/causes.py` | The cause taxonomy. Single source of truth (invariant 4). |
| `src/matcher.py` | Deterministic matching, tolerant ingest, evidence signals. No AI. |
| `src/exception_agent.py` | Packages flagged exceptions + signals for the reasoner. |
| `src/reasoner.py` | Pluggable LLM backends (Groq / Anthropic / Gemini) + the rule baseline control. |
| `src/report.py` | Two-phase throughput, accuracy, honest exception list. `--data-dir`, `--out`, `--html`. |
| `src/render_report.py` | `report.json` → readable `report.html`. |
| `src/qa_agent.py` | Plain-English Q&A, two-tier context, no RAG by design. |
| `src/grade.py` | The honesty layer: structural + attribution grading, adversarial mode. |
| `tests/test_ingest.py` | Proves the ingest layer accepts datasets that aren't ours. |

### Why the Q&A agent has no RAG

At this batch size the model is grounded by handing it the structured result
set directly. That is a deliberate choice, not a shortcut: at this scale
there is no retrieval to get wrong, and it is simpler to audit. Retrieval
earns its complexity on thousands of records; forcing it here would just be
an extra thing to get wrong for no benefit.

The context is two-tier, because "full context" does not mean "every raw
field for every record". Handing the model all transactions' fee breakdowns
to answer a question about one broken settlement blew past the free tier's
per-request token limit for no reason. So: a compact status roster for
everything, plus full raw records only for what is actually flagged.

---

## 8. Honest limitations

- **The data is synthetic.** It has to be — attribution accuracy is only
  measurable against causes we injected. The ingest tests (§4) address
  *format* generality; they cannot address whether real Razorpay
  discrepancies have the same *statistical shape* as our injected ones.
- **The taxonomy is closed.** Seven causes. A real cause nobody enumerated
  falls through to the nearest label rather than being reported as novel.
  The rule baseline in particular exists because we knew in advance exactly
  which causes we injected; on genuinely unseen data it would degrade in a
  way the LLM plausibly would not, which is precisely what §5c probes.
- **Duplicate detection assumes `order_id` is meaningful.** A merchant who
  legitimately reuses order IDs would see false positives.
- **Date ambiguity is not guessed at.** `03/04/2026` is 3 April day-first and
  4 March month-first, and no parser can tell from the string. Day-first is
  tried first because this is an Indian payments dataset; the behaviour is
  documented rather than silently assumed.
- **T+2 is hardcoded** as the settlement SLA. Merchants on different payout
  schedules would need `SETTLEMENT_SLA_DAYS` changed.
