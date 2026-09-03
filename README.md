# AI Finance Controller — reconciliation with measured accuracy

**Razorpay AI Buildathon — Track 4.**

Reconciles a Razorpay transaction ledger against a bank settlement feed.
Deterministic arithmetic resolves everything a calculator can resolve — fees,
GST, batching, recorded refunds, T+2 timing. An LLM is used only for the ~8%
that survives, to attribute a root cause to each remaining exception.

The track's bar is *"throughput plus measured accuracy plus an honest
exception list. One cherry-picked match proves nothing."* So every number
below is measured against a ground truth the system cannot see, across
hundreds of independently generated datasets — **including the AI layer**,
which is graded against a rule-based control so that "the LLM scored X%" can
be read next to "and a handful of if-statements scored Y%".

---

## Measured results

### Deterministic matcher

Graded against hidden ground truth. Precision and recall are reported
separately, not as a single "match rate", because the two failures cost
different things: a missed exception loses the merchant money, a false flag
wastes an analyst's afternoon.

| | Fixed recipe, 30 seeds | **Varied regimes, 40 seeds** |
|---|---|---|
| Datasets | 120 | **160** |
| Records | 205,479 | **263,673** |
| Precision | 1.0000 | **1.0000** |
| Recall | 1.0000 | **1.0000** |
| True positives | 23,527 | **33,182** |
| False positives | **0** | **0** |
| False negatives | **0** | **0** |
| Datasets with zero errors | 120/120 | **160/160** |

The right-hand column is the one that matters. Changing the seed changes
every amount and date but leaves the *recipe* identical — still ~12% bad
settlements, still the same batch-size mix. `--vary` draws the regime itself
from the seed: exception rates from 4% to 30%, batch sizes from
all-singletons to batches of six, reporting windows from 5 to 21 days. A
high-volume merchant with clean books and a small one with messy books are
different problems. Both hold.

Every dataset also contains legitimate, properly-recorded refunds that exist
purely as **false-positive bait**. A matcher that ignored `refunds.csv` would
flag them and its precision would drop below 1.0.

### Throughput — two phases, reported separately

Phase 1 scales with record count; phase 2 scales with *exception* count. One
blended number would hide the entire architecture.

| Transactions | Records | Matching | Records/sec | Exceptions | Reaching the AI |
|---|---|---|---|---|---|
| 1,000 | 1,563 | 0.011s | 137,960 | 128 | 8.2% |
| 5,000 | 7,871 | 0.058s | 136,578 | 647 | 8.2% |
| 25,000 | 39,217 | 0.311s | 125,975 | 3,223 | 8.2% |
| 100,000 | 156,725 | **1.408s** | 111,291 | 12,874 | 8.2% |

156,725 records reconciled in 1.4 seconds. The AI share stays flat at 8.2%
at every scale — over 91% of records never touch a model.

### Root-cause attribution

The reasoner must name the true cause of each exception from a fixed
taxonomy of seven. Graded against the injected truth.

**Rule baseline (the control): 96.97%** — 24,528 of 25,294 cases across the
160 varied-regime datasets, orphans excluded.

Its calibration is the interesting part:

| Stated confidence | Correct | Accuracy |
|---|---|---|
| high | 16,544 / 16,544 | **1.0000** |
| medium | 7,984 / 8,750 | 0.9120 |

Every single "high confidence" call, across 16,544 of them, was right. That
means confidence is a usable routing signal rather than decoration — you can
auto-clear the high-confidence findings and put a human on the rest.

**LLM on the adversarial sample:** see below.

### The comparison that actually discriminates

Grading an LLM on a random sample mostly measures how often it agrees with
you about easy cases. Our first honest run scored 40/40 — and so did the rule
baseline. Two perfect scores on an easy exam say nothing about which grader
is better.

So `--hard-cases` sweeps many datasets with the **free** rule baseline, keeps
only the cases it gets *wrong*, and grades the LLM on those. The baseline
scores 0% on that sample by construction. The question becomes: *when a
hand-tuned threshold breaks, does reasoning recover it?*

**The answer, so far, is no — and that is reported here rather than buried.**

| On 34 cases the rule baseline got wrong | |
|---|---|
| Rule baseline, by construction | 0 / 34 = **0.000** |
| LLM (`gpt-oss-20b`), of cases it answered | 11 / 22 = **0.500** |
| **Control — always guess the commonest label** | 19 / 22 = **0.864** |
| Excluded: no usable answer returned | 12 (10 × HTTP 429, 2 truncated) |

The LLM did **not** beat a strategy that ignores the evidence entirely and
always says `unrecorded_refund`.

That constant guess is not a real rival system — the sample is *selected* by
"the baseline was wrong", the baseline errs asymmetrically, and that skews the
label mix 19:3. A constant guess exploits the selection and would be useless
on the true distribution. But the model failing to beat it means **this run
does not demonstrate that the LLM recovers the baseline's failures**, and
`grade.py` now computes that control automatically so the comparison can
never be quietly skipped.

Two caveats, both pointing the same way:

- **The run was cut short.** The free tier's 200,000 tokens-per-day ceiling
  was exhausted partway through, so 12 of 34 cases returned no usable answer.
  Those are excluded from the denominator — a call that never returned is our
  infrastructure failing, not the model being wrong — which leaves a thin
  22-case sample. `reasoning_effort=low` now cuts token spend enough for a
  full run to fit the budget; **this needs re-running.**
- **22 cases is small.** The honest reading is "not demonstrated", not
  "disproven".

What this does establish: **the deterministic layer is where the value is.**
The matcher is exact, the rule baseline attributes 96.97% correctly with
perfectly calibrated confidence, and on the residual hard tail the LLM has
not yet been shown to add anything. A submission that quoted "50% recovery on
adversarial cases" as a win would have been quoting a number worse than
guessing.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 data/generate_data.py          # canonical dataset
python3 src/report.py --html           # reconcile + readable report
```

Without an API key everything still runs — the reasoner falls back to the
rule baseline, so the pipeline is testable end to end for free. For real LLM
attribution, put a key in `.env`:

```
GROQ_API_KEY=...          # or ANTHROPIC_API_KEY / GEMINI_API_KEY
```

> Install `requirements.txt` in the venv. Without `python-dotenv` the `.env`
> file is never read, the key never loads, and the system silently falls back
> to the rule baseline.

## Run it on your own data

Point it at any directory containing the three CSVs:

```bash
python3 src/report.py --data-dir /path/to/my/data --html
python3 src/qa_agent.py --data-dir /path/to/my/data "why was stl_0008 short?"
```

**Required columns** (extra columns are carried through and ignored):

| File | Columns |
|---|---|
| `razorpay_transactions.csv` | `transaction_id`, `order_id`, `amount`, `razorpay_fee`, `gst_on_fee`, `net_amount`, `timestamp` |
| `bank_settlements.csv` | `settlement_id`, `settled_amount`, `settlement_date`, `transaction_ids_included` |
| `refunds.csv` *(optional)* | `transaction_id`, `refund_amount` |

The ingest layer accepts the dialects real exports actually use — `1,234.56`,
`Rs 1,234.56`, `(45.00)` for accounting negatives, ISO-T timestamps with
timezones, `DD/MM/YYYY`, and ID lists separated by `;` `,` `|` or spaces. A
dataset it genuinely cannot read fails loudly, naming the file, column and
row — it never half-reconciles and reports a confident wrong total.

```bash
python3 tests/test_ingest.py      # 12 tests proving the above
```

## Reproduce every number

```bash
# Structural accuracy across 160 varied-regime datasets (no API key needed)
python3 src/grade.py --vary --seeds $(seq 1 40) --sizes 60 300 1000 3000 \
    --out data/validation_varied.json

# Adversarial LLM evaluation — grades the model ONLY on cases the rule
# baseline gets wrong. Needs an API key.
python3 src/grade.py --hard-cases --seeds $(seq 1 40) --sizes 300 \
    --sleep 5 --out data/validation_hard.json
```

`--sleep` paces calls under a free tier's tokens-per-minute ceiling. Without
it the SDK sits in backoff and the run takes far longer.

Run everything from the repo root, not from `src/`.

## Layout

```
data/generate_data.py   synthetic data + hidden ground truth (--seed/--n/--vary)
src/causes.py           the cause taxonomy — single source of truth
src/matcher.py          deterministic matching + evidence signals. No AI.
src/reasoner.py         pluggable LLM backends + the rule baseline control
src/exception_agent.py  feeds flagged exceptions to the reasoner
src/report.py           two-phase throughput, accuracy, honest exception list
src/render_report.py    report.json -> report.html
src/qa_agent.py         plain-English Q&A (two-tier context, no RAG by design)
src/grade.py            the honesty layer: structural + attribution grading
tests/test_ingest.py    proves the ingest layer accepts datasets that aren't ours
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and the reasoning
behind it — in particular §3, the evidence boundary, which is what makes the
attribution accuracy figure a real measurement rather than a formality.

## Honest limitations

- **The data is synthetic.** It has to be — attribution accuracy is only
  measurable against causes you injected. The ingest tests address *format*
  generality; they cannot prove real discrepancies have the same statistical
  shape as our injected ones.
- **The taxonomy is closed.** Seven causes. A real cause nobody enumerated
  gets mapped to the nearest label rather than reported as novel.
- **The rule baseline knows the answers exist.** Its thresholds were written
  by someone who knew which four causes were injected. On genuinely unseen
  data it would degrade in a way the LLM plausibly would not — which is
  exactly what the adversarial evaluation probes.
- **T+2 is hardcoded** as the settlement SLA, and duplicate detection assumes
  `order_id` is meaningful.
