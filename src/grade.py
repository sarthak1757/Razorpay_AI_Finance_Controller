"""
The honesty layer: grades BOTH halves of the system against ground truth.

Track 4's bar is "throughput plus measured accuracy plus an honest exception
list. One cherry-picked match proves nothing." Two things follow from that
which most of this kind of project skips:

1. The DETERMINISTIC matcher's accuracy has to be measured on more than one
   dataset, or it's a lucky seed rather than a result.
2. The AI layer's accuracy has to be measured AT ALL. An LLM that writes a
   fluent, confident, wrong explanation looks identical in a demo to one
   that's right. The only way to tell them apart is to check its answer
   against a cause you know for a fact, because you injected it.

So this file computes two separate numbers:
  - structural accuracy: did the matcher find the real exceptions, and did it
    wrongly flag anything clean? (precision/recall per category)
  - attribution accuracy: given a flagged exception, did the reasoner name the
    TRUE root cause? Measured for the LLM and for the rule baseline side by
    side, because "our AI scored 90%" means nothing until you know what a
    handful of if-statements scores on the same task.

Neither matcher.py nor reasoner.py can see ground_truth.json. Only this file
opens it, and only after they've both committed to their answers.

Usage:
    python3 src/grade.py                      # grade the current data/ dataset
    python3 src/grade.py --seeds 1 2 3 4 5    # multi-seed structural validation
    python3 src/grade.py --seeds 1 2 3 --n 300 --with-ai
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from matcher import load_data, match          # noqa: E402
from exception_agent import build_cases       # noqa: E402
from reasoner import (  # noqa: E402
    get_reasoner, RuleBaselineReasoner, DailyQuotaExhausted,
)
from causes import BENIGN_CAUSES, CAUSE_UNKNOWN  # noqa: E402
from generate_data import generate            # noqa: E402


# --------------------------------------------------------------------------
# Structural accuracy: did the deterministic matcher find the right things?
# --------------------------------------------------------------------------

def grade_structural(settlement_results, transaction_results, ground_truth):
    """Precision/recall for the matcher, per exception category.

    Reported as precision AND recall rather than a single "match rate"
    because they fail differently and both matter: missing a real exception
    loses the merchant money, while flagging a clean settlement wastes an
    analyst's afternoon. The legitimate recorded refunds in every dataset
    exist specifically as false-positive bait for the second kind.
    """
    true_causes = ground_truth["true_causes"]

    truth = {
        "AMOUNT_MISMATCH": {
            sid for sid, c in true_causes.items()
            if sid.startswith("stl_") and c != "orphan_reference"
        },
        "ORPHAN": {
            sid for sid, c in true_causes.items() if c == "orphan_reference"
        },
        "MISSING_SETTLEMENT": {
            tid for tid, c in true_causes.items() if tid.startswith("txn_")
        },
        "DUPLICATE": set(ground_truth["duplicate_entry"]),
    }

    detected = {
        "AMOUNT_MISMATCH": {r["settlement_id"] for r in settlement_results
                            if r["status"] == "AMOUNT_MISMATCH"},
        "ORPHAN": {r["settlement_id"] for r in settlement_results
                   if r["status"] == "ORPHAN"},
        "MISSING_SETTLEMENT": {tid for tid, r in transaction_results.items()
                               if r["status"] == "MISSING_SETTLEMENT"},
        "DUPLICATE": {tid for tid, r in transaction_results.items()
                      if r["status"] == "DUPLICATE"},
    }

    per_category = {}
    total_tp = total_fp = total_fn = 0
    for category in truth:
        tp = len(truth[category] & detected[category])
        fp = len(detected[category] - truth[category])
        fn = len(truth[category] - detected[category])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_category[category] = {
            "expected": len(truth[category]),
            "detected": len(detected[category]),
            "true_positives": tp,
            "false_positives": sorted(detected[category] - truth[category]),
            "false_negatives": sorted(truth[category] - detected[category]),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else 1.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else 1.0,
        }

    return {
        "per_category": per_category,
        "totals": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) else 1.0,
            "recall": round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) else 1.0,
            "perfect": total_fp == 0 and total_fn == 0,
        },
    }


# --------------------------------------------------------------------------
# Attribution accuracy: did the reasoner name the TRUE cause?
# --------------------------------------------------------------------------

def grade_novel_detection(scored):
    """Can the reasoner admit it doesn't recognise something?

    WHY THIS CANNOT BE ONE NUMBER:
    The `unknown_anomaly` label is trivially gameable in both directions, and
    each direction looks fine on a metric that ignores the other.

      - Never say "unknown"      -> perfect accuracy on the known causes,
                                    catches zero novel problems, and files
                                    every surprise under a confident wrong
                                    label. This is what the system did BEFORE
                                    the label existed.
      - Always say "unknown"     -> catches every novel problem and destroys
                                    attribution entirely.

    So three numbers are reported together, and they have to be read together:

      recall     - of the genuinely novel cases, how many were flagged as
                   unrecognised? (did we catch the surprises?)
      precision  - of the cases called unknown, how many really were novel?
                   (or are we crying "I don't know" over ordinary fee errors?)
      accuracy on KNOWN causes - did adding the escape hatch damage the
                   ordinary attribution it is supposed to leave alone?

    That last one is the regression guard. A reasoner that improves its recall
    by retreating to "unknown" whenever it is unsure will show it there.
    """
    novel_true = [g for g in scored if g["true_cause"] == CAUSE_UNKNOWN]
    called_unknown = [g for g in scored if g["predicted_cause"] == CAUSE_UNKNOWN]
    known_true = [g for g in scored if g["true_cause"] != CAUSE_UNKNOWN]

    if not scored:
        return None
    # NOTE: this deliberately reports even when no novel case was injected and
    # none was claimed. An earlier version bailed out in that situation, which
    # quietly biased the regression guard: only datasets where the reasoner had
    # said "unknown" at least once contributed to accuracy_on_known_causes, so
    # the guard was computed over exactly the datasets where something had gone
    # wrong. A regression guard that only looks at the runs with regressions is
    # not a guard. Empty denominators surface as None instead.

    caught = sum(1 for g in novel_true if g["predicted_cause"] == CAUSE_UNKNOWN)
    right_to_claim = sum(1 for g in called_unknown if g["true_cause"] == CAUSE_UNKNOWN)
    known_correct = sum(1 for g in known_true if g["correct"])

    # What the novel cases got mislabelled AS, when they were missed. A model
    # that confidently calls a chargeback a "fee error" is the exact failure
    # this whole feature exists to expose, so name it rather than burying it
    # in an aggregate.
    missed_as = defaultdict(int)
    for g in novel_true:
        if g["predicted_cause"] != CAUSE_UNKNOWN:
            missed_as[g["predicted_cause"]] += 1

    return {
        "novel_cases": len(novel_true),
        "novel_caught": caught,
        "recall": round(caught / len(novel_true), 4) if novel_true else None,
        "times_called_unknown": len(called_unknown),
        "correctly_called_unknown": right_to_claim,
        "precision": round(right_to_claim / len(called_unknown), 4) if called_unknown else None,
        "known_cause_cases": len(known_true),
        "known_cause_correct": known_correct,
        "accuracy_on_known_causes": round(known_correct / len(known_true), 4) if known_true else None,
        "novel_missed_as": dict(missed_as),
    }


def grade_attribution(cases, reasoner, ground_truth, progress=False, sleep_between=0.0):
    """Ask a reasoner to explain every case, then score its causes.

    ORPHAN cases are scored separately and kept out of the headline number:
    there is only one possible cause for an orphan, so getting it right
    demonstrates nothing and would inflate the score for free.

    Progress is printed for real-LLM runs because on a free tier these calls
    serialize behind a tokens-per-minute limit - a silent terminal for ten
    minutes is indistinguishable from a hang, and the honest fix is to show
    the work rather than to hide it. A Ctrl-C mid-run keeps whatever has been
    graded so far instead of throwing it away.
    """
    true_causes = ground_truth["true_causes"]

    graded = []
    start = time.perf_counter()
    interrupted = False
    errors = 0

    for i, case in enumerate(cases, 1):
        case_id = case.get("settlement_id") or case.get("transaction_id")
        try:
            result = reasoner.explain(case)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n  [interrupted after {i - 1}/{len(cases)} cases - scoring what we have]")
            break
        except Exception as e:  # rate limit, transport error, bad JSON upstream
            errors += 1
            result = {
                "likely_cause": "api_error",
                "confidence": "low",
                "explanation": f"[{type(e).__name__}: {e}]",
                "recommended_action": "Retry - the call failed rather than the model being wrong.",
            }

        correct = result.get("likely_cause") == true_causes.get(case_id)
        graded.append({
            "case_id": case_id,
            "kind": case["kind"],
            "true_cause": true_causes.get(case_id),
            "predicted_cause": result.get("likely_cause"),
            "correct": correct,
            "confidence": result.get("confidence"),
            "explanation": result.get("explanation"),
            "recommended_action": result.get("recommended_action"),
        })

        if progress:
            running = sum(1 for g in graded if g["correct"] and g["kind"] != "ORPHAN")
            denom = sum(1 for g in graded if g["kind"] != "ORPHAN")
            rate = f"{running}/{denom}" if denom else "n/a"
            mark = "OK " if correct else "MISS"
            print(f"  [{i:>3}/{len(cases)}] {mark} {case_id:<12} "
                  f"true={true_causes.get(case_id):<28} got={result.get('likely_cause'):<28} "
                  f"running={rate}", flush=True)

        if sleep_between and i < len(cases):
            time.sleep(sleep_between)

    elapsed = time.perf_counter() - start

    scored = [g for g in graded if g["kind"] != "ORPHAN"]
    correct = sum(1 for g in scored if g["correct"])

    by_kind = defaultdict(lambda: {"total": 0, "correct": 0})
    for g in scored:
        by_kind[g["kind"]]["total"] += 1
        by_kind[g["kind"]]["correct"] += int(g["correct"])

    # Confidence calibration: a model that says "high" should be right more
    # often than one that says "low". If it isn't, its confidence is noise and
    # nobody should route work by it.
    by_confidence = defaultdict(lambda: {"total": 0, "correct": 0})
    for g in scored:
        bucket = by_confidence[g["confidence"] or "unstated"]
        bucket["total"] += 1
        bucket["correct"] += int(g["correct"])

    return {
        "reasoner": reasoner.name,
        "cases_scored": len(scored),
        "cases_correct": correct,
        "attribution_accuracy": round(correct / len(scored), 4) if scored else None,
        "novel_detection": grade_novel_detection(scored),
        "orphan_cases_excluded": len(graded) - len(scored),
        "interrupted": interrupted,
        "api_errors": errors,
        "seconds": round(elapsed, 3),
        "avg_seconds_per_case": round(elapsed / len(graded), 3) if graded else None,
        "by_kind": {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None}
            for k, v in by_kind.items()
        },
        "by_confidence": {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None}
            for k, v in by_confidence.items()
        },
        "detail": graded,
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def grade_dataset(data_dir, with_ai=False, max_ai_cases=None, sleep_between=0.0):
    transactions, settlements, refunds = load_data(data_dir)
    with open(os.path.join(data_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)

    t0 = time.perf_counter()
    settlement_results, transaction_results = match(transactions, settlements, refunds)
    match_seconds = time.perf_counter() - t0

    total_records = len(transactions) + len(settlements)
    result = {
        "records": total_records,
        "transactions": len(transactions),
        "settlements": len(settlements),
        "match_seconds": round(match_seconds, 6),
        "match_records_per_second": round(total_records / match_seconds) if match_seconds else None,
        "structural": grade_structural(settlement_results, transaction_results, ground_truth),
    }

    cases = build_cases(settlement_results, transaction_results)
    result["exceptions_flagged"] = len(cases)

    # The rule baseline always runs - it's free, and it's the control.
    result["baseline"] = grade_attribution(cases, RuleBaselineReasoner(), ground_truth)

    if with_ai:
        reasoner = get_reasoner()
        if isinstance(reasoner, RuleBaselineReasoner):
            print("[grade] --with-ai requested but no API key found; skipping LLM grading.")
        else:
            ai_cases = cases if max_ai_cases is None else cases[:max_ai_cases]
            if len(ai_cases) < len(cases):
                print(f"  grading {len(ai_cases)}/{len(cases)} exceptions with {reasoner.name} "
                      f"(capped by --max-ai-cases)")
            else:
                print(f"  grading {len(ai_cases)} exceptions with {reasoner.name}")
            result["ai"] = grade_attribution(
                ai_cases, reasoner, ground_truth,
                progress=True, sleep_between=sleep_between,
            )

    return result


def collect_novel_sample(seeds, sizes, novel_rate, per_class, vary=True, sample_seed=0):
    """Build a CLASS-BALANCED sample for novel-cause detection.

    WHY BALANCED, AND WHY THAT DOESN'T FAKE THE RESULT:
    Novel cases are a minority (~20% of mismatches). Sampling at the natural
    rate would spend most of a limited API budget re-confirming ordinary
    cases, and leave too few novel ones to estimate recall from. So this takes
    an equal number of each.

    The catch is that PRECISION depends on the class balance - measuring it on
    a 50/50 sample would report a precision this system will never achieve on
    real data, which would be a flattering lie. Recall and the false-alarm
    rate, however, are base-rate INDEPENDENT: each is measured within one
    class. So we measure those two, and derive precision at the true base rate
    (measured separately from the full free baseline sweep):

        precision = (recall * p) / (recall * p + false_alarm_rate * (1 - p))

    That reports what the system would actually do on real data, from a sample
    small enough to afford.

    The sample is drawn with a fixed seed and returned in a fixed order, so
    running a second model over it grades the IDENTICAL cases. Two models
    compared on two different samples is not a comparison.
    """
    novel, known = [], []
    tmp_root = tempfile.mkdtemp(prefix="recon_novel_sample_")
    try:
        for size in sizes:
            for seed in seeds:
                ds_dir = os.path.join(tmp_root, f"seed{seed}_n{size}")
                generate(seed=seed, n_transactions=size, out_dir=ds_dir, verbose=False,
                         vary=vary, novel_rate=novel_rate)
                transactions, settlements, refunds = load_data(ds_dir)
                with open(os.path.join(ds_dir, "ground_truth.json")) as f:
                    ground_truth = json.load(f)
                settlement_results, transaction_results = match(transactions, settlements, refunds)
                for case in build_cases(settlement_results, transaction_results):
                    if case["kind"] != "AMOUNT_MISMATCH":
                        continue  # novel mechanisms only affect settlement amounts
                    case_id = case["settlement_id"]
                    true_cause = ground_truth["true_causes"].get(case_id)
                    item = {
                        "label": f"seed={seed} n={size}",
                        "case": case,
                        "case_id": case_id,
                        "true_cause": true_cause,
                        "novel_mechanism": ground_truth.get("novel_mechanisms", {}).get(case_id),
                    }
                    (novel if true_cause == CAUSE_UNKNOWN else known).append(item)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    total = len(novel) + len(known)
    base_rate = len(novel) / total if total else 0.0

    rng = random.Random(sample_seed)
    rng.shuffle(novel)
    rng.shuffle(known)
    sample = novel[:per_class] + known[:per_class]
    rng.shuffle(sample)  # interleave, so a rate-limited run doesn't lose one whole class
    return sample, base_rate, len(novel), len(known)


def score_novel_sample(sample, reasoner, base_rate, sleep_between=0.0, progress=True):
    """Run one reasoner over the balanced sample and derive its real-world
    precision. Non-answers are excluded from every denominator, for the same
    reason as everywhere else in this file: a call that never returned is our
    infrastructure failing, not the model being wrong."""
    NON_ANSWERS = {"empty_response", "unparsed", "api_error", "off_taxonomy"}
    detail = []
    aborted = None
    consecutive_failures = 0
    for i, item in enumerate(sample, 1):
        try:
            result = reasoner.explain(item["case"])
        except DailyQuotaExhausted as e:
            # No point continuing - every remaining case would be an api_error,
            # and a run that grinds out 100 non-answers looks superficially
            # like a completed evaluation.
            aborted = f"daily quota exhausted after {i - 1} cases: {e}"
            print(f"\n  [ABORT] {aborted}", flush=True)
            break
        except Exception as e:
            result = {"likely_cause": "api_error", "confidence": "low",
                      "explanation": f"[{type(e).__name__}: {e}]"}
        predicted = result.get("likely_cause")
        answered = predicted not in NON_ANSWERS
        is_novel = item["true_cause"] == CAUSE_UNKNOWN
        said_unknown = predicted == CAUSE_UNKNOWN
        detail.append({
            "case_id": item["case_id"], "label": item["label"],
            "true_cause": item["true_cause"], "novel_mechanism": item["novel_mechanism"],
            "predicted": predicted, "answered": answered,
            "is_novel": is_novel, "said_unknown": said_unknown,
            "exact_correct": answered and predicted == item["true_cause"],
            "confidence": result.get("confidence"),
            "explanation": result.get("explanation"),
        })
        if progress:
            mark = "OK " if detail[-1]["exact_correct"] else ("FAIL" if not answered else "MISS")
            print(f"  [{i:>3}/{len(sample)}] {mark} {item['case_id']:<12} "
                  f"true={item['true_cause']:<22} got={predicted:<22}"
                  f"{' <novel:' + str(item['novel_mechanism']) + '>' if is_novel else ''}",
                  flush=True)
        # A run that is failing every call is not producing data, it is
        # producing an illusion of data. Stop and say why.
        consecutive_failures = 0 if answered else consecutive_failures + 1
        if consecutive_failures >= 8:
            aborted = (f"{consecutive_failures} consecutive non-answers after {i} cases "
                       f"- the API is failing, not the model. Last: {predicted}")
            print(f"\n  [ABORT] {aborted}", flush=True)
            break

        if sleep_between and i < len(sample):
            time.sleep(sleep_between)

    answered = [d for d in detail if d["answered"]]
    novel_a = [d for d in answered if d["is_novel"]]
    known_a = [d for d in answered if not d["is_novel"]]

    caught = sum(1 for d in novel_a if d["said_unknown"])
    false_alarms = sum(1 for d in known_a if d["said_unknown"])

    recall = caught / len(novel_a) if novel_a else None
    far = false_alarms / len(known_a) if known_a else None

    # Precision the system would achieve at the REAL novel rate, not the
    # rate we sampled at. See collect_novel_sample's docstring.
    precision = None
    if recall is not None and far is not None:
        denom = recall * base_rate + far * (1 - base_rate)
        precision = round(recall * base_rate / denom, 4) if denom else None

    return {
        "reasoner": reasoner.name,
        # A truncated run must never be mistakable for a complete one - same
        # reasoning as the run_note on data/validation_hard.json.
        "aborted_early": aborted is not None,
        "abort_reason": aborted,
        "cases_attempted": len(detail),
        "sample_size": len(sample),
        "answered": len(answered),
        "non_answers": len(detail) - len(answered),
        "novel_scored": len(novel_a),
        "novel_caught": caught,
        "recall": round(recall, 4) if recall is not None else None,
        "known_scored": len(known_a),
        "false_alarms": false_alarms,
        "false_alarm_rate": round(far, 4) if far is not None else None,
        "precision_at_true_base_rate": precision,
        "base_rate_used": round(base_rate, 4),
        "exact_accuracy_on_known": round(
            sum(1 for d in known_a if d["exact_correct"]) / len(known_a), 4) if known_a else None,
        "detail": detail,
    }


def collect_hard_cases(seeds, sizes, vary=False, novel_rate=0.0):
    """Find the exceptions the rule baseline gets WRONG, across many datasets.

    WHY THIS MODE EXISTS:
    Grading the LLM on a random sample mostly measures how often it agrees
    with us about easy cases. Our first real run scored 40/40 - and so did the
    rule baseline, because at that dataset size the genuinely ambiguous cases
    barely occur. Two perfect scores on an easy exam say nothing about which
    grader is better.

    So this sweeps a lot of datasets with the FREE baseline, keeps only the
    cases it fails, and hands those to the model. It's a deliberately
    adversarial sample: the question isn't "can the LLM do what rules already
    do", it's "when a hand-tuned threshold breaks, does reasoning recover it?"
    It's also cheap - the sweep costs nothing and the LLM only pays for the
    handful of cases that actually discriminate.
    """
    hard = []
    total_cases = 0
    tmp_root = tempfile.mkdtemp(prefix="recon_hardcases_")
    try:
        for size in sizes:
            for seed in seeds:
                ds_dir = os.path.join(tmp_root, f"seed{seed}_n{size}")
                generate(seed=seed, n_transactions=size, out_dir=ds_dir, verbose=False,
                         vary=vary, novel_rate=novel_rate)
                transactions, settlements, refunds = load_data(ds_dir)
                with open(os.path.join(ds_dir, "ground_truth.json")) as f:
                    ground_truth = json.load(f)
                settlement_results, transaction_results = match(transactions, settlements, refunds)
                cases = build_cases(settlement_results, transaction_results)
                total_cases += len([c for c in cases if c["kind"] != "ORPHAN"])

                baseline = RuleBaselineReasoner()
                for case in cases:
                    if case["kind"] == "ORPHAN":
                        continue
                    case_id = case.get("settlement_id") or case.get("transaction_id")
                    predicted = baseline.explain(case).get("likely_cause")
                    true_cause = ground_truth["true_causes"].get(case_id)
                    if predicted != true_cause:
                        hard.append({
                            "label": f"seed={seed} n={size}",
                            "case": case,
                            "case_id": case_id,
                            "true_cause": true_cause,
                            "baseline_predicted": predicted,
                        })
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return hard, total_cases


def run_hard_case_mode(seeds, sizes, sleep_between, vary=False, novel_rate=0.0):
    print(f"Sweeping {len(seeds) * len(sizes)} datasets with the free rule baseline "
          f"to find its failures...")
    hard, total_cases = collect_hard_cases(seeds, sizes, vary=vary, novel_rate=novel_rate)
    print(f"  baseline failed {len(hard)} of {total_cases} scoreable cases "
          f"({len(hard) / total_cases * 100:.2f}%)\n")

    if not hard:
        print("No baseline failures found - nothing to discriminate on. "
              "Try more seeds or a larger --sizes.")
        return None

    reasoner = get_reasoner()
    if isinstance(reasoner, RuleBaselineReasoner):
        print("No API key found - cannot grade the LLM on these. Set one in .env.")
        return None

    print(f"Grading {len(hard)} baseline-failure cases with {reasoner.name}:")
    correct = 0
    infra_failures = 0
    detail = []
    # Answers we never actually received - empty content, unreadable output, a
    # failed call. Scoring these as "the model got it wrong" would report our
    # own bugs as findings about model quality, so they're counted separately
    # and excluded from the accuracy denominator.
    NON_ANSWERS = {"empty_response", "unparsed", "api_error", "off_taxonomy"}
    for i, item in enumerate(hard, 1):
        try:
            result = reasoner.explain(item["case"])
        except Exception as e:
            result = {"likely_cause": "api_error", "confidence": "low",
                      "explanation": f"[{type(e).__name__}: {e}]"}
        predicted = result.get("likely_cause")
        answered = predicted not in NON_ANSWERS
        is_right = answered and predicted == item["true_cause"]
        correct += int(is_right)
        infra_failures += int(not answered)
        detail.append({**{k: v for k, v in item.items() if k != "case"},
                       "llm_predicted": predicted,
                       "llm_answered": answered,
                       "llm_confidence": result.get("confidence"),
                       "llm_explanation": result.get("explanation"),
                       "llm_correct": is_right})
        mark = "OK " if is_right else ("FAIL" if not answered else "MISS")
        print(f"  [{i:>2}/{len(hard)}] {mark} "
              f"{item['label']} {item['case_id']:<12} "
              f"true={item['true_cause']:<22} baseline={item['baseline_predicted']:<22} "
              f"llm={predicted:<22} conf={result.get('confidence')}", flush=True)
        if sleep_between and i < len(hard):
            time.sleep(sleep_between)

    answered = len(hard) - infra_failures

    # THE CONTROL THIS SAMPLE ACTUALLY NEEDS.
    #
    # "The LLM scored 50% on cases the rule baseline got wrong" sounds like a
    # win, and it is worthless without knowing what a degenerate strategy
    # scores on the SAME sample. It turns out that matters enormously here,
    # because this sample is not drawn at random - it is selected by "the
    # baseline was wrong", and the baseline is wrong asymmetrically. That
    # selection skews the true-cause mix hard, and a classifier that ignores
    # the evidence entirely and always says the commonest label can score very
    # well on it.
    #
    # That constant guess is NOT a legitimate rival system - it exploits how
    # the sample was selected and would be terrible on the real distribution.
    # But it is a necessary sanity check on the METRIC: if the model cannot
    # beat it, then this evaluation has not demonstrated that the model adds
    # anything, and reporting the raw percentage as a success would be exactly
    # the kind of decorative number this project exists to avoid.
    answered_items = [d for d in detail if d["llm_answered"]]
    majority = None
    if answered_items:
        counts = defaultdict(int)
        for d in answered_items:
            counts[d["true_cause"]] += 1
        label, hits = max(counts.items(), key=lambda kv: kv[1])
        majority = {
            "label": label,
            "correct": hits,
            "total": len(answered_items),
            "accuracy": round(hits / len(answered_items), 4),
            "true_cause_mix": dict(counts),
        }

    llm_acc = correct / answered if answered else 0
    print("\n" + "=" * 66)
    print("ADVERSARIAL SAMPLE: cases the rule baseline got wrong")
    print("=" * 66)
    print(f"  baseline accuracy on these, by definition: 0/{len(hard)} = 0.000")
    print(f"  LLM accuracy (of cases it answered):       {correct}/{answered} = "
          f"{llm_acc:.3f}")
    if infra_failures:
        print(f"  excluded - no usable answer returned:      {infra_failures}"
              f"  (empty/unparseable/API error, not a wrong answer)")
    print(f"  LLM accuracy (counting non-answers wrong): {correct}/{len(hard)} = "
          f"{correct / len(hard):.3f}")

    if majority:
        print(f"\n  CONTROL - always guess '{majority['label']}':"
              f"    {majority['correct']}/{majority['total']} = {majority['accuracy']:.3f}")
        print(f"  true-cause mix in this sample: {majority['true_cause_mix']}")
        if majority["accuracy"] >= llm_acc:
            print("\n  >> READ THIS: the model did NOT beat a constant guess on this")
            print("     sample. Selecting on 'the baseline was wrong' skews the label")
            print("     mix, and a strategy that ignores the evidence entirely scores")
            print("     at least as well. This run therefore does NOT show the LLM")
            print("     recovering the baseline's failures. Report it as such.")
        else:
            print("\n  The model beats the constant-guess control on this sample,")
            print("  so it is using the evidence rather than the label skew.")

    return {
        "majority_class_control": majority,
        "hard_cases_found": len(hard),
        "total_scoreable_cases_swept": total_cases,
        "baseline_failure_rate": round(len(hard) / total_cases, 4),
        "llm_correct_on_baseline_failures": correct,
        "llm_cases_answered": answered,
        "llm_infrastructure_failures": infra_failures,
        "llm_accuracy_of_answered": round(correct / answered, 4) if answered else None,
        "llm_accuracy_strict": round(correct / len(hard), 4),
        "reasoner": reasoner.name,
        "detail": detail,
    }


def run_novel_eval_mode(seeds, sizes, novel_rate, per_class, sleep_between,
                        backend, out_path):
    """LLM vs rule baseline on novel-cause detection, same sample for both."""
    print(f"Building a balanced sample from {len(seeds) * len(sizes)} datasets "
          f"(novel_rate={novel_rate})...")
    sample, base_rate, n_novel, n_known = collect_novel_sample(
        seeds, sizes, novel_rate, per_class)
    print(f"  found {n_novel} novel + {n_known} known mismatch cases "
          f"(true novel base rate {base_rate:.3f})")
    print(f"  sampled {len(sample)} cases "
          f"({sum(1 for s in sample if s['true_cause'] == CAUSE_UNKNOWN)} novel / "
          f"{sum(1 for s in sample if s['true_cause'] != CAUSE_UNKNOWN)} known)\n")

    results = {}

    print("Rule baseline (free control) on this sample:")
    results["baseline"] = score_novel_sample(
        sample, RuleBaselineReasoner(), base_rate, progress=False)
    b = results["baseline"]
    print(f"  recall {b['recall']}  false-alarm rate {b['false_alarm_rate']}  "
          f"precision@base {b['precision_at_true_base_rate']}\n")

    reasoner = get_reasoner(backend)
    print(f"Grading {len(sample)} cases with {reasoner.name}:")
    results[reasoner.name] = score_novel_sample(
        sample, reasoner, base_rate, sleep_between=sleep_between)

    print("\n" + "=" * 74)
    print("NOVEL-CAUSE DETECTION - can it admit the numbers match nothing known?")
    print("=" * 74)
    header = f"  {'':<14} {'recall':>8} {'false-alarm':>12} {'precision@base':>15} {'non-answers':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, r in results.items():
        print(f"  {name:<14} {str(r['recall']):>8} {str(r['false_alarm_rate']):>12} "
              f"{str(r['precision_at_true_base_rate']):>15} {r['non_answers']:>12}")
    print(f"\n  CONTROLS - never say unknown: recall 0.000")
    print(f"             always say unknown: recall 1.000, "
          f"precision {base_rate:.3f}, known-cause accuracy 0.000")

    llm = results[reasoner.name]
    if llm["recall"] is not None and b["recall"] is not None:
        delta = llm["recall"] - b["recall"]
        print(f"\n  {reasoner.name} vs rule baseline on recall: {delta:+.3f}")
        if delta <= 0:
            print("  >> The model did NOT beat the threshold at spotting novel causes.")
        elif llm["false_alarm_rate"] > b["false_alarm_rate"]:
            print("  >> Higher recall, but ALSO more false alarms - check whether it is")
            print("     just saying 'unknown' more often rather than reasoning better.")
        else:
            print("  >> Higher recall at no extra false-alarm cost - a real improvement.")

    payload = {"novel_evaluation": {
        "base_rate": base_rate, "sample_size": len(sample),
        "population_novel": n_novel, "population_known": n_known,
        "results": results,
    }}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n(Full detail saved to {out_path})")
    return payload


def main():
    p = argparse.ArgumentParser(description="Grade the matcher and the reasoning layer against ground truth.")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Generate and grade fresh datasets at these seeds. Omit to grade ./data as-is.")
    p.add_argument("--n", type=int, default=60, help="Transactions per generated dataset.")
    p.add_argument("--sizes", type=int, nargs="*", default=None,
                   help="Grade across several dataset sizes (uses the first seed).")
    p.add_argument("--with-ai", action="store_true",
                   help="Also grade the real LLM (needs an API key; costs calls).")
    p.add_argument("--max-ai-cases", type=int, default=None,
                   help="Cap LLM-graded exceptions PER DATASET. On a free tier the "
                        "tokens-per-minute limit makes calls serialize, so an uncapped "
                        "run across several large datasets can take 15+ minutes.")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Seconds to wait between LLM calls. Use ~1-2 if you keep hitting "
                        "rate-limit backoff, which is slower than pacing yourself.")
    p.add_argument("--hard-cases", action="store_true",
                   help="Adversarial mode: sweep datasets with the free rule baseline, "
                        "then grade the LLM ONLY on the cases the baseline got wrong. "
                        "This is the comparison that actually discriminates.")
    p.add_argument("--novel-eval", action="store_true",
                   help="Grade an LLM AND the rule baseline on novel-cause detection, "
                        "using one class-balanced sample so both see identical cases. "
                        "Reports recall and false-alarm rate (both base-rate independent) "
                        "and derives precision at the true novel rate.")
    p.add_argument("--sample-per-class", type=int, default=50,
                   help="Novel cases AND known cases to sample for --novel-eval. "
                        "Total LLM calls = 2x this.")
    p.add_argument("--reasoner", default=None,
                   help="Force a backend: groq, gemini, anthropic, baseline. "
                        "Without this, the first key found in .env wins - which makes "
                        "comparing two models impossible.")
    p.add_argument("--novel-rate", type=float, default=0.0,
                   help="Inject this fraction of settlements with a mechanism OUTSIDE the "
                        "taxonomy. The correct answer for those is 'unknown_anomaly'. Use "
                        "this to measure whether the system admits it doesn't recognise "
                        "something, instead of confidently guessing the nearest label.")
    p.add_argument("--vary", action="store_true",
                   help="Vary each dataset's REGIME (exception rate, batch sizes, refund "
                        "and duplicate frequency, reporting window), not just its amounts. "
                        "Without this, every seed is a different draw of the same recipe.")
    p.add_argument("--out", default="data/validation.json")
    args = p.parse_args()

    if args.novel_eval:
        run_novel_eval_mode(
            args.seeds or list(range(1, 21)),
            args.sizes or [args.n],
            args.novel_rate or 0.08,
            args.sample_per_class,
            args.sleep,
            args.reasoner,
            args.out if args.out != "data/validation.json" else "data/validation_novel_eval.json",
        )
        return

    if args.hard_cases:
        result = run_hard_case_mode(
            args.seeds or list(range(1, 11)),
            args.sizes or [args.n],
            args.sleep,
            vary=args.vary,
            novel_rate=args.novel_rate,
        )
        if result:
            with open(args.out, "w") as f:
                json.dump({"hard_case_evaluation": result}, f, indent=2)
            print(f"\n(Full detail saved to {args.out})")
        return

    runs = []

    if args.seeds is None and args.sizes is None:
        print("Grading the existing dataset in data/ ...")
        runs.append({"label": "data/", **grade_dataset(
            "data", with_ai=args.with_ai,
            max_ai_cases=args.max_ai_cases, sleep_between=args.sleep)})
    else:
        seeds = args.seeds or [42]
        sizes = args.sizes or [args.n]
        tmp_root = tempfile.mkdtemp(prefix="recon_validation_")
        try:
            for size in sizes:
                for seed in seeds:
                    ds_dir = os.path.join(tmp_root, f"seed{seed}_n{size}")
                    generate(seed=seed, n_transactions=size, out_dir=ds_dir, verbose=False,
                             vary=args.vary, novel_rate=args.novel_rate)
                    if args.with_ai:
                        print(f"\nseed={seed} n={size}:")
                    run = grade_dataset(
                        ds_dir, with_ai=args.with_ai,
                        max_ai_cases=args.max_ai_cases, sleep_between=args.sleep)
                    runs.append({"label": f"seed={seed} n={size}", "seed": seed, "n": size, **run})
                    print(f"  {runs[-1]['label']:<18} "
                          f"{run['records']:>5} records | "
                          f"match {run['match_records_per_second']:>7,}/s | "
                          f"structural P={run['structural']['totals']['precision']:.3f} "
                          f"R={run['structural']['totals']['recall']:.3f} | "
                          f"baseline attribution {run['baseline']['attribution_accuracy']:.3f}"
                          + (f" | AI {run['ai']['attribution_accuracy']:.3f}" if "ai" in run else ""))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    summary = summarize(runs)
    payload = {"summary": summary, "runs": runs}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print_summary(summary)
    print(f"\n(Full detail saved to {args.out})")


def summarize(runs):
    n = len(runs)
    perfect = sum(1 for r in runs if r["structural"]["totals"]["perfect"])
    total_tp = sum(r["structural"]["totals"]["true_positives"] for r in runs)
    total_fp = sum(r["structural"]["totals"]["false_positives"] for r in runs)
    total_fn = sum(r["structural"]["totals"]["false_negatives"] for r in runs)

    summary = {
        "datasets_graded": n,
        "total_records": sum(r["records"] for r in runs),
        "structural": {
            "datasets_with_zero_errors": perfect,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) else 1.0,
            "recall": round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) else 1.0,
        },
        "match_records_per_second": {
            "min": min(r["match_records_per_second"] for r in runs),
            "max": max(r["match_records_per_second"] for r in runs),
        },
    }

    for layer in ("baseline", "ai"):
        graded = [r[layer] for r in runs if layer in r]
        if not graded:
            continue
        scored = sum(g["cases_scored"] for g in graded)
        correct = sum(g["cases_correct"] for g in graded)
        conf = defaultdict(lambda: {"total": 0, "correct": 0})
        for g in graded:
            for bucket, v in g["by_confidence"].items():
                conf[bucket]["total"] += v["total"]
                conf[bucket]["correct"] += v["correct"]
        # Novel-cause detection totalled across every dataset in the run.
        # Summed from raw counts rather than averaging per-dataset rates, so a
        # dataset that happened to contain two novel cases doesn't carry the
        # same weight as one that contained forty.
        nd = defaultdict(int)
        missed_as = defaultdict(int)
        saw_novel = False
        for g in graded:
            n = g.get("novel_detection")
            if not n:
                continue
            saw_novel = True
            for k in ("novel_cases", "novel_caught", "times_called_unknown",
                      "correctly_called_unknown", "known_cause_cases", "known_cause_correct"):
                nd[k] += n[k]
            for k, v in n["novel_missed_as"].items():
                missed_as[k] += v

        summary[layer] = {
            "reasoner": graded[0]["reasoner"],
            "cases_scored": scored,
            "cases_correct": correct,
            "attribution_accuracy": round(correct / scored, 4) if scored else None,
            "novel_detection": ({
                **dict(nd),
                "recall": round(nd["novel_caught"] / nd["novel_cases"], 4) if nd["novel_cases"] else None,
                "precision": round(nd["correctly_called_unknown"] / nd["times_called_unknown"], 4)
                             if nd["times_called_unknown"] else None,
                "accuracy_on_known_causes": round(nd["known_cause_correct"] / nd["known_cause_cases"], 4)
                             if nd["known_cause_cases"] else None,
                "novel_missed_as": dict(missed_as),
            } if saw_novel else None),
            "by_confidence": {
                k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None}
                for k, v in sorted(conf.items())
            },
        }

    return summary


def print_summary(s):
    print("\n" + "=" * 66)
    print("VALIDATION SUMMARY")
    print("=" * 66)
    print(f"\nDatasets graded: {s['datasets_graded']}  ({s['total_records']} records total)")

    st = s["structural"]
    print("\nDeterministic matcher (structural accuracy):")
    print(f"  precision {st['precision']:.4f}   recall {st['recall']:.4f}")
    print(f"  {st['true_positives']} true positives, {st['false_positives']} false positives, "
          f"{st['false_negatives']} false negatives")
    print(f"  datasets with zero errors: {st['datasets_with_zero_errors']}/{s['datasets_graded']}")
    print(f"  matching throughput: {s['match_records_per_second']['min']:,} - "
          f"{s['match_records_per_second']['max']:,} records/sec")

    for layer, label in (("baseline", "Rule baseline (control)"), ("ai", "LLM reasoner")):
        if layer not in s:
            continue
        a = s[layer]
        print(f"\n{label} - root cause attribution [{a['reasoner']}]:")
        print(f"  accuracy {a['attribution_accuracy']:.4f} "
              f"({a['cases_correct']}/{a['cases_scored']} cases, orphans excluded)")
        print("  by stated confidence:")
        for bucket, v in a["by_confidence"].items():
            print(f"    {bucket:<10} {v['correct']:>3}/{v['total']:<3} = "
                  f"{(v['accuracy'] if v['accuracy'] is not None else 0):.3f}")

        n = a.get("novel_detection")
        if n:
            def rate(value):
                return f"{value:.3f}" if value is not None else "  n/a"

            print("  novel-cause detection (mechanisms OUTSIDE the taxonomy):")
            print(f"    recall     {n['novel_caught']:>5}/{n['novel_cases']:<5} = "
                  f"{rate(n['recall'])}   (novel cases correctly flagged unrecognised)")
            print(f"    precision  {n['correctly_called_unknown']:>5}/{n['times_called_unknown']:<5} = "
                  f"{rate(n['precision'])}   "
                  f"(of everything called unknown, how much really was)")
            print(f"    accuracy on KNOWN causes: {n['known_cause_correct']}/{n['known_cause_cases']} = "
                  f"{(n['accuracy_on_known_causes'] or 0):.4f}  <- regression guard")
            if n["novel_missed_as"]:
                print(f"    novel cases missed, filed instead as: {n['novel_missed_as']}")
            # The two degenerate strategies, so neither number can be read
            # alone (invariant 8). "Never say unknown" is what the system did
            # before this label existed; "always say unknown" is the other way
            # to game it. A useful reasoner has to beat BOTH at once.
            if n["novel_cases"]:
                base = n["novel_cases"] / (n["novel_cases"] + n["known_cause_cases"])
                print(f"    CONTROLS - never say unknown: recall 0.000, "
                      f"known-cause accuracy unaffected")
                print(f"               always say unknown: recall 1.000, "
                      f"precision {base:.3f}, known-cause accuracy 0.000")

    if "ai" in s and "baseline" in s:
        delta = s["ai"]["attribution_accuracy"] - s["baseline"]["attribution_accuracy"]
        print(f"\n  LLM vs rule baseline: {delta:+.4f}")


if __name__ == "__main__":
    main()
