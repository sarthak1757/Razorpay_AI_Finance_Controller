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
import shutil
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from matcher import load_data, match          # noqa: E402
from exception_agent import build_cases       # noqa: E402
from reasoner import get_reasoner, RuleBaselineReasoner  # noqa: E402
from causes import BENIGN_CAUSES              # noqa: E402
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


def collect_hard_cases(seeds, sizes):
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
                generate(seed=seed, n_transactions=size, out_dir=ds_dir, verbose=False)
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


def run_hard_case_mode(seeds, sizes, sleep_between):
    print(f"Sweeping {len(seeds) * len(sizes)} datasets with the free rule baseline "
          f"to find its failures...")
    hard, total_cases = collect_hard_cases(seeds, sizes)
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
    detail = []
    for i, item in enumerate(hard, 1):
        try:
            result = reasoner.explain(item["case"])
        except Exception as e:
            result = {"likely_cause": "api_error", "confidence": "low",
                      "explanation": f"[{type(e).__name__}: {e}]"}
        predicted = result.get("likely_cause")
        is_right = predicted == item["true_cause"]
        correct += int(is_right)
        detail.append({**{k: v for k, v in item.items() if k != "case"},
                       "llm_predicted": predicted,
                       "llm_confidence": result.get("confidence"),
                       "llm_explanation": result.get("explanation"),
                       "llm_correct": is_right})
        print(f"  [{i:>2}/{len(hard)}] {'OK ' if is_right else 'MISS'} "
              f"{item['label']} {item['case_id']:<12} "
              f"true={item['true_cause']:<22} baseline={item['baseline_predicted']:<22} "
              f"llm={predicted:<22} conf={result.get('confidence')}", flush=True)
        if sleep_between and i < len(hard):
            time.sleep(sleep_between)

    print("\n" + "=" * 66)
    print("ADVERSARIAL SAMPLE: cases the rule baseline got wrong")
    print("=" * 66)
    print(f"  baseline accuracy on these, by definition: 0/{len(hard)} = 0.000")
    print(f"  LLM accuracy on the same cases:            {correct}/{len(hard)} = "
          f"{correct / len(hard):.3f}")
    return {
        "hard_cases_found": len(hard),
        "total_scoreable_cases_swept": total_cases,
        "baseline_failure_rate": round(len(hard) / total_cases, 4),
        "llm_correct_on_baseline_failures": correct,
        "llm_accuracy_on_baseline_failures": round(correct / len(hard), 4),
        "reasoner": reasoner.name,
        "detail": detail,
    }


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
    p.add_argument("--out", default="data/validation.json")
    args = p.parse_args()

    if args.hard_cases:
        result = run_hard_case_mode(
            args.seeds or list(range(1, 11)),
            args.sizes or [args.n],
            args.sleep,
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
                    generate(seed=seed, n_transactions=size, out_dir=ds_dir, verbose=False)
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
        summary[layer] = {
            "reasoner": graded[0]["reasoner"],
            "cases_scored": scored,
            "cases_correct": correct,
            "attribution_accuracy": round(correct / scored, 4) if scored else None,
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

    if "ai" in s and "baseline" in s:
        delta = s["ai"]["attribution_accuracy"] - s["baseline"]["attribution_accuracy"]
        print(f"\n  LLM vs rule baseline: {delta:+.4f}")


if __name__ == "__main__":
    main()
