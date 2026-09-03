"""
Strip per-case detail out of a validation run, leaving the numbers.

WHY: a full sweep records every graded case - its true cause, the predicted
cause, the model's explanation. Across 160 datasets and ~25,000 cases that is
21 MB of JSON that nobody reads, in a repo a reviewer has to clone. The
summary carries every figure quoted in the README and is a few KB.

The full dumps stay reproducible - the exact commands are in the README - so
nothing is lost except the weight. Run:

    python3 scripts/summarize_validation.py data/validation.json ...
"""

import json
import os
import sys


def strip(payload):
    """Drop the detail arrays, keep every computed number."""
    out = {k: v for k, v in payload.items() if k != "runs"}

    if "runs" in payload:
        runs = []
        for run in payload["runs"]:
            slim = {}
            for key, value in run.items():
                if isinstance(value, dict) and "detail" in value:
                    slim[key] = {k: v for k, v in value.items() if k != "detail"}
                else:
                    slim[key] = value
            runs.append(slim)
        out["runs"] = runs

    # Hard-case runs keep their detail: it's small, and the per-case rows are
    # the evidence for the majority-class finding rather than bulk noise.
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        with open(path) as f:
            payload = json.load(f)

        summary_path = path.replace(".json", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(strip(payload), f, indent=2)

        before = os.path.getsize(path) / 1024
        after = os.path.getsize(summary_path) / 1024
        print(f"  {path}: {before:,.0f} KB -> {summary_path}: {after:,.0f} KB")


if __name__ == "__main__":
    main()
