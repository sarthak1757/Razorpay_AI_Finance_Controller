"""
Turns data/report.json into a clean, readable HTML report - something a
judge (or you, recording the pitch video) can actually look at, instead of
raw JSON. Doesn't touch the pipeline logic at all, purely presentation.

Run report.py first to generate/refresh data/report.json, then this.
"""

import json
from datetime import datetime, timezone

STATUS_COLORS = {
    "AMOUNT_MISMATCH": "#b45309",
    "ORPHAN": "#b91c1c",
    "MISSING_SETTLEMENT": "#b91c1c",
}
CONFIDENCE_COLORS = {"high": "#15803d", "medium": "#b45309", "low": "#b91c1c"}


def render(report: dict) -> str:
    t, a, m = report["throughput"], report["accuracy"], report["money_flagged"]
    exceptions = report["exceptions"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def stat_tile(label, value, sub=""):
        return f"""
        <div class="tile">
          <div class="tile-value">{value}</div>
          <div class="tile-label">{label}</div>
          {f'<div class="tile-sub">{sub}</div>' if sub else ''}
        </div>"""

    tiles = "".join([
        stat_tile("Settlement match rate", f"{a['settlement_match_rate']*100:.1f}%",
                   f"{a['settlements_matched']}/{a['settlements_total']} settlements"),
        stat_tile("Transaction settlement rate", f"{a['transaction_settlement_rate']*100:.1f}%",
                   f"{a['transactions_settled']}/{a['transactions_total_excl_duplicates']} (dupes excluded)"),
        stat_tile("Records processed", f"{t['total_records_processed']}",
                   f"{t['transactions']} txns + {t['settlements']} settlements"),
        stat_tile("Amount at risk", f"Rs {m['amount_at_risk_unsettled']:,.2f}", "sitting in unsettled transactions"),
        stat_tile("Amount flagged", f"Rs {m['amount_flagged_mismatch']:,.2f}", "in settlement discrepancies"),
        stat_tile("Exceptions found", f"{len(exceptions)}", f"reasoned by: {report['reasoner_used']}"),
    ])

    rows = []
    for e in exceptions:
        label = e.get("settlement_id") or e.get("transaction_id")
        color = STATUS_COLORS.get(e["kind"], "#6b7280")
        conf_color = CONFIDENCE_COLORS.get(e["confidence"], "#6b7280")
        rows.append(f"""
        <tr>
          <td><span class="badge" style="background:{color}1a;color:{color}">{e['kind']}</span></td>
          <td class="mono">{label}</td>
          <td>{e['explanation']}</td>
          <td><span class="badge" style="background:{conf_color}1a;color:{conf_color}">{e['confidence']}</span></td>
          <td>{e['recommended_action']}</td>
        </tr>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Reconciliation Report</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; background: #f8fafc; color: #1e293b;
          margin: 0; padding: 40px 24px; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #64748b; font-size: 13px; margin-bottom: 28px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 32px; }}
  .tile {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; }}
  .tile-value {{ font-size: 22px; font-weight: 600; }}
  .tile-label {{ font-size: 12px; color: #475569; margin-top: 4px; }}
  .tile-sub {{ font-size: 11px; color: #94a3b8; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e2e8f0;
           border-radius: 10px; overflow: hidden; font-size: 13px; }}
  th {{ text-align: left; background: #f1f5f9; padding: 10px 12px; font-size: 11px;
        text-transform: uppercase; color: #64748b; letter-spacing: 0.04em; }}
  td {{ padding: 12px; border-top: 1px solid #f1f5f9; vertical-align: top; }}
  .mono {{ font-family: ui-monospace, monospace; font-size: 12px; }}
  .badge {{ padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .footer {{ margin-top: 20px; font-size: 11px; color: #94a3b8; }}
</style></head>
<body><div class="wrap">
  <h1>AI Finance Controller - Reconciliation Report</h1>
  <div class="subtitle">Multi-source reconciliation: Razorpay transactions vs. bank settlements</div>
  <div class="tiles">{tiles}</div>
  <table>
    <thead><tr><th>Type</th><th>ID</th><th>Explanation</th><th>Confidence</th><th>Recommended action</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <div class="footer">Generated {generated_at} - full machine-readable detail in report.json</div>
</div></body></html>"""


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Render report.json as readable HTML.")
    p.add_argument("--report", default="data/report.json",
                   help="Path to a report.json produced by report.py")
    p.add_argument("--out", default=None,
                   help="Where to write the HTML. Defaults to the report's name with .html")
    args = p.parse_args()

    with open(args.report) as f:
        report = json.load(f)
    out_path = args.out or os.path.splitext(args.report)[0] + ".html"
    with open(out_path, "w") as f:
        f.write(render(report))
    print(f"Wrote {out_path}")
