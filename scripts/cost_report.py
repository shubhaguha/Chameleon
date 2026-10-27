"""Summarize a dataset's cost ledger (written by the Gateway during repairs).

    python3 scripts/cost_report.py --data-dir ./data --dataset utkface_myrun_123 [--json]

Same numbers as GET /v1/datasets/<dataset>/cost/ on the Gateway, but works without the stack running.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Gateway"))
import cost_ledger  # noqa: E402


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 100 else f"{v:,.1f}"
    return f"{v:,}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="CHAMELEON_DATA_DIR")
    parser.add_argument("--dataset", required=True, help="dataset id, as in results/<dataset>/")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args()

    path = cost_ledger.ledger_path(args.dataset, os.path.join(args.data_dir, "results"))
    rows = cost_ledger.read_rows(path)
    if not rows:
        sys.exit(f"no ledger at {path}")
    summary = cost_ledger.summarize(rows)
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"{args.dataset}  ({', '.join(summary['backends']) or 'unknown backend'})")
    print(f"queries {summary['queries']}  failed {summary['failed']}  ddt rejected {summary['ddt_rejected']}  "
          f"reviewed {summary['reviewed']}  accepted {summary['accepted']}  "
          f"queries/accepted {fmt(summary['queries_per_accepted'])}  gateway time {summary['seconds'] / 3600:.2f} h")
    keys = sorted(summary["usage"])
    if keys:
        print(f"\n{'usage':<22}{'total':>14}{'per accepted':>16}")
        for k in keys:
            print(f"{k:<22}{fmt(summary['usage'][k]):>14}{fmt(summary['usage_per_accepted'].get(k)):>16}")
    print(f"\n{'pattern':<10}{'queries':>9}{'ddt rej':>9}{'accepted':>10}{'q/acc':>8}{'cost_usd':>12}")
    for pattern, g in sorted(summary["by_pattern"].items()):
        print(f"{pattern:<10}{g['queries']:>9}{g['ddt_rejected']:>9}{g['accepted']:>10}"
              f"{fmt(g['queries_per_accepted']):>8}{fmt(g['usage'].get('cost_usd')):>12}")


if __name__ == "__main__":
    main()
