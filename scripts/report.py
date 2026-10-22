#!/usr/bin/env python
"""Print README-ready Markdown tables from a saved benchmark JSON.

Keeps the numbers quoted in the documentation verifiable: everything in the
README's result tables can be regenerated from ``results/*_benchmark.json``.

Usage::

    python scripts/report.py                         # every saved benchmark
    python scripts/report.py --assay A4GRB6_PSEAI_Chen_2020
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drucion.config import RESULTS_DIR
from drucion.evaluation.plots import METHOD_LABELS

METRIC_LABELS = {
    "best_measured": "Best measured fitness",
    "auc_normalised": "Normalised AUC",
    "top1pct_recall": "Top-1% variants found",
    "evals_to_threshold": "Measurements to 99th pct",
}


def _fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def summary_table(d: dict) -> str:
    rows = [
        "| Method | Best measured | Normalised AUC | Top-1% variants found "
        "| Measurements to reach 99th pct |",
        "|---|---|---|---|---|",
    ]
    for m in d["methods"]:
        s = d["summary"][m]
        evals = s["evals_to_threshold_mean"]
        hit = s.get("threshold_hit_rate") or 0.0
        if evals is None:
            etxt = "never"
        else:
            etxt = f"{evals:.0f}"
            if hit < 1.0:
                etxt += f" ({hit * 100:.0f}% of seeds)"
        label = METHOD_LABELS.get(m, m)
        rows.append(
            f"| {label} | {_fmt(s['best_measured_mean'])} ± "
            f"{_fmt(s['best_measured_std'])} | {_fmt(s['auc_normalised_mean'])} ± "
            f"{_fmt(s['auc_normalised_std'])} | "
            f"{s['top1pct_recall_mean'] * 100:.1f}% | {etxt} |"
        )
    return "\n".join(rows)


def paired_table(d: dict, method: str = "ucb") -> str:
    paired = d.get("paired") or {}
    entry = (paired.get("methods") or {}).get(method)
    if not entry:
        return f"_(no paired data for {method})_"
    rows = ["| Metric | Seeds won | Mean difference | p |", "|---|---|---|---|"]
    for key, label in METRIC_LABELS.items():
        m = entry["metrics"].get(key)
        if not m:
            continue
        won = f"{m['wins']} / {m['n']}"
        if m["ties"]:
            won += f" ({m['ties']} tied)"
        diff = f"{m['mean_difference']:+.3f}"
        p = m["wilcoxon_p"]
        ptxt = "—" if p is None else f"{p:.3f}"
        if p is not None and p < 0.05:
            ptxt = f"**{ptxt}**"
        rows.append(f"| {label} | {won} | {diff} | {ptxt} |")
    return "\n".join(rows)


def surrogate_table(d: dict) -> str:
    sc = d.get("surrogate_curves") or {}
    if not sc:
        return "_(no surrogate curves in this file)_"
    rows = ["| Method | Final held-out ρ | Normalised AUC |", "|---|---|---|"]
    for m in d["methods"]:
        if m not in sc:
            continue
        rows.append(
            f"| {METHOD_LABELS.get(m, m)} | {sc[m]['mean'][-1]:.3f} | "
            f"{_fmt(d['summary'][m]['auc_normalised_mean'])} |"
        )
    return "\n".join(rows)


def efficiency_lines(d: dict) -> str:
    eff = d.get("efficiency")
    if not eff:
        return "_(no efficiency data in this file)_"
    out = [
        f"Baseline ({METHOD_LABELS.get(eff['baseline'], eff['baseline'])}) "
        f"final best = {eff['target']:.3f}"
    ]
    for m, v in eff["methods"].items():
        if m == eff["baseline"]:
            continue
        if v["budget_to_match"] is None:
            out.append(f"  {METHOD_LABELS.get(m, m)}: never matched it")
        else:
            out.append(
                f"  {METHOD_LABELS.get(m, m)}: matched at "
                f"{v['budget_to_match']} measurements "
                f"({v['savings_fraction'] * 100:.0f}% fewer)"
            )
    return "\n".join(out)


def report(path: Path) -> None:
    d = json.loads(path.read_text())
    print(f"\n{'=' * 78}\n{d['assay_id']}  ({len(d['seeds'])} seeds)\n{'=' * 78}")
    ps = d["pool_stats"]
    print(
        f"pool: mean {ps['pool_mean']:.3f}  p90 {ps['p90']:.3f}  "
        f"p99 {ps['p99']:.3f}  max {ps['pool_max']:.3f}\n"
    )
    print("## Summary\n")
    print(summary_table(d))
    print("\n## Paired vs random (UCB)\n")
    print(paired_table(d, "ucb"))
    print("\n## Surrogate accuracy\n")
    print(surrogate_table(d))
    print("\n## Experimental efficiency\n")
    print(efficiency_lines(d))
    if d.get("notes"):
        print("\n## Run notes\n")
        for n in dict.fromkeys(d["notes"]):
            print(f"  - {n}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assay", default=None)
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    results_dir = Path(args.results_dir or RESULTS_DIR)
    if args.assay:
        paths = [results_dir / f"{args.assay}_benchmark.json"]
    else:
        paths = sorted(results_dir.glob("*_benchmark.json"))
    if not paths:
        print(f"No benchmark JSON found in {results_dir}")
        return 1
    for path in paths:
        if not path.exists():
            print(f"Missing: {path}")
            return 1
        report(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
