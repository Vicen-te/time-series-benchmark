"""Fail the build when a benchmark run is worse than the committed baseline.

Usage
-----
    python scripts/check_regression.py
    python scripts/check_regression.py --tolerance 0.05 --metric mae
    python scripts/check_regression.py --summary-out "$GITHUB_STEP_SUMMARY"

Exits 1 when any model regresses past the tolerance, when a model in the
baseline is missing from the run, or when the models were not all scored on
the same rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TABLES_DIR
from src.quality_gate import compare_summaries, load_summary


DEFAULT_BASELINE = TABLES_DIR / "baseline_cpu.csv"
DEFAULT_CANDIDATE = TABLES_DIR / "benchmark_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--metric", default="rmse")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Fraction of the baseline the metric may worsen by (default 0.03).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Append the markdown report to this file as well as stdout.",
    )
    parser.add_argument(
        "--allow-unequal-n",
        action="store_true",
        help="Skip the check that every model was scored on the same rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path in (args.baseline, args.candidate):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    report = compare_summaries(
        baseline=load_summary(args.baseline, args.metric),
        candidate=load_summary(args.candidate, args.metric),
        metric=args.metric,
        tolerance=args.tolerance,
        require_equal_n=not args.allow_unequal_n,
    )

    markdown = (
        f"### Benchmark gate\n\n"
        f"Baseline `{args.baseline.name}` vs this run.\n\n"
        f"{report.to_markdown()}\n"
    )
    print(markdown)

    if args.summary_out is not None:
        with args.summary_out.open("a", encoding="utf-8") as fh:
            fh.write(markdown)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
