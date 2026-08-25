"""Measure input drift between the training window and the scored window.

Usage
-----
    python scripts/check_drift.py
    python scripts/check_drift.py --write-reference
    python scripts/check_drift.py --tolerance 0.02 --summary-out "$GITHUB_STEP_SUMMARY"

ETTh1 is a fixed CSV, so the drift this reports does not change from day to
day: it is a standing property of the split, and a large one. The check is
therefore not "did the world move since yesterday" but "is the feature
pipeline still producing the distribution it produced when the reference was
recorded". A PSI that shifts means the data or the features changed, which is
worth failing a build over; the absolute numbers are worth reading once and
then keeping in view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import CONFIG, TABLES_DIR, TARGET_COLUMN
from src.data import load_etth1, temporal_split
from src.drift import compare_to_reference, drift_report, to_markdown


DEFAULT_REFERENCE = TABLES_DIR / "drift_reference.csv"
DEFAULT_OUTPUT = TABLES_DIR / "drift_report.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Overwrite the reference with the values measured now.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="How far a feature's PSI may move from the reference (default 0.02).",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def build_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature matrices for the train and test windows, target column included."""

    from src.features import build_feature_matrix

    splits = temporal_split(load_etth1())
    frames = []
    for slice_ in (splits.train, splits.test):
        X, y, _ = build_feature_matrix(
            slice_, target=TARGET_COLUMN, horizon=CONFIG.horizon
        )
        frames.append(X.assign(**{f"{TARGET_COLUMN}_target": y}))
    return frames[0], frames[1]


def main() -> int:
    args = parse_args()

    train_X, test_X = build_matrices()
    report = drift_report(train_X, test_X, bins=args.bins)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    if args.write_reference:
        report.to_csv(args.reference, index=False)
        print(f"Reference written to {args.reference}")
        return 0

    failures: list[str] = []
    if args.reference.exists():
        failures = compare_to_reference(
            current=report,
            reference=pd.read_csv(args.reference),
            tolerance=args.tolerance,
        )
    else:
        print(f"note: {args.reference} not found, reporting without a comparison")

    counts = report["band"].value_counts().to_dict()
    verdict = "unchanged" if not failures else "CHANGED"
    markdown = (
        "### Input drift, train window vs scored window\n\n"
        f"{len(report)} features: "
        + ", ".join(f"{n} {band}" for band, n in counts.items())
        + ".\n\n"
        + to_markdown(report)
        + f"\n\nAgainst the recorded reference: **{verdict}** "
        f"(tolerance {args.tolerance:.4f} on PSI).\n"
    )
    if failures:
        markdown += "\n" + "\n".join(f"- {f}" for f in failures) + "\n"

    print(markdown)
    if args.summary_out is not None:
        with args.summary_out.open("a", encoding="utf-8") as fh:
            fh.write(markdown)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
