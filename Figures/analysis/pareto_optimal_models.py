#!/usr/bin/env python
"""
List Pareto-optimal object and pose models from model_summaries.csv.

Pareto definition:
  - maximize detection/pose quality
  - minimize latency

Metric columns:
  - object: mAP50-95
  - pose:   Validation2 mAP50-95

Examples:
  python Figures/analysis/pareto_optimal_models.py
  python Figures/analysis/pareto_optimal_models.py --top-n 10
  python Figures/analysis/pareto_optimal_models.py --save-csv research/pareto_optimal_models.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CSV_DEFAULT = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
LATENCY_COLUMN = "Latency ms"
TASK_COLUMNS = {
    "object": "mAP50-95",
    "pose": "Validation2 mAP50-95",
}
BASE_COLUMNS = ["Model", "Location", LATENCY_COLUMN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Pareto analysis to model_summaries.csv for object and pose models."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_DEFAULT,
        help=f"CSV path (default: {CSV_DEFAULT})",
    )
    parser.add_argument(
        "--task",
        choices=["object", "pose", "both"],
        default="both",
        help="Task to evaluate (default: both).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optional limit after Pareto filtering. By default all Pareto-optimal rows are shown.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to save the combined Pareto-optimal rows as CSV.",
    )
    return parser.parse_args()


def _norm(value: object) -> str:
    return str(value).replace("\\", "/").strip().lower()


def infer_task(location: object) -> str:
    loc = _norm(location)
    if "/object/" in loc:
        return "object"
    if "/pose/" in loc:
        return "pose"
    return "other"


def pareto_frontier(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """
    Return non-dominated rows for:
      - maximizing metric_column
      - minimizing latency
    """
    ordered = df.sort_values(by=[LATENCY_COLUMN, metric_column], ascending=[True, False]).copy()
    best_metric_so_far = float("-inf")
    frontier_rows = []

    for _, row in ordered.iterrows():
        current_metric = float(row[metric_column])
        if current_metric > best_metric_so_far:
            frontier_rows.append(row)
            best_metric_so_far = current_metric

    if not frontier_rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(frontier_rows)


def prepare_task_frame(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str]:
    metric_column = TASK_COLUMNS[task]
    task_df = df[df["task"] == task].copy()
    task_df[metric_column] = pd.to_numeric(task_df[metric_column], errors="coerce")
    task_df[LATENCY_COLUMN] = pd.to_numeric(task_df[LATENCY_COLUMN], errors="coerce")
    task_df = task_df.dropna(subset=[metric_column, LATENCY_COLUMN]).copy()
    return task_df, metric_column


def format_for_output(frontier: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    display = frontier.sort_values(by=[metric_column, LATENCY_COLUMN], ascending=[False, True]).copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    ordered_columns = ["rank", "Model", metric_column, LATENCY_COLUMN, "Location"]
    return display[ordered_columns]


def print_frontier(
    task: str,
    source_rows: int,
    frontier_size: int,
    display_frontier: pd.DataFrame,
    metric_column: str,
) -> None:
    title = f"{task.upper()} PARETO-OPTIMAL MODELS"
    print()
    print(title)
    print("-" * len(title))
    print(f"Metric: {metric_column}")
    print(f"Source rows with numeric values: {source_rows}")
    print(f"Pareto frontier size: {frontier_size}")
    if display_frontier.empty:
        print("(no rows)")
        return
    print(display_frontier.to_string(index=False))


def main() -> int:
    args = parse_args()

    if args.top_n is not None and args.top_n < 1:
        print("Error: --top-n must be >= 1.", file=sys.stderr)
        return 2

    if not args.csv.exists():
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(args.csv)
    except Exception as exc:
        print(f"Error: failed to load CSV: {exc}", file=sys.stderr)
        return 1

    required_columns = sorted(set(BASE_COLUMNS + list(TASK_COLUMNS.values())))
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        print(f"Error: CSV missing required columns: {missing}", file=sys.stderr)
        return 1

    df["task"] = df["Location"].apply(infer_task)
    requested_tasks = ["object", "pose"] if args.task == "both" else [args.task]
    saved_frames: list[pd.DataFrame] = []

    for task in requested_tasks:
        task_df, metric_column = prepare_task_frame(df, task)
        frontier = pareto_frontier(task_df, metric_column)
        display_frontier = format_for_output(frontier, metric_column)
        if args.top_n is not None:
            display_frontier = display_frontier.head(args.top_n)
        print_frontier(task, len(task_df), len(frontier), display_frontier, metric_column)

        if not display_frontier.empty:
            save_frame = display_frontier.copy()
            save_frame.insert(0, "task", task)
            save_frame.insert(1, "metric_column", metric_column)
            saved_frames.append(save_frame)

    if args.save_csv is not None:
        combined = pd.concat(saved_frames, ignore_index=True) if saved_frames else pd.DataFrame()
        try:
            args.save_csv.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(args.save_csv, index=False)
            print()
            print(f"Saved CSV: {args.save_csv}")
        except Exception as exc:
            print(f"Error: failed to save CSV: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
