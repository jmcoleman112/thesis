#!/usr/bin/env python
"""
Print all object and pose models faster than a given latency threshold.

Metric columns:
  - object: mAP50-95
  - pose:   Validation2 mAP50-95

Edit the config values below, then run:
  python Figures/analysis/models_under_latency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
MAX_LATENCY_MS = 50
TASK = "object"  # "object", "pose", or "both"
SAVE_CSV_PATH = None  # e.g. Path("research/models_under_20ms.csv")

LATENCY_COLUMN = "Latency ms"
TASK_METRIC_COLUMNS = {
    "object": "mAP50-95",
    "pose": "Validation2 mAP50-95",
}


def _norm(value: object) -> str:
    return str(value).replace("\\", "/").strip().lower()


def infer_task(location: object) -> str:
    loc = _norm(location)
    if "/object/" in loc:
        return "object"
    if "/pose/" in loc:
        return "pose"
    return "other"


def prepare_task_frame(df: pd.DataFrame, task: str, max_latency_ms: float) -> tuple[pd.DataFrame, str]:
    metric_column = TASK_METRIC_COLUMNS[task]
    task_df = df[df["task"] == task].copy()
    task_df[LATENCY_COLUMN] = pd.to_numeric(task_df[LATENCY_COLUMN], errors="coerce")
    task_df[metric_column] = pd.to_numeric(task_df[metric_column], errors="coerce")
    task_df = task_df.dropna(subset=[LATENCY_COLUMN]).copy()
    task_df = task_df[task_df[LATENCY_COLUMN] <= max_latency_ms].copy()
    task_df = task_df.sort_values(by=[LATENCY_COLUMN, metric_column], ascending=[True, False], na_position="last")
    return task_df, metric_column


def format_for_output(task_df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    display = task_df.copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    return display[["rank", "Model", metric_column, LATENCY_COLUMN, "Location"]]


def print_task_rows(task: str, source_rows: int, filtered_rows: pd.DataFrame, metric_column: str) -> None:
    title = f"{task.upper()} MODELS UNDER LATENCY THRESHOLD"
    print()
    print(title)
    print("-" * len(title))
    print(f"Metric: {metric_column}")
    print(f"Source rows: {source_rows}")
    print(f"Matching rows: {len(filtered_rows)}")
    if filtered_rows.empty:
        print("(no rows)")
        return
    print(format_for_output(filtered_rows, metric_column).to_string(index=False))


def main() -> int:
    if MAX_LATENCY_MS <= 0:
        print("Error: MAX_LATENCY_MS must be > 0.", file=sys.stderr)
        return 2
    if TASK not in {"object", "pose", "both"}:
        print("Error: TASK must be 'object', 'pose', or 'both'.", file=sys.stderr)
        return 2

    if not CSV_PATH.exists():
        print(f"Error: CSV not found: {CSV_PATH}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as exc:
        print(f"Error: failed to load CSV: {exc}", file=sys.stderr)
        return 1

    required_columns = ["Model", "Location", LATENCY_COLUMN, *TASK_METRIC_COLUMNS.values()]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        print(f"Error: CSV missing required columns: {missing}", file=sys.stderr)
        return 1

    df["task"] = df["Location"].apply(infer_task)
    requested_tasks = ["object", "pose"] if TASK == "both" else [TASK]
    saved_frames: list[pd.DataFrame] = []

    for task in requested_tasks:
        source_rows = len(df[df["task"] == task])
        filtered_rows, metric_column = prepare_task_frame(df, task, MAX_LATENCY_MS)
        print_task_rows(task, source_rows, filtered_rows, metric_column)

        if not filtered_rows.empty:
            save_frame = format_for_output(filtered_rows, metric_column).copy()
            save_frame.insert(0, "task", task)
            save_frame.insert(1, "metric_column", metric_column)
            saved_frames.append(save_frame)

    if SAVE_CSV_PATH is not None:
        combined = pd.concat(saved_frames, ignore_index=True) if saved_frames else pd.DataFrame()
        try:
            SAVE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(SAVE_CSV_PATH, index=False)
            print()
            print(f"Saved CSV: {SAVE_CSV_PATH}")
        except Exception as exc:
            print(f"Error: failed to save CSV: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
