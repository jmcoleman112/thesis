#!/usr/bin/env python
"""
Aggregate summary report for model_summaries.csv.

Examples:
  python Figures/summarize_model_inventory.py
  python Figures/summarize_model_inventory.py --top-n 15
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

CSV_DEFAULT = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"

REQUIRED_COLUMNS = ["Model", "Location", "mAP50-95", "Latency ms"]
DISTILL_TOKENS = ("distill", "distillation", "student", "teacher", "kd")
ARTIFACT_ORDER = ["pt", "engine", "other"]
STAGE_ORDER = [
    "baseline",
    "distilled",
    "pruned",
    "quantized",
    "distilled_pruned",
    "distilled_quantized",
    "pruned_quantized",
    "distilled_pruned_quantized",
    "other",
]
TRANSFORM_STAGE_ORDER = [
    "distilled",
    "pruned",
    "quantized",
    "distilled_pruned",
    "distilled_quantized",
    "pruned_quantized",
    "distilled_pruned_quantized",
]

DISPLAY_COLS = [
    "Model",
    "task",
    "family",
    "artifact",
    "stage",
    "mAP50-95",
    "Latency ms",
    "FPS (avg)",
]


def _norm(text: object) -> str:
    return str(text).replace("\\", "/").strip().lower()


def infer_task(location: object) -> str:
    loc = _norm(location)
    if "/object/" in loc:
        return "object"
    if "/pose/" in loc:
        return "pose"
    return "other"


def infer_family(model: object, location: object) -> str:
    m = _norm(model)
    loc = _norm(location)

    model_match = re.search(r"^(\d+[a-z])(?:_|$)", m)
    if model_match:
        return model_match.group(1)

    loc_match = re.search(r"/(?:object|pose)/(\d+[a-z])(?:_ds3)?/", loc)
    if loc_match:
        return loc_match.group(1)

    return "unknown"


def infer_series(family: object) -> str:
    f = _norm(family)
    m = re.match(r"^(\d+)", f)
    return m.group(1) if m else "unknown"


def infer_artifact(model: object) -> str:
    m = _norm(model)
    if ".engine" in m:
        return "engine"
    if m.endswith(".pt"):
        return "pt"
    return "other"


def infer_quant_mode(model: object, location: object) -> str:
    m = _norm(model)
    loc = _norm(location)
    if "int8" in m or "/int8/" in loc:
        return "int8"
    if "fp16" in m or "/fp16/" in loc:
        return "fp16"
    return "none"


def infer_pruning_ratio(model: object, location: object) -> str:
    m = _norm(model)
    loc = _norm(location)

    loc_match = re.search(r"/pruning(?:_quantization)?/(\d+)(?:/|$)", loc)
    if loc_match:
        return loc_match.group(1)

    model_match = re.search(r"_p(\d+)(?:_|\\.|$)", m)
    if model_match:
        return model_match.group(1)

    return "-"


def infer_stage(model: object, location: object) -> str:
    m = _norm(model)
    loc = _norm(location)
    combined = f"{m} {loc}"

    has_distill = any(token in combined for token in DISTILL_TOKENS)
    has_pruning = "pruning" in combined or bool(re.search(r"_p\d+", m))
    has_quant = "quantization" in combined or "fp16" in combined or "int8" in combined
    has_baseline = "/baseline/" in loc or "baseline" in m

    transform_stage = {
        (True, False, False): "distilled",
        (False, True, False): "pruned",
        (False, False, True): "quantized",
        (True, True, False): "distilled_pruned",
        (True, False, True): "distilled_quantized",
        (False, True, True): "pruned_quantized",
        (True, True, True): "distilled_pruned_quantized",
    }
    stage = transform_stage.get((has_distill, has_pruning, has_quant))
    if stage:
        return stage
    if has_baseline:
        return "baseline"
    return "other"


def with_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["task"] = enriched["Location"].apply(infer_task)
    enriched["family"] = enriched.apply(lambda r: infer_family(r["Model"], r["Location"]), axis=1)
    enriched["series"] = enriched["family"].apply(infer_series)
    enriched["artifact"] = enriched["Model"].apply(infer_artifact)
    enriched["quant_mode"] = enriched.apply(lambda r: infer_quant_mode(r["Model"], r["Location"]), axis=1)
    enriched["pruning_ratio"] = enriched.apply(lambda r: infer_pruning_ratio(r["Model"], r["Location"]), axis=1)
    enriched["stage"] = enriched.apply(lambda r: infer_stage(r["Model"], r["Location"]), axis=1)
    enriched["is_ds3"] = enriched.apply(
        lambda r: "ds3" in _norm(r["Model"]) or "/ds3" in _norm(r["Location"]),
        axis=1,
    )
    return enriched


def apply_filter_policy(df: pd.DataFrame) -> pd.DataFrame:
    # Match analysis policy:
    # - drop object non-DS3 11* rows
    # - keep object DS3 11* rows, object 26* rows, and all pose rows
    out = df.copy()
    mask_drop_object_11_non_ds3 = (
        (out["task"] == "object")
        & (out["family"].astype(str).str.startswith("11"))
        & (~out["is_ds3"])
    )
    return out[~mask_drop_object_11_non_ds3].copy()


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    return df.to_string(index=False)


def top_n_table(df: pd.DataFrame, by: str, n: int, ascending: bool) -> pd.DataFrame:
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    return df.sort_values(by=by, ascending=ascending).head(n)[cols]


def pick_best_map(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values(by=["mAP50-95", "Latency ms"], ascending=[False, True])
    return ordered.iloc[0]


def pick_fastest(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values(by=["Latency ms", "mAP50-95"], ascending=[True, False])
    return ordered.iloc[0]


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maximize mAP50-95 while minimizing Latency ms.
    """
    ordered = df.sort_values(by=["Latency ms", "mAP50-95"], ascending=[True, False]).copy()
    frontier_rows = []
    best_map_so_far = float("-inf")
    for _, row in ordered.iterrows():
        current_map = float(row["mAP50-95"])
        if current_map > best_map_so_far:
            frontier_rows.append(row)
            best_map_so_far = current_map
    return pd.DataFrame(frontier_rows)


def summarize(df: pd.DataFrame, top_n: int) -> None:
    print_section("Overview")
    overview = pd.DataFrame(
        [
            ("Total rows", len(df)),
            ("Unique model names", df["Model"].nunique()),
            ("Unique locations", df["Location"].nunique()),
            ("DS3 rows", int(df["is_ds3"].sum())),
            ("Non-DS3 rows", int((~df["is_ds3"]).sum())),
            ("Object rows", int((df["task"] == "object").sum())),
            ("Pose rows", int((df["task"] == "pose").sum())),
        ],
        columns=["Metric", "Value"],
    )
    print(format_table(overview))

    print_section("Counts by Task")
    by_task = df.groupby("task").size().reset_index(name="count").sort_values("count", ascending=False)
    print(format_table(by_task))

    print_section("Counts by Family")
    by_family = df.groupby("family").size().reset_index(name="count").sort_values("count", ascending=False)
    print(format_table(by_family))

    print_section("Counts by Stage")
    by_stage = (
        pd.DataFrame({"stage": STAGE_ORDER})
        .assign(count=lambda t: t["stage"].map(df.groupby("stage").size()).fillna(0).astype(int))
    )
    by_stage = by_stage[by_stage["count"] > 0].copy()
    by_stage = by_stage.sort_values("count", ascending=False)
    print(format_table(by_stage))

    print_section("Transform Category Coverage (7 total, excludes baseline)")
    transform_counts = (
        pd.DataFrame({"stage": TRANSFORM_STAGE_ORDER})
        .assign(count=lambda t: t["stage"].map(df.groupby("stage").size()).fillna(0).astype(int))
    )
    print(format_table(transform_counts))

    print_section("Counts by Stage + Task (e.g., quantized pose)")
    by_stage_task = (
        df.groupby(["stage", "task"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "stage", "task"], ascending=[False, True, True])
    )
    by_stage_task["bucket"] = by_stage_task["stage"] + " " + by_stage_task["task"]
    print(format_table(by_stage_task[["bucket", "count"]]))

    print_section("Counts by Artifact (.pt / .engine)")
    by_artifact = (
        pd.DataFrame({"artifact": ARTIFACT_ORDER})
        .assign(count=lambda t: t["artifact"].map(df.groupby("artifact").size()).fillna(0).astype(int))
    )
    by_artifact = by_artifact[by_artifact["count"] > 0].copy()
    by_artifact = by_artifact.sort_values("count", ascending=False)
    print(format_table(by_artifact))

    print_section("Artifact Split by Stage (pt / engine / other)")
    by_stage_artifact = (
        df.groupby(["stage", "artifact"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=STAGE_ORDER, fill_value=0)
    )
    for artifact in ARTIFACT_ORDER:
        if artifact not in by_stage_artifact.columns:
            by_stage_artifact[artifact] = 0
    by_stage_artifact = by_stage_artifact[ARTIFACT_ORDER]
    by_stage_artifact["total"] = by_stage_artifact.sum(axis=1)
    by_stage_artifact = by_stage_artifact[by_stage_artifact["total"] > 0]
    by_stage_artifact = by_stage_artifact.reset_index().rename(columns={"index": "stage"})
    print(format_table(by_stage_artifact))

    print_section("Artifact Split by Stage + Task (pt / engine / other)")
    by_stage_task_artifact = (
        df.groupby(["stage", "task", "artifact"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for artifact in ARTIFACT_ORDER:
        if artifact not in by_stage_task_artifact.columns:
            by_stage_task_artifact[artifact] = 0
    by_stage_task_artifact["total"] = by_stage_task_artifact[ARTIFACT_ORDER].sum(axis=1)
    by_stage_rank = {name: idx for idx, name in enumerate(STAGE_ORDER)}
    task_order = {"object": 0, "pose": 1, "other": 2}
    by_stage_task_artifact["stage_rank"] = by_stage_task_artifact["stage"].map(by_stage_rank).fillna(999)
    by_stage_task_artifact["task_rank"] = by_stage_task_artifact["task"].map(task_order).fillna(999)
    by_stage_task_artifact = by_stage_task_artifact.sort_values(
        ["stage_rank", "task_rank", "task"], ascending=[True, True, True]
    )
    by_stage_task_artifact = by_stage_task_artifact[
        ["stage", "task", *ARTIFACT_ORDER, "total"]
    ]
    print(format_table(by_stage_task_artifact))

    print_section("Counts by Quantization Mode")
    by_quant = df.groupby("quant_mode").size().reset_index(name="count").sort_values("count", ascending=False)
    print(format_table(by_quant))

    print_section("Counts by Pruning Ratio")
    by_prune = (
        df[df["pruning_ratio"] != "-"]
        .groupby("pruning_ratio")
        .size()
        .reset_index(name="count")
        .sort_values("pruning_ratio")
    )
    print(format_table(by_prune))

    print_section("Top Model Name Frequencies")
    by_name = df.groupby("Model").size().reset_index(name="count").sort_values("count", ascending=False).head(top_n)
    print(format_table(by_name))

    print_section(f"Top {top_n} by mAP50-95")
    print(format_table(top_n_table(df, by="mAP50-95", n=top_n, ascending=False)))

    print_section(f"Top {top_n} Fastest (Lowest Latency)")
    print(format_table(top_n_table(df, by="Latency ms", n=top_n, ascending=True)))

    print_section("Best Overall / Fastest Overall")
    best_map = pick_best_map(df)
    fastest = pick_fastest(df)
    best_summary = pd.DataFrame(
        [
            (
                "Best mAP50-95",
                best_map["Model"],
                best_map["task"],
                best_map["family"],
                best_map["stage"],
                f"{best_map['mAP50-95']:.6f}",
                f"{best_map['Latency ms']:.2f}",
            ),
            (
                "Fastest",
                fastest["Model"],
                fastest["task"],
                fastest["family"],
                fastest["stage"],
                f"{fastest['mAP50-95']:.6f}",
                f"{fastest['Latency ms']:.2f}",
            ),
        ],
        columns=["Category", "Model", "Task", "Family", "Stage", "mAP50-95", "Latency ms"],
    )
    print(format_table(best_summary))

    print_section("Best Model per Family")
    by_family_rows = []
    for family, family_df in df.groupby("family"):
        winner = pick_best_map(family_df)
        by_family_rows.append(
            (
                family,
                winner["Model"],
                winner["stage"],
                float(winner["mAP50-95"]),
                float(winner["Latency ms"]),
            )
        )
    best_per_family = pd.DataFrame(
        by_family_rows, columns=["family", "model", "stage", "mAP50-95", "Latency ms"]
    ).sort_values("family")
    print(format_table(best_per_family))

    print_section("Most Accurate and Fastest per Task")
    per_task_rows = []
    preferred_task_order = ["object", "pose"]
    remaining_tasks = sorted([t for t in df["task"].unique() if t not in preferred_task_order])
    ordered_tasks = [t for t in preferred_task_order if t in set(df["task"])] + remaining_tasks

    for task in ordered_tasks:
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue

        most_accurate = pick_best_map(task_df)
        fastest = pick_fastest(task_df)

        per_task_rows.append(
            (
                task,
                "most_accurate",
                most_accurate["Model"],
                most_accurate["family"],
                most_accurate["stage"],
                float(most_accurate["mAP50-95"]),
                float(most_accurate["Latency ms"]),
            )
        )
        per_task_rows.append(
            (
                task,
                "fastest",
                fastest["Model"],
                fastest["family"],
                fastest["stage"],
                float(fastest["mAP50-95"]),
                float(fastest["Latency ms"]),
            )
        )

    per_task_summary = pd.DataFrame(
        per_task_rows,
        columns=["task", "category", "model", "family", "stage", "mAP50-95", "Latency ms"],
    )
    print(format_table(per_task_summary))

    print_section("Pareto Frontier (Max mAP50-95, Min Latency)")
    frontier = pareto_frontier(df)
    frontier_cols = [c for c in DISPLAY_COLS if c in frontier.columns]
    print(format_table(frontier[frontier_cols].sort_values("Latency ms")))
    print(f"\nPareto frontier size: {len(frontier)} models")


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize model inventory from model_summaries.csv")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT, help=f"CSV path (default: {CSV_DEFAULT})")
    parser.add_argument("--top-n", type=int, default=10, help="Rows to show in top lists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_n < 1:
        print("Error: --top-n must be >= 1", file=sys.stderr)
        return 2

    if not args.csv.exists():
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(args.csv)
        ensure_columns(df, REQUIRED_COLUMNS)
    except Exception as exc:
        print(f"Error: failed to load CSV: {exc}", file=sys.stderr)
        return 1

    # Ensure numeric metrics are numeric before ranking.
    df["mAP50-95"] = pd.to_numeric(df["mAP50-95"], errors="coerce")
    df["Latency ms"] = pd.to_numeric(df["Latency ms"], errors="coerce")
    if "FPS (avg)" in df.columns:
        df["FPS (avg)"] = pd.to_numeric(df["FPS (avg)"], errors="coerce")

    usable = df.dropna(subset=["mAP50-95", "Latency ms"]).copy()
    dropped = len(df) - len(usable)
    if dropped:
        print(f"Warning: dropped {dropped} rows with non-numeric mAP50-95 or Latency ms.")

    if usable.empty:
        print("Error: no rows with numeric mAP50-95 and Latency ms.", file=sys.stderr)
        return 1

    enriched = with_derived_columns(usable)
    filtered = apply_filter_policy(enriched)
    dropped_by_policy = len(enriched) - len(filtered)
    if dropped_by_policy > 0:
        print(
            "Filter policy: dropped object non-DS3 11* rows "
            f"(dropped {dropped_by_policy})."
        )
    summarize(filtered, top_n=args.top_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
