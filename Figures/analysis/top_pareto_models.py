#!/usr/bin/env python
"""
Report top Pareto-frontier models for pose and object tasks.

Pareto definition:
  - maximize mAP50-95
  - minimize Latency ms

Default behavior:
  - evaluate both tasks (pose + object)
  - return top 5 frontier models per task, ranked by highest mAP then lowest latency

Examples:
  python Figures/top_pareto_models.py
  python Figures/top_pareto_models.py --top-n 8 --task pose
  python Figures/top_pareto_models.py --task both --max-latency-ms 120
  python Figures/top_pareto_models.py --object-ds3-only --drop-pt
  python Figures/top_pareto_models.py --save-csv research/top_pareto_models.csv
  python Figures/top_pareto_models.py --task object --rank-mode weighted --norm-method winsorized
  python Figures/top_pareto_models.py --all-frontier --save-csv research/all_frontier_models.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

CSV_DEFAULT = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
REQUIRED_COLUMNS = ["Model", "Location", "mAP50-95", "Latency ms"]
HARDWARE_USAGE_COLUMNS = ["Power (W)", "Temp (?C)", "GPU Util %", "CPU Util %"]
HARDWARE_REDUCTION_COLUMNS = {
    "Power (W)": "Power Red. vs Baseline %",
    "Temp (?C)": "Temp Red. vs Baseline %",
    "GPU Util %": "GPU Red. vs Baseline %",
    "CPU Util %": "CPU Red. vs Baseline %",
}

DISPLAY_COLUMNS = [
    "rank",
    "Model",
    "mAP50-95",
    "Latency ms",
    "Location",
]


def _norm(value: object) -> str:
    return str(value).replace("\\", "/").strip().lower()


def infer_task(location: object) -> str:
    loc = _norm(location)
    if "/pose/" in loc:
        return "pose"
    if "/object/" in loc:
        return "object"
    return "other"


def is_ds3(model: object, location: object) -> bool:
    m = _norm(model)
    loc = _norm(location)
    return "ds3" in m or "/ds3" in loc


def is_pt_model(model: object) -> bool:
    return _norm(model).endswith(".pt")


def ensure_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def extract_family(model: object, location: object) -> str:
    m = _norm(model)
    loc = _norm(location)

    model_match = re.search(r"^(\d+[a-z])(?:_|$)", m)
    if model_match:
        return model_match.group(1)

    loc_match = re.search(r"/(?:object|pose)/(\d+[a-z])(?:_ds3|-pose)?/", loc)
    if loc_match:
        return loc_match.group(1)

    return "unknown"


def is_baseline_engine(model: object, location: object) -> bool:
    m = _norm(model)
    loc = _norm(location)
    return "baseline.engine" in m and "/baseline/" in loc


def _baseline_key(task: str, family: str, is_ds3_value: bool) -> tuple[str, str, bool]:
    return (task, family, is_ds3_value if task == "object" else False)


def build_baseline_lookup(task_reference_df: pd.DataFrame) -> dict[tuple[str, str, bool], pd.Series]:
    baseline_rows = task_reference_df[
        task_reference_df.apply(lambda r: is_baseline_engine(r["Model"], r["Location"]), axis=1)
    ].copy()

    if baseline_rows.empty:
        return {}

    baseline_rows["task"] = baseline_rows["Location"].apply(infer_task)
    baseline_rows["family"] = baseline_rows.apply(
        lambda r: extract_family(r["Model"], r["Location"]),
        axis=1,
    )
    baseline_rows["is_ds3"] = baseline_rows.apply(
        lambda r: is_ds3(r["Model"], r["Location"]),
        axis=1,
    )

    for col in ["mAP50-95", "Latency ms", *HARDWARE_USAGE_COLUMNS]:
        if col in baseline_rows.columns:
            baseline_rows[col] = pd.to_numeric(baseline_rows[col], errors="coerce")

    baseline_rows = baseline_rows.sort_values(
        by=["mAP50-95", "Latency ms"],
        ascending=[False, True],
    )

    lookup: dict[tuple[str, str, bool], pd.Series] = {}
    for _, row in baseline_rows.iterrows():
        key = _baseline_key(str(row["task"]), str(row["family"]), bool(row["is_ds3"]))
        if key not in lookup:
            lookup[key] = row
    return lookup


def enrich_with_hardware(top_models: pd.DataFrame, task_reference_df: pd.DataFrame) -> pd.DataFrame:
    if top_models.empty:
        return top_models

    usage_cols_present = [col for col in HARDWARE_USAGE_COLUMNS if col in top_models.columns]
    if not usage_cols_present:
        return top_models

    enriched = top_models.copy()
    baseline_lookup = build_baseline_lookup(task_reference_df)

    enriched["task"] = enriched["Location"].apply(infer_task)
    enriched["family"] = enriched.apply(lambda r: extract_family(r["Model"], r["Location"]), axis=1)
    enriched["is_ds3"] = enriched.apply(lambda r: is_ds3(r["Model"], r["Location"]), axis=1)
    enriched["Baseline Model"] = pd.NA

    for usage_col in usage_cols_present:
        enriched[usage_col] = pd.to_numeric(enriched[usage_col], errors="coerce")
        red_col = HARDWARE_REDUCTION_COLUMNS[usage_col]
        enriched[red_col] = pd.NA

    for idx, row in enriched.iterrows():
        task = str(row["task"])
        family = str(row["family"])
        ds3_value = bool(row["is_ds3"])
        key = _baseline_key(task, family, ds3_value)
        baseline = baseline_lookup.get(key)
        if baseline is None:
            continue

        enriched.at[idx, "Baseline Model"] = baseline["Model"]
        row_is_baseline = is_baseline_engine(row["Model"], row["Location"])

        for usage_col in usage_cols_present:
            model_val = pd.to_numeric(row[usage_col], errors="coerce")
            baseline_val = pd.to_numeric(baseline[usage_col], errors="coerce")
            red_col = HARDWARE_REDUCTION_COLUMNS[usage_col]

            if pd.isna(model_val) or pd.isna(baseline_val) or baseline_val == 0:
                continue

            if row_is_baseline:
                enriched.at[idx, red_col] = 0.0
            else:
                enriched.at[idx, red_col] = (baseline_val - model_val) / baseline_val * 100.0

    for usage_col in usage_cols_present:
        enriched[usage_col] = enriched[usage_col].round(2)
        red_col = HARDWARE_REDUCTION_COLUMNS[usage_col]
        enriched[red_col] = pd.to_numeric(enriched[red_col], errors="coerce").round(2)

    return enriched.drop(columns=["task", "family", "is_ds3"], errors="ignore")


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return non-dominated rows for maximizing mAP50-95 and minimizing Latency ms.
    """
    ordered = df.sort_values(by=["Latency ms", "mAP50-95"], ascending=[True, False]).copy()
    frontier_rows = []
    best_map_so_far = float("-inf")

    for _, row in ordered.iterrows():
        current_map = float(row["mAP50-95"])
        if current_map > best_map_so_far:
            frontier_rows.append(row)
            best_map_so_far = current_map

    if not frontier_rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(frontier_rows)


def _normalized_series(
    series: pd.Series,
    higher_is_better: bool,
    *,
    method: str,
    winsor_low: float,
    winsor_high: float,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if method == "rank":
        rank_pct = values.rank(method="average", pct=True)
        if higher_is_better:
            return rank_pct
        return 1.0 - rank_pct

    if method == "winsorized":
        low_v = float(values.quantile(winsor_low / 100.0))
        high_v = float(values.quantile(winsor_high / 100.0))
        if high_v == low_v:
            return pd.Series([1.0] * len(values), index=values.index)
        clipped = values.clip(lower=low_v, upper=high_v)
        if higher_is_better:
            return (clipped - low_v) / (high_v - low_v)
        return (high_v - clipped) / (high_v - low_v)

    min_v = float(values.min())
    max_v = float(values.max())
    if max_v == min_v:
        return pd.Series([1.0] * len(values), index=values.index)
    if higher_is_better:
        return (values - min_v) / (max_v - min_v)
    return (max_v - values) / (max_v - min_v)


def top_frontier_models(
    frontier: pd.DataFrame,
    top_n: int,
    *,
    rank_mode: str,
    map_weight: float,
    latency_weight: float,
    normalization_source: pd.DataFrame,
    norm_method: str,
    winsor_low: float,
    winsor_high: float,
) -> pd.DataFrame:
    if rank_mode == "weighted":
        ranked = frontier.copy()
        map_norm = _normalized_series(
            normalization_source["mAP50-95"],
            higher_is_better=True,
            method=norm_method,
            winsor_low=winsor_low,
            winsor_high=winsor_high,
        )
        lat_norm = _normalized_series(
            normalization_source["Latency ms"],
            higher_is_better=False,
            method=norm_method,
            winsor_low=winsor_low,
            winsor_high=winsor_high,
        )
        map_norm_by_index = map_norm.reindex(ranked.index)
        lat_norm_by_index = lat_norm.reindex(ranked.index)
        ranked["score"] = map_weight * map_norm_by_index + latency_weight * lat_norm_by_index
        ranked = ranked.sort_values(
            by=["score", "mAP50-95", "Latency ms"],
            ascending=[False, False, True],
        ).head(top_n)
    else:
        ranked = frontier.sort_values(by=["mAP50-95", "Latency ms"], ascending=[False, True]).head(top_n).copy()

    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def print_task_table(task: str, source_rows: int, frontier: pd.DataFrame, top_models: pd.DataFrame) -> None:
    title = f"{task.upper()} TOP {len(top_models)} FROM PARETO FRONTIER"
    print()
    print(title)
    print("-" * len(title))
    print(f"Source rows: {source_rows}")
    print(f"Pareto frontier size: {len(frontier)}")
    if top_models.empty:
        print("(no rows)")
        return
    display_cols = DISPLAY_COLUMNS.copy()
    if "Baseline Model" in top_models.columns:
        display_cols.append("Baseline Model")
    for col in HARDWARE_USAGE_COLUMNS:
        if col in top_models.columns:
            display_cols.append(col)
            display_cols.append(HARDWARE_REDUCTION_COLUMNS[col])
    if "score" in top_models.columns:
        display_cols.append("score")
    print(top_models[display_cols].to_string(index=False))


def print_frontier_table(task: str, source_rows: int, frontier_models: pd.DataFrame) -> None:
    title = f"{task.upper()} ALL {len(frontier_models)} PARETO FRONTIER MODELS"
    print()
    print(title)
    print("-" * len(title))
    print(f"Source rows: {source_rows}")
    print(f"Pareto frontier size: {len(frontier_models)}")
    if frontier_models.empty:
        print("(no rows)")
        return

    display_cols = DISPLAY_COLUMNS.copy()
    if "Baseline Model" in frontier_models.columns:
        display_cols.append("Baseline Model")
    for col in HARDWARE_USAGE_COLUMNS:
        if col in frontier_models.columns:
            display_cols.append(col)
            display_cols.append(HARDWARE_REDUCTION_COLUMNS[col])

    print(frontier_models[display_cols].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get top Pareto-frontier models for pose and object.")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT, help=f"CSV path (default: {CSV_DEFAULT})")
    parser.add_argument("--top-n", type=int, default=5, help="Top models to show per task (default: 5)")
    parser.add_argument(
        "--all-frontier",
        action="store_true",
        help="List all Pareto frontier models per task (ignores --top-n and ranking).",
    )
    parser.add_argument(
        "--task",
        choices=["pose", "object", "both"],
        default="both",
        help="Task to evaluate (default: both)",
    )
    parser.add_argument(
        "--object-ds3-only",
        action="store_true",
        help="Only keep DS3 rows for object task (matches object scatter filtering).",
    )
    parser.add_argument(
        "--drop-pt",
        action="store_true",
        help="Drop .pt rows before frontier calculation.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to save combined results as CSV.",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=120,
        help="Hard latency cap in milliseconds; drop rows with Latency ms above this value.",
    )
    parser.add_argument(
        "--rank-mode",
        choices=["lexicographic", "weighted"],
        default="lexicographic",
        help="How to rank models on the Pareto frontier (default: lexicographic).",
    )
    parser.add_argument(
        "--map-weight",
        type=float,
        default=0.5,
        help="mAP weight for --rank-mode weighted (default: 0.5).",
    )
    parser.add_argument(
        "--latency-weight",
        type=float,
        default=0.5,
        help="Latency weight for --rank-mode weighted (default: 0.5).",
    )
    parser.add_argument(
        "--norm-method",
        choices=["winsorized", "minmax", "rank"],
        default="winsorized",
        help="Normalization method for weighted mode (default: winsorized).",
    )
    parser.add_argument(
        "--winsor-low",
        type=float,
        default=5.0,
        help="Lower percentile for winsorized normalization (default: 5).",
    )
    parser.add_argument(
        "--winsor-high",
        type=float,
        default=95.0,
        help="Upper percentile for winsorized normalization (default: 95).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_n < 1 and not args.all_frontier:
        print("Error: --top-n must be >= 1", file=sys.stderr)
        return 2
    if args.map_weight < 0 or args.latency_weight < 0:
        print("Error: --map-weight and --latency-weight must be >= 0", file=sys.stderr)
        return 2
    if args.rank_mode == "weighted" and (args.map_weight + args.latency_weight) == 0:
        print("Error: weighted mode requires map-weight + latency-weight > 0", file=sys.stderr)
        return 2
    if args.max_latency_ms is not None and args.max_latency_ms <= 0:
        print("Error: --max-latency-ms must be > 0", file=sys.stderr)
        return 2
    if not (0 <= args.winsor_low < args.winsor_high <= 100):
        print("Error: winsor percentiles must satisfy 0 <= winsor-low < winsor-high <= 100", file=sys.stderr)
        return 2

    if not args.csv.exists():
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(args.csv)
        ensure_columns(df)
    except Exception as exc:
        print(f"Error: failed to load CSV: {exc}", file=sys.stderr)
        return 1

    df["mAP50-95"] = pd.to_numeric(df["mAP50-95"], errors="coerce")
    df["Latency ms"] = pd.to_numeric(df["Latency ms"], errors="coerce")
    usable = df.dropna(subset=["mAP50-95", "Latency ms"]).copy()
    dropped = len(df) - len(usable)
    if dropped:
        print(f"Warning: dropped {dropped} rows with non-numeric mAP50-95 or Latency ms.")

    if usable.empty:
        print("Error: no rows with numeric mAP50-95 and Latency ms.", file=sys.stderr)
        return 1

    usable["task"] = usable["Location"].apply(infer_task)

    requested_tasks = ["pose", "object"] if args.task == "both" else [args.task]
    output_rows = []

    for task in requested_tasks:
        task_reference_df = usable[usable["task"] == task].copy()
        if task == "object" and args.object_ds3_only:
            task_reference_df = task_reference_df[
                task_reference_df.apply(lambda r: is_ds3(r["Model"], r["Location"]), axis=1)
            ].copy()

        task_df = task_reference_df.copy()
        if args.drop_pt:
            task_df = task_df[~task_df["Model"].apply(is_pt_model)].copy()
        if args.max_latency_ms is not None:
            before_cap = len(task_df)
            task_df = task_df[task_df["Latency ms"] <= args.max_latency_ms].copy()
            dropped_cap = before_cap - len(task_df)
            if dropped_cap:
                print(f"Info: {task} dropped {dropped_cap} rows with Latency ms > {args.max_latency_ms:.2f}.")

        if task_df.empty:
            print_task_table(task, 0, task_df, task_df)
            continue

        frontier = pareto_frontier(task_df)
        if args.all_frontier:
            frontier_models = frontier.sort_values(by=["mAP50-95", "Latency ms"], ascending=[False, True]).copy()
            frontier_models.insert(0, "rank", range(1, len(frontier_models) + 1))
            frontier_models = enrich_with_hardware(frontier_models, task_reference_df=task_reference_df)
            print_frontier_table(task, len(task_df), frontier_models)
            selected_rows = frontier_models
        else:
            top_models = top_frontier_models(
                frontier,
                args.top_n,
                rank_mode=args.rank_mode,
                map_weight=args.map_weight,
                latency_weight=args.latency_weight,
                normalization_source=task_df,
                norm_method=args.norm_method,
                winsor_low=args.winsor_low,
                winsor_high=args.winsor_high,
            )
            top_models = enrich_with_hardware(top_models, task_reference_df=task_reference_df)
            print_task_table(task, len(task_df), frontier, top_models)
            selected_rows = top_models

        if not selected_rows.empty:
            with_task = selected_rows.copy()
            with_task["task"] = task
            ordered_cols = ["task"] + [c for c in with_task.columns if c != "task"]
            with_task = with_task[ordered_cols]
            output_rows.append(with_task)

    if args.save_csv is not None:
        if output_rows:
            combined = pd.concat(output_rows, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=["task"] + DISPLAY_COLUMNS)
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
