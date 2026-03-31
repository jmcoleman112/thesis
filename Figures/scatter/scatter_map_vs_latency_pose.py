#!/usr/bin/env python
"""
Scatter plot of mAP vs latency for pose models only.

Run:
  python Figures/scatter_map_vs_latency_pose.py

Set SERIES_MODE in this file to one of: "11", "26", "both".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_COLS = ["Model", "Location", "Latency ms"]
MAP_COLUMN = "mAP50-95"
MIN_MAP = 0.76
MAX_LATENCY_MS = 300.0
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FILTER_TOKEN = "/pose/"
SERIES_FAMILIES = {
    "11": ("11n", "11s", "11m", "11l"),
    "26": ("26n", "26s", "26m", "26l"),
    "both": ("11n", "11s", "11m", "11l", "26n", "26s", "26m", "26l"),
}
SERIES_MODE = "both"
IEEE_ONE_COL_WIDTH_IN = 3.5
IEEE_ONE_COL_HEIGHT_IN = 2.5
TEMP_LABEL_ORIGINAL_BASELINE = True
EXCLUDED_POSE_FAMILIES = {"11l", "26m"}

CATEGORY_ORDER = [
    ("baseline_pt", "Baseline", "#1f77b4"),
    ("baseline_engine", "Hardware Accel.", "#ff7f0e"),
    ("pruned_engine", "Pruned", "#2ca02c"),
    ("quant_engine_baseline", "Quantized", "#d62728"),
    ("quant_engine_pruned", "Quantized + Pruned", "#9467bd"),
    ("distilled", "Distillation", "#8c564b"),
    ("distilled_pruned", "Distillation + Pruned", "#17becf"),
    ("distilled_pruned_quantized", "Distillation + Pruned + Quantized", "#bcbd22"),
    ("distilled_quantized", "Distillation + Quantized", "#e377c2"),
    ("other", "Other", "#7f7f7f"),
]


def _normalize_location(value: str) -> str:
    return str(value).replace("\\", "/").lower()


def extract_family(model: str, location: str) -> str | None:
    m = str(model).strip().lower()
    loc = _normalize_location(location)

    model_match = re.search(r"^((?:11|26)[nmls])(?:_|$)", m)
    if model_match:
        return model_match.group(1)

    loc_match = re.search(r"/pose/((?:11|26)[nmls])-pose/", loc)
    if loc_match:
        return loc_match.group(1)

    return None


def classify_row(model: str, location: str) -> str:
    loc = _normalize_location(location)
    m = str(model).lower()
    is_engine = ".engine" in m
    is_pt = m.endswith(".pt")

    # Treat transformed .pt checkpoints the same as transformed engine artifacts.
    if "/distillation_pruning_quantization/" in loc and (is_engine or is_pt):
        return "distilled_pruned_quantized"
    if "/distillation_pruning/" in loc and (is_engine or is_pt):
        return "distilled_pruned"
    if "/distilled-quantized/" in loc and (is_engine or is_pt):
        return "distilled_quantized"
    if "/distillation/" in loc and (is_engine or is_pt):
        return "distilled"
    if "/pruning_quantization/" in loc and (is_engine or is_pt):
        return "quant_engine_pruned"
    if "/quantization/" in loc and (is_engine or is_pt):
        return "quant_engine_baseline"
    if "/pruning/" in loc and (is_engine or is_pt):
        return "pruned_engine"
    if "/baseline/" in loc:
        if is_pt:
            return "baseline_pt"
        if is_engine:
            return "baseline_engine"
    return "other"


def original_baseline_label(model: str, location: str) -> str:
    m = str(model).lower()
    source_match = re.search(r"_from_((?:11|26)[nmls])(?:_|\\.|$)", m)
    if source_match:
        return source_match.group(1)

    family = extract_family(model, location)
    return family if family is not None else "?"


def is_pruned_row(model: str, location: str) -> bool:
    loc = _normalize_location(location)
    m = str(model).lower()
    return "/pruning" in loc or bool(re.search(r"_p\d+(?:_|\\.|$)", m))


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS + [MAP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def main() -> int:
    if SERIES_MODE not in SERIES_FAMILIES:
        print(f"Error: invalid SERIES_MODE '{SERIES_MODE}'. Use one of: {', '.join(SERIES_FAMILIES)}.", file=sys.stderr)
        return 1
    family_allowlist = SERIES_FAMILIES[SERIES_MODE]

    try:
        df = load_csv(CSV_PATH)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    subset = df[df["Location"].apply(lambda v: FILTER_TOKEN in _normalize_location(v))].copy()
    subset["family"] = subset.apply(lambda r: extract_family(r["Model"], r["Location"]), axis=1)
    subset = subset[subset["family"].isin(family_allowlist)].copy()
    subset = subset[~subset["family"].isin(EXCLUDED_POSE_FAMILIES)].copy()
    if subset.empty:
        print(f"Error: no rows found for pose models in series '{SERIES_MODE}'.", file=sys.stderr)
        return 1

    non_pruned = subset.copy()
    non_pruned[MAP_COLUMN] = pd.to_numeric(non_pruned[MAP_COLUMN], errors="coerce")
    non_pruned["Latency ms"] = pd.to_numeric(non_pruned["Latency ms"], errors="coerce")
    numeric = non_pruned.dropna(subset=[MAP_COLUMN, "Latency ms"]).copy()
    usable = numeric[(numeric[MAP_COLUMN] > MIN_MAP) & (numeric["Latency ms"] <= MAX_LATENCY_MS)].copy()

    dropped_non_numeric = len(non_pruned) - len(numeric)
    if dropped_non_numeric:
        print(f"Warning: dropped {dropped_non_numeric} rows with non-numeric {MAP_COLUMN} or Latency ms.")

    dropped_low_map = len(numeric) - len(usable)
    if dropped_low_map:
        print(
            f"Warning: dropped {dropped_low_map} rows with {MAP_COLUMN} <= {MIN_MAP:.2f} "
            f"or Latency ms > {MAX_LATENCY_MS:.0f}."
        )

    if usable.empty:
        print(
            f"Error: no pose rows with numeric mAP/Latency, {MAP_COLUMN} > {MIN_MAP:.2f}, "
            f"and Latency ms <= {MAX_LATENCY_MS:.0f}.",
            file=sys.stderr,
        )
        return 1

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FormatStrFormatter
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    usable["category"] = usable.apply(lambda r: classify_row(r["Model"], r["Location"]), axis=1)

    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDTH_IN, IEEE_ONE_COL_HEIGHT_IN), dpi=300)
    for key, _, color in CATEGORY_ORDER:
        group = usable[usable["category"] == key]
        if group.empty:
            continue
        ax.scatter(
            group["Latency ms"],
            group[MAP_COLUMN],
            alpha=0.8,
            s=12,
            color=color,
        )

    ax.set_xlabel("Latency ms", fontsize=7)
    ax.set_ylabel(MAP_COLUMN, fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    if TEMP_LABEL_ORIGINAL_BASELINE:
        for _, row in usable.iterrows():
            ax.annotate(
                original_baseline_label(row["Model"], row["Location"]),
                (float(row["Latency ms"]), float(row[MAP_COLUMN])),
                textcoords="offset points",
                xytext=(2, 2),
                fontsize=4.6,
                alpha=0.9,
            )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=4, label=label)
        for _, label, color in CATEGORY_ORDER
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=5,
        loc="lower right",
        borderaxespad=0.2,
        handletextpad=0.3,
    )

    fig.tight_layout(pad=0.2)
    plt.show()
    save_suffix = "11_26" if SERIES_MODE == "both" else SERIES_MODE
    prompt_save_figure(fig, default_name=f"pose_{save_suffix}_map_vs_latency")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
