#!/usr/bin/env python
"""
Two-panel scatter plot of mAP vs latency for object models,
split between 11x and 26x families, using a shared legend.

Run:
  python Figures/scatter_map_vs_latency_object_split_11_26.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_COLS = ["Model", "Location", "Latency ms"]
MAP_COLUMN = "mAP50-95"
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FILTER_TOKEN = "/object/"

CATEGORY_ORDER = [
    ("baseline_pt", "Baseline", "#1f77b4"),
    ("baseline_engine", "Hardware Accel.", "#ff7f0e"),
    ("pruned_engine", "Pruned", "#2ca02c"),
    ("quant_engine_baseline", "Quantized", "#d62728"),
    ("quant_engine_pruned", "Quantized and Pruned", "#9467bd"),
    ("other", "Other", "#7f7f7f"),
]


def _normalize_location(value: str) -> str:
    return str(value).replace("\\", "/").lower()


def is_ds3(model: str, location: str) -> bool:
    m = str(model).lower()
    loc = _normalize_location(location)
    return "ds3" in m or "/ds3" in loc


def classify_row(model: str, location: str) -> str:
    loc = _normalize_location(location)
    m = str(model).lower()
    is_engine = ".engine" in m
    is_pt = m.endswith(".pt")

    if "/pruning_quantization/" in loc and is_engine:
        return "quant_engine_pruned"
    if "/quantization/" in loc and is_engine:
        return "quant_engine_baseline"
    if "/pruning/" in loc and is_engine:
        return "pruned_engine"
    if "/baseline/" in loc:
        if is_pt:
            return "baseline_pt"
        if is_engine:
            return "baseline_engine"
    return "other"


def model_family(model: str) -> str | None:
    m = str(model).strip()
    if m.startswith("11"):
        return "11"
    if m.startswith("26"):
        return "26"
    return None


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS + [MAP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def prepare_subset(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["Location"].apply(lambda v: FILTER_TOKEN in _normalize_location(v))].copy()
    subset = subset[subset.apply(lambda r: is_ds3(r["Model"], r["Location"]), axis=1)]
    if subset.empty:
        raise ValueError("No DS3 object models found (location contains /object/ and DS3).")

    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset["Latency ms"] = pd.to_numeric(subset["Latency ms"], errors="coerce")
    usable = subset.dropna(subset=[MAP_COLUMN, "Latency ms"]).copy()
    if usable.empty:
        raise ValueError("No rows with numeric mAP and Latency.")

    usable["category"] = usable.apply(lambda r: classify_row(r["Model"], r["Location"]), axis=1)
    usable["family"] = usable["Model"].apply(model_family)
    return usable


def main() -> int:
    try:
        df = load_csv(CSV_PATH)
        usable = prepare_subset(df)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    df_11 = usable[usable["family"] == "11"].copy()
    df_26 = usable[usable["family"] == "26"].copy()

    if df_11.empty and df_26.empty:
        print("Error: no 11x or 26x object models found.", file=sys.stderr)
        return 1
    if df_11.empty:
        print("Warning: no 11x object models found.")
    if df_26.empty:
        print("Warning: no 26x object models found.")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax_left, ax_right = axes

    for key, label, color in CATEGORY_ORDER:
        g11 = df_11[df_11["category"] == key]
        if not g11.empty:
            ax_left.scatter(g11["Latency ms"], g11[MAP_COLUMN], alpha=0.8, s=30, color=color)

        g26 = df_26[df_26["category"] == key]
        if not g26.empty:
            ax_right.scatter(g26["Latency ms"], g26[MAP_COLUMN], alpha=0.8, s=30, color=color)

    ax_left.set_title("11x Object Models")
    ax_right.set_title("26x Object Models")
    ax_left.set_xlabel("Latency ms")
    ax_right.set_xlabel("Latency ms")
    ax_left.set_ylabel(MAP_COLUMN)
    ax_left.grid(True, linestyle="--", alpha=0.4)
    ax_right.grid(True, linestyle="--", alpha=0.4)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=6, label=label)
        for _, label, color in CATEGORY_ORDER
    ]
    ax_right.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    fig.suptitle(f"{MAP_COLUMN} vs Latency (Object models)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    prompt_save_figure(fig, default_name="object_map_vs_latency_11_vs_26")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
