#!/usr/bin/env python
"""
Two-panel scatter plot of mAP vs latency:
  - left: object models (DS3 only)
  - right: pose models

Layout targets IEEE two-column width with one shared legend below both panels.

Run:
  python Figures/scatter_map_vs_latency_object_pose_two_col.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_COLS = ["Model", "Location", "Latency ms"]
MAP_COLUMN = "mAP50-95"
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"

IEEE_TWO_COL_WIDTH_IN = 7.16
FIG_HEIGHT_IN = 2.25
IEEE_SERIF_STACK = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]

OBJECT_MIN_MAP = 0.10
POSE_MIN_MAP = 0.76
POSE_MAX_LATENCY_MS = 300.0
POSE_SERIES_MODE = "both"
POSE_SERIES_FAMILIES = {
    "11": ("11n", "11s", "11m", "11l"),
    "26": ("26n", "26s", "26m", "26l"),
    "both": ("11n", "11s", "11m", "11l", "26n", "26s", "26m", "26l"),
}
EXCLUDED_POSE_FAMILIES = {"11l", "26m"}
OBJECT_PANEL_TITLE = "Object"
POSE_PANEL_TITLE = "Keypoint"

CATEGORY_ORDER = [
    ("baseline_pt", "Baseline", "#1f77b4"),
    ("baseline_engine", "Hardware Accel.", "#ff7f0e"),
    ("pruned_engine", "Pruned", "#2ca02c"),
    ("quant_engine_baseline", "Quantized", "#d62728"),
    ("quant_engine_pruned", "Quantized, Pruned", "#9467bd"),
    ("distilled", "Distillation", "#8c564b"),
    ("distilled_pruned", "Distilled, Pruned", "#17becf"),
    ("distilled_pruned_quantized", "Distilled, Pruned, Quantized", "#bcbd22"),
    ("distilled_quantized", "Distilled, Quantized", "#e377c2"),
    ("other", "Other", "#7f7f7f"),
]


def _normalize_location(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def _is_ds3(model: object, location: object) -> bool:
    m = str(model).lower()
    loc = _normalize_location(location)
    return "ds3" in m or "/ds3" in loc


def _extract_pose_family(model: object, location: object) -> str | None:
    m = str(model).strip().lower()
    loc = _normalize_location(location)

    model_match = re.search(r"^((?:11|26)[nmls])(?:_|$)", m)
    if model_match:
        return model_match.group(1)

    loc_match = re.search(r"/pose/((?:11|26)[nmls])-pose/", loc)
    if loc_match:
        return loc_match.group(1)

    return None


def _classify_row(model: object, location: object) -> str:
    loc = _normalize_location(location)
    m = str(model).lower()
    is_engine = ".engine" in m
    is_pt = m.endswith(".pt")

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


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS + [MAP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def _build_object_rows(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["Location"].apply(lambda v: "/object/" in _normalize_location(v))].copy()
    subset = subset[subset.apply(lambda r: _is_ds3(r["Model"], r["Location"]), axis=1)]
    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset["Latency ms"] = pd.to_numeric(subset["Latency ms"], errors="coerce")
    subset = subset.dropna(subset=[MAP_COLUMN, "Latency ms"]).copy()
    subset = subset[subset[MAP_COLUMN] >= OBJECT_MIN_MAP].copy()
    subset["category"] = subset.apply(lambda r: _classify_row(r["Model"], r["Location"]), axis=1)
    return subset


def _build_pose_rows(df: pd.DataFrame) -> pd.DataFrame:
    if POSE_SERIES_MODE not in POSE_SERIES_FAMILIES:
        raise ValueError(f"Invalid POSE_SERIES_MODE '{POSE_SERIES_MODE}'.")

    subset = df[df["Location"].apply(lambda v: "/pose/" in _normalize_location(v))].copy()
    subset["family"] = subset.apply(lambda r: _extract_pose_family(r["Model"], r["Location"]), axis=1)
    subset = subset[subset["family"].isin(POSE_SERIES_FAMILIES[POSE_SERIES_MODE])].copy()
    subset = subset[~subset["family"].isin(EXCLUDED_POSE_FAMILIES)].copy()
    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset["Latency ms"] = pd.to_numeric(subset["Latency ms"], errors="coerce")
    subset = subset.dropna(subset=[MAP_COLUMN, "Latency ms"]).copy()
    subset = subset[(subset[MAP_COLUMN] > POSE_MIN_MAP) & (subset["Latency ms"] <= POSE_MAX_LATENCY_MS)].copy()
    subset["category"] = subset.apply(lambda r: _classify_row(r["Model"], r["Location"]), axis=1)
    return subset


def _plot_panel(ax: object, panel_df: pd.DataFrame, *, marker_size: int, panel_label: str) -> list[str]:
    from matplotlib.ticker import FormatStrFormatter

    present_keys: list[str] = []
    for key, _, color in CATEGORY_ORDER:
        group = panel_df[panel_df["category"] == key]
        if group.empty:
            continue
        present_keys.append(key)
        ax.scatter(
            group["Latency ms"],
            group[MAP_COLUMN],
            alpha=0.8,
            s=marker_size,
            color=color,
        )

    ax.set_xlabel("Latency (ms)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(panel_label, fontsize=8, pad=2)
    return present_keys


def main() -> int:
    try:
        df = _load_csv(CSV_PATH)
        object_rows = _build_object_rows(df)
        pose_rows = _build_pose_rows(df)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if object_rows.empty and pose_rows.empty:
        print("Error: no usable object/pose rows for plotting.", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = IEEE_SERIF_STACK

    fig, (ax_object, ax_pose) = plt.subplots(1, 2, figsize=(IEEE_TWO_COL_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    present = set()
    if object_rows.empty:
        ax_object.text(0.5, 0.5, "No object rows", ha="center", va="center", fontsize=7)
        ax_object.set_xlabel("Latency ms", fontsize=7)
        ax_object.tick_params(axis="both", labelsize=6)
        ax_object.grid(True, linestyle="--", alpha=0.4)
        ax_object.set_title(OBJECT_PANEL_TITLE, fontsize=8, pad=2)
    else:
        present.update(_plot_panel(ax_object, object_rows, marker_size=14, panel_label=OBJECT_PANEL_TITLE))

    if pose_rows.empty:
        ax_pose.text(0.5, 0.5, "No pose rows", ha="center", va="center", fontsize=7)
        ax_pose.set_xlabel("Latency ms", fontsize=7)
        ax_pose.tick_params(axis="both", labelsize=6)
        ax_pose.grid(True, linestyle="--", alpha=0.4)
        ax_pose.set_title(POSE_PANEL_TITLE, fontsize=8, pad=2)
    else:
        present.update(_plot_panel(ax_pose, pose_rows, marker_size=12, panel_label=POSE_PANEL_TITLE))

    ax_object.set_ylabel(MAP_COLUMN, fontsize=7)
    ax_pose.set_ylabel("")

    label_by_key = {key: label for key, label, _ in CATEGORY_ORDER}
    color_by_key = {key: color for key, _, color in CATEGORY_ORDER}
    ordered_present = [key for key, _, _ in CATEGORY_ORDER if key in present]

    if ordered_present:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color_by_key[key],
                markersize=4.8,
                label=label_by_key[key],
            )
            for key in ordered_present
        ]
        fig.legend(
            handles=handles,
            frameon=False,
            fontsize=5.8,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(5, len(handles)),
            columnspacing=0.8,
            handletextpad=0.3,
            borderaxespad=0.0,
        )

    fig.tight_layout(pad=0.06)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.995, bottom=0.35, wspace=0.16)
    plt.show()
    prompt_save_figure(fig, default_name="object_pose_map_vs_latency_two_col")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
