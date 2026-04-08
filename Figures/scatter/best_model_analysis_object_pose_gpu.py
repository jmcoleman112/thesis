#!/usr/bin/env python
"""
GPU scatter plot for the shortlisted object and pose models, alongside their
matching baseline .pt checkpoints.

Run:
  python Figures/scatter/best_model_analysis_object_pose_gpu.py
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
GPU_COLUMN = "GPU Util %"
MAP_COLUMN = "mAP50-95"
FONT_FAMILY = "Times New Roman"
FIG_WIDTH_IN = 5.4
FIG_HEIGHT_IN = 2.35

BASE_POINT_COLOR = "#8d8d8d"
OBJECT_HIGHLIGHT_COLOR = "#d95f02"
POSE_HIGHLIGHT_COLOR = "#1b9e77"
BASE_POINT_SIZE = 14
HIGHLIGHT_POINT_SIZE = 34

OBJECT_MIN_MAP = 0.70
POSE_MIN_MAP = 0.97

OBJECT_HIGHLIGHT_MODELS = [
    "26s_DS3_p90_768_fp16.engine",
    "26s_DS3_768_fp16.engine",
    "11n_DS3_p90_fp16.engine",
    "11n_DS3_from_11l_fp16.engine",
    "26s_DS3_int8.engine",
    "26s_DS3_p90_960_fp16.engine",
    "26s_DS3_960_fp16.engine",
    "26s_DS3_from_26m_fp16.engine",
    "26s_DS3_fp16.engine",
]

POSE_HIGHLIGHT_MODELS = [
    "26n_pose_from_26l_640_fp16.engine",
    "26n_pose_from_26x_768_fp16.engine",
    "26n_pose_fp16.engine",
    "26n_pose_p90_640_fp16.engine",
]

OBJECT_ANNOTATION_OFFSETS = {
    "11n_DS3_p90_fp16.engine": (6, -2),
    "11n_DS3_from_11l_fp16.engine": (-37, -2),
    "26s_DS3_fp16.engine": (-19, 3),
    "26s_DS3_int8.engine": (6, -2),
    "26s_DS3_960_fp16.engine": (6, 0),
    "26s_DS3_768_fp16.engine": (6, -2),
    "26s_DS3_p90_960_fp16.engine": (-45, 0),
    "26s_DS3_p90_768_fp16.engine": (-45, -2),
    "26s_DS3_from_26m_fp16.engine": (6, -2),
}

POSE_ANNOTATION_OFFSETS = {
    "26n_pose_fp16.engine": (6, -4),
    "26n_pose_p90_640_fp16.engine": (6, -4),
    "26n_pose_from_26l_640_fp16.engine": (6, 2),
    "26n_pose_from_26x_768_fp16.engine": (6, 2),
}


def _normalize_location(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def _load_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required = ["Model", "Location", MAP_COLUMN, GPU_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def _is_object_ds3(model: object, location: object) -> bool:
    model_text = str(model).lower()
    location_text = _normalize_location(location)
    return "/object/" in location_text and ("ds3" in model_text or "/ds3" in location_text)


def _prepare_subset(df: pd.DataFrame, *, mask: pd.Series, min_map: float) -> pd.DataFrame:
    subset = df[mask].copy()
    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset[GPU_COLUMN] = pd.to_numeric(subset[GPU_COLUMN], errors="coerce")
    subset = subset.dropna(subset=[MAP_COLUMN, GPU_COLUMN]).copy()
    subset = subset[subset[MAP_COLUMN] >= min_map].copy()
    if subset.empty:
        raise ValueError("No rows remain after numeric conversion and mAP filtering.")
    return subset


def _format_label(model_name: str) -> str:
    model_text = str(model_name)
    lower_text = model_text.lower()

    label = model_text.replace(".engine", "").replace(".pt", "")
    label = label.replace("_DS3_", "_")
    label = label.replace("_pose_", "_")
    label = label.replace("_from_", "_<-_")
    parts = [part for part in label.split("_") if part]
    formatted_parts: list[str] = []
    for part in parts:
        lower_part = part.lower()
        if lower_part == "<-":
            formatted_parts.append("<-")
        elif lower_part == "fp16":
            formatted_parts.append("FP16")
        elif lower_part == "int8":
            formatted_parts.append("INT8")
        elif lower_part == "p90":
            formatted_parts.append("P90")
        else:
            formatted_parts.append(part)
    return " ".join(formatted_parts)


def _apply_annotations(ax: object, rows: pd.DataFrame, offsets: dict[str, tuple[int, int]], color: str) -> None:
    import matplotlib.patheffects as pe

    for _, row in rows.iterrows():
        model_name = str(row["Model"])
        dx, dy = offsets.get(model_name, (6, 6))
        annotation = ax.annotate(
            _format_label(model_name),
            xy=(row[GPU_COLUMN], row[MAP_COLUMN]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5,
            color=color,
            zorder=5,
        )
        annotation.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])


def _plot_panel(
    ax: object,
    usable: pd.DataFrame,
    highlight_models: list[str],
    *,
    title: str,
    highlight_color: str,
    offsets: dict[str, tuple[int, int]],
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    from matplotlib.ticker import PercentFormatter

    highlight = usable[usable["Model"].isin(highlight_models)].copy()

    missing_highlights = [model for model in highlight_models if model not in set(highlight["Model"])]
    if missing_highlights:
        print(f"Warning: missing highlighted models for {title}:")
        for model in missing_highlights:
            print(f"  - {model}")

    ax.scatter(
        usable[GPU_COLUMN],
        usable[MAP_COLUMN],
        alpha=0.65,
        s=BASE_POINT_SIZE,
        color=BASE_POINT_COLOR,
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        highlight[GPU_COLUMN],
        highlight[MAP_COLUMN],
        alpha=0.98,
        s=HIGHLIGHT_POINT_SIZE,
        color=highlight_color,
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
    )

    _apply_annotations(ax, highlight, offsets, highlight_color)

    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlabel("GPU Util %", fontsize=8 if show_xlabel else 0)
    ax.set_ylabel("mAP50-95 (%)" if show_ylabel else "", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, linestyle="--", alpha=0.4)


def _compute_axis_limits(values: pd.Series, *, pad_fraction: float = 0.05) -> tuple[float, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    pad = span * pad_fraction if span > 0 else max(abs(maximum) * pad_fraction, 0.01)
    lower = max(0.0, minimum - pad)
    upper = min(1.0, maximum + pad)
    return lower, upper


def _compute_percent_axis_limits(values: pd.Series, *, pad_fraction: float = 0.05, step: float = 0.01) -> tuple[float, float]:
    lower, upper = _compute_axis_limits(values, pad_fraction=pad_fraction)
    lower = math.floor(lower / step) * step
    upper = math.ceil(upper / step) * step
    if upper <= lower:
        upper = min(1.0, lower + step)
    return max(0.0, lower), min(1.0, upper)


def main() -> int:
    try:
        df = _load_csv()
        object_usable = _prepare_subset(
            df,
            mask=df.apply(lambda row: _is_object_ds3(row["Model"], row["Location"]), axis=1),
            min_map=OBJECT_MIN_MAP,
        )
        pose_usable = _prepare_subset(
            df,
            mask=df["Location"].apply(lambda value: "/pose/" in _normalize_location(value)),
            min_map=POSE_MIN_MAP,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    _plot_panel(
        axes[0],
        object_usable,
        OBJECT_HIGHLIGHT_MODELS,
        title="Object Models",
        highlight_color=OBJECT_HIGHLIGHT_COLOR,
        offsets=OBJECT_ANNOTATION_OFFSETS,
        show_ylabel=True,
        show_xlabel=False,
    )
    _plot_panel(
        axes[1],
        pose_usable,
        POSE_HIGHLIGHT_MODELS,
        title="Pose Models",
        highlight_color=POSE_HIGHLIGHT_COLOR,
        offsets=POSE_ANNOTATION_OFFSETS,
        show_ylabel=False,
        show_xlabel=False,
    )

    object_ymin, object_ymax = _compute_percent_axis_limits(object_usable[MAP_COLUMN])
    pose_ymin, pose_ymax = _compute_percent_axis_limits(pose_usable[MAP_COLUMN], pad_fraction=0.08, step=0.005)
    axes[0].set_ylim(object_ymin, object_ymax)
    axes[1].set_ylim(pose_ymin, pose_ymax)
    axes[1].tick_params(axis="y", labelleft=True)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BASE_POINT_COLOR,
            markersize=4.5,
            label="Other models",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OBJECT_HIGHLIGHT_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5.5,
            label="Object shortlist",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=POSE_HIGHLIGHT_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5.5,
            label="Pose shortlist",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    fig.supxlabel("GPU Util %", fontsize=8, y=0.10)
    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.94, wspace=0.16)
    plt.show()
    prompt_save_figure(fig, default_name="best_model_analysis_object_pose_gpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
