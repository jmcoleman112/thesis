#!/usr/bin/env python
"""
Three-panel hardware scatter grid for shortlisted object or pose models.

Layout:
  - Top row: GPU usage and power draw
  - Bottom row: temperature centered at the same width

Run:
  python Figures/scatter/best_model_analysis_hardware_grid.py --task object
  python Figures/scatter/best_model_analysis_hardware_grid.py --task pose
  python Figures/scatter/best_model_analysis_hardware_grid.py --task object --no-show --save-name best_model_analysis_object_hardware_grid
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure, sanitize_filename

CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "produced_images"

FONT_FAMILY = "Times New Roman"
FIG_WIDTH_IN = 6.8
FIG_HEIGHT_IN = 6.9

MAP_COLUMN = "mAP50-95"
BASE_POINT_COLOR = "#9a9a9a"
BASE_POINT_SIZE = 20
HIGHLIGHT_POINT_SIZE = 74

OBJECT_HIGHLIGHT_COLOR = "#d95f02"
POSE_HIGHLIGHT_COLOR = "#1b9e77"

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

PANELS = [
    {
        "key": "gpu",
        "column": "GPU Util %",
        "label": "GPU Util %",
        "title": "GPU Usage",
        "tick_fmt": "%.0f",
        "pad_fraction": 0.06,
        "min_pad": 2.0,
        "x_reference_value": 80.0,
        "x_reference_label": "Limit: 80%",
        "x_axis_max": 85.0,
    },
    {
        "key": "power",
        "column": "Power (W)",
        "label": "Power (W)",
        "title": "Power Draw",
        "tick_fmt": "%.1f",
        "pad_fraction": 0.08,
        "min_pad": 0.15,
        "x_reference_value": 15.0,
        "x_reference_label": "Limit: 15 W",
        "x_axis_max": 20.0,
    },
    {
        "key": "temp",
        "column": "Temp (?C)",
        "label": "Temperature (C)",
        "title": "Temperature",
        "tick_fmt": "%.0f",
        "pad_fraction": 0.08,
        "min_pad": 0.25,
        "x_reference_value": 75.0,
        "x_reference_label": "Limit: 75 C",
        "x_axis_max": 80.0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("object", "pose"), required=True, help="Which shortlist to plot.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to model_summaries.csv.")
    parser.add_argument("--save-name", help="Save directly to Figures/produced_images without prompting.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    return parser.parse_args()


def _normalize_location(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["Model", "Location", MAP_COLUMN] + [panel["column"] for panel in PANELS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def _is_object_ds3(model: object, location: object) -> bool:
    model_text = str(model).lower()
    location_text = _normalize_location(location)
    return "/object/" in location_text and ("ds3" in model_text or "/ds3" in location_text)


def _prepare_subset(df: pd.DataFrame, *, task: str) -> pd.DataFrame:
    if task == "object":
        mask = df.apply(lambda row: _is_object_ds3(row["Model"], row["Location"]), axis=1)
        min_map = OBJECT_MIN_MAP
    else:
        mask = df["Location"].apply(lambda value: "/pose/" in _normalize_location(value))
        min_map = POSE_MIN_MAP

    subset = df[mask].copy()
    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    for panel in PANELS:
        subset[panel["column"]] = pd.to_numeric(subset[panel["column"]], errors="coerce")

    subset = subset.dropna(subset=[MAP_COLUMN] + [panel["column"] for panel in PANELS]).copy()
    subset = subset[subset[MAP_COLUMN] >= min_map].copy()
    if subset.empty:
        raise ValueError("No rows remain after numeric conversion and mAP filtering.")
    return subset


def _compute_axis_limits(values: pd.Series, *, pad_fraction: float, min_pad: float) -> tuple[float, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    pad = max(span * pad_fraction, min_pad)
    if span == 0:
        pad = max(pad, max(abs(maximum) * pad_fraction, min_pad))
    return minimum - pad, maximum + pad


def _compute_map_limits(values: pd.Series, *, task: str) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    if task == "object":
        step = 0.01
        pad_fraction = 0.05
    else:
        step = 0.005
        pad_fraction = 0.08
    span = upper - lower
    pad = span * pad_fraction if span > 0 else step
    lower = max(0.0, math.floor((lower - pad) / step) * step)
    upper = min(1.0, math.ceil((upper + pad) / step) * step)
    if upper <= lower:
        upper = min(1.0, lower + step)
    return lower, upper


def _highlight_config(task: str) -> tuple[list[str], str, str]:
    if task == "object":
        return OBJECT_HIGHLIGHT_MODELS, OBJECT_HIGHLIGHT_COLOR, "Object shortlist"
    return POSE_HIGHLIGHT_MODELS, POSE_HIGHLIGHT_COLOR, "Pose shortlist"


def _draw_limit_guide(ax, panel: dict[str, object]) -> None:
    x_reference_value = panel.get("x_reference_value")
    x_reference_label = panel.get("x_reference_label")
    if x_reference_value is None:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = x_max - x_min
    y_span = y_max - y_min

    ax.axvline(
        float(x_reference_value),
        color="#d62728",
        linestyle=":",
        linewidth=1.25,
        zorder=2,
    )
    if x_reference_label:
        ax.text(
            float(x_reference_value) - x_span * 0.012,
            y_min + y_span * 0.02,
            str(x_reference_label),
            fontsize=8.6,
            color="#d62728",
            ha="right",
            va="bottom",
        )


def _plot_panel(ax: object, usable: pd.DataFrame, *, task: str, panel: dict[str, object], show_ylabel: bool) -> None:
    from matplotlib.ticker import FormatStrFormatter, PercentFormatter

    highlight_models, highlight_color, _ = _highlight_config(task)
    highlight = usable[usable["Model"].isin(highlight_models)].copy()

    missing_highlights = [model for model in highlight_models if model not in set(highlight["Model"])]
    if missing_highlights:
        print(f"Warning: missing highlighted models for {task} / {panel['key']}:")
        for model in missing_highlights:
            print(f"  - {model}")

    x_col = str(panel["column"])
    ax.scatter(
        usable[x_col],
        usable[MAP_COLUMN],
        alpha=0.65,
        s=BASE_POINT_SIZE,
        color=BASE_POINT_COLOR,
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        highlight[x_col],
        highlight[MAP_COLUMN],
        alpha=0.98,
        s=HIGHLIGHT_POINT_SIZE,
        color=highlight_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
    )

    x_min, x_max = _compute_axis_limits(
        usable[x_col],
        pad_fraction=float(panel["pad_fraction"]),
        min_pad=float(panel["min_pad"]),
    )
    x_axis_max = panel.get("x_axis_max")
    if x_axis_max is not None:
        x_max = max(x_max, float(x_axis_max))
    ax.set_xlim(x_min, x_max)

    y_min, y_max = _compute_map_limits(usable[MAP_COLUMN], task=task)
    ax.set_ylim(y_min, y_max)
    _draw_limit_guide(ax, panel)

    ax.set_title(str(panel["title"]), fontsize=10.8, pad=4)
    ax.set_xlabel(str(panel["label"]), fontsize=9.8)
    ax.set_ylabel("mAP50-95 (%)" if show_ylabel else "", fontsize=9.8)
    ax.tick_params(axis="both", labelsize=8.6)
    ax.xaxis.set_major_formatter(FormatStrFormatter(str(panel["tick_fmt"])))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, linestyle="--", alpha=0.4)


def build_figure(usable: pd.DataFrame, *, task: str):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)
    grid = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.02])

    ax_gpu = fig.add_subplot(grid[0, 0:2])
    ax_power = fig.add_subplot(grid[0, 2:4], sharey=ax_gpu)
    ax_temp = fig.add_subplot(grid[1, 1:3], sharey=ax_gpu)

    _plot_panel(ax_gpu, usable, task=task, panel=PANELS[0], show_ylabel=True)
    _plot_panel(ax_power, usable, task=task, panel=PANELS[1], show_ylabel=False)
    _plot_panel(ax_temp, usable, task=task, panel=PANELS[2], show_ylabel=True)

    ax_power.tick_params(axis="y", labelleft=False)

    _, highlight_color, highlight_label = _highlight_config(task)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BASE_POINT_COLOR,
            markersize=5.8,
            label="Other models",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=highlight_color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8.2,
            label=highlight_label,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=False,
        fontsize=7.8,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.09, right=0.985, top=0.97, bottom=0.14, hspace=0.34, wspace=0.42)
    return fig, (ax_gpu, ax_power, ax_temp)


def save_direct(fig, name: str) -> Path:
    cleaned = sanitize_filename(name)
    if not cleaned.lower().endswith(".png"):
        cleaned = f"{cleaned}.png"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / cleaned
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    return output_path


def main() -> int:
    args = parse_args()

    try:
        df = _load_csv(args.csv)
        usable = _prepare_subset(df, task=args.task)
        fig, _ = build_figure(usable, task=args.task)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Task:", args.task)
    print("Rows used:", len(usable))
    print("CSV:", args.csv)

    default_name = f"best_model_analysis_{args.task}_hardware_grid"
    if args.save_name:
        save_direct(fig, args.save_name)

    if not args.no_show:
        plt.show()
        if not args.save_name:
            prompt_save_figure(fig, default_name=default_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
