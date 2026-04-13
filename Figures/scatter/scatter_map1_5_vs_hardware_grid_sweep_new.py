#!/usr/bin/env python
"""
Sweep JSON scatter plot grid: Pass mAP1-5 vs GPU usage, power draw, and temperature.

Layout:
  - Top row: GPU usage and power draw
  - Bottom row: temperature spanning the full width

Run:
  python Figures/scatter/scatter_map1_5_vs_hardware_grid_sweep_new.py
  python Figures/scatter/scatter_map1_5_vs_hardware_grid_sweep_new.py --no-show --save-name sweep_new_map1_5_hardware_grid
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from scatter_map1_5_sweep_new_utils import (
    FONT_FAMILY,
    JSON_PATH,
    OUT_DIR,
    X_AXIS_MARGIN_FRAC,
    X_REFERENCE_LINE_COLOR,
    _apply_axis_limits,
    _build_legend_handles,
    load_rows,
)
from figure_save_dialog import prompt_save_figure, sanitize_filename

FIG_WIDTH_IN = 6.9
FIG_HEIGHT_IN = 7.0
POINT_SIZE = 46

PANELS = [
    {
        "key": "gpu",
        "x_col": "GPU (%)",
        "x_label": "GPU (%)",
        "title": "GPU Usage",
        "x_tick_fmt": "%.1f",
        "x_min_pad": 1.0,
        "x_reference_value": 80.0,
        "x_reference_label": "Limit: 80%",
        "x_axis_max": 85.0,
    },
    {
        "key": "power",
        "x_col": "PS Power (W)",
        "x_label": "Power (W)",
        "title": "Power Draw",
        "x_tick_fmt": "%.1f",
        "x_min_pad": 0.2,
        "x_reference_value": 20.0,
        "x_reference_label": "Limit: 20 W",
        "x_axis_max": 25.0,
    },
    {
        "key": "temp",
        "x_col": "Temp (C)",
        "x_label": "Temperature (C)",
        "title": "Temperature",
        "x_tick_fmt": "%.1f",
        "x_min_pad": 0.2,
        "x_reference_value": 68.0,
        "x_reference_label": "Limit: 68 C",
        "x_axis_max": 80.0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=JSON_PATH,
        help="Path to the sweep JSON file.",
    )
    parser.add_argument(
        "--save-name",
        help="Save directly to Figures/produced_images without prompting.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the matplotlib window.",
    )
    return parser.parse_args()


def _draw_reference_guides(ax, panel: dict[str, object], x_span: float, y_span: float) -> None:
    x_reference_value = panel.get("x_reference_value")
    x_reference_label = panel.get("x_reference_label")

    if x_reference_value is not None:
        ax.axvline(
            float(x_reference_value),
            color=X_REFERENCE_LINE_COLOR,
            linestyle=":",
            linewidth=1.15,
            zorder=1,
        )

    y_min, _ = ax.get_ylim()
    if x_reference_value is not None and x_reference_label:
        ax.text(
            float(x_reference_value) - x_span * 0.012,
            y_min + y_span * 0.02,
            str(x_reference_label),
            fontsize=9.0,
            color=X_REFERENCE_LINE_COLOR,
            ha="right",
            va="bottom",
        )


def _compute_y_limits(rows: list[dict[str, object]]) -> tuple[float, float]:
    values = [float(row["mAP1-5_pct"]) for row in rows]
    value_min = min(values)
    value_max = max(values)
    pad = max((value_max - value_min) * 0.14, 2.5)
    lower = math.floor((value_min - pad) / 2.0) * 2.0
    upper = math.ceil((value_max + pad) / 2.0) * 2.0
    if upper <= lower:
        upper = lower + 2.0
    return lower, upper


def _plot_panel(
    ax,
    rows: list[dict[str, object]],
    panel: dict[str, object],
    *,
    show_ylabel: bool,
    y_limits: tuple[float, float],
) -> None:
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    x_col = str(panel["x_col"])
    for row in rows:
        ax.scatter(
            float(row[x_col]),
            float(row["mAP1-5_pct"]),
            s=POINT_SIZE,
            color=str(row["object_color"]),
            marker=str(row["pose_marker"]),
            edgecolors="white",
            linewidths=0.4,
            alpha=0.92,
            zorder=2,
        )

    x_values = [float(row[x_col]) for row in rows]
    _apply_axis_limits(ax, x_values, axis="x", margin_frac=X_AXIS_MARGIN_FRAC, min_pad=float(panel["x_min_pad"]))
    x_axis_max = panel.get("x_axis_max")
    if x_axis_max is not None:
        current_left, current_right = ax.get_xlim()
        ax.set_xlim(current_left, max(current_right, float(x_axis_max)))

    ax.set_ylim(*y_limits)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    _draw_reference_guides(ax, panel, x_max - x_min, y_max - y_min)

    ax.set_title(str(panel["title"]), fontsize=12.2, pad=5)
    ax.set_xlabel(str(panel["x_label"]), fontsize=11.4)
    ax.set_ylabel("Pass mAP$_{1-5}$ (%)" if show_ylabel else "", fontsize=11.4)
    ax.tick_params(axis="both", labelsize=10.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter(str(panel["x_tick_fmt"])))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle="--", alpha=0.4)


def build_figure(rows: list[dict[str, object]]):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)
    grid = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.04])

    ax_gpu = fig.add_subplot(grid[0, 0:2])
    ax_power = fig.add_subplot(grid[0, 2:4], sharey=ax_gpu)
    ax_temp = fig.add_subplot(grid[1, 1:3], sharey=ax_gpu)
    y_limits = _compute_y_limits(rows)

    _plot_panel(ax_gpu, rows, PANELS[0], show_ylabel=True, y_limits=y_limits)
    _plot_panel(ax_power, rows, PANELS[1], show_ylabel=False, y_limits=y_limits)
    _plot_panel(ax_temp, rows, PANELS[2], show_ylabel=True, y_limits=y_limits)

    ax_power.tick_params(axis="y", labelleft=False)

    object_handles, pose_handles = _build_legend_handles()
    fig.legend(
        handles=object_handles,
        frameon=False,
        fontsize=8.2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=4,
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.legend(
        handles=pose_handles,
        frameon=False,
        fontsize=8.2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.16, hspace=0.34, wspace=0.48)
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
        rows = load_rows(args.json, x_col="GPU (%)")
        fig, _ = build_figure(rows)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Rows used:", len(rows))
    print("JSON:", args.json)

    if args.save_name:
        save_direct(fig, args.save_name)

    if not args.no_show:
        plt.show()
        if not args.save_name:
            prompt_save_figure(fig, default_name="sweep_new_map1_5_hardware_grid")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
