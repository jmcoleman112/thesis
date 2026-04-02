#!/usr/bin/env python
"""
Shared plotting logic for rank-02 full-stack combo scatter charts.

Data source:
  research/sweep_replay/replay_20260319_204802/rank_02/batch_20260319_215159/batch_summary.csv

The helper supports derived y-metrics such as possession accuracy and raw
hardware/runtime x-metrics such as GPU usage, latency, power, and temperature.

By default, the script shows the figure and then uses the standard save dialog
to save into Figures/produced_images. Pass --save-name to save directly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from figure_save_dialog import prompt_save_figure, sanitize_filename

CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "sweep_replay"
    / "replay_20260319_204802"
    / "rank_02"
    / "batch_20260319_215159"
    / "batch_summary.csv"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "produced_images"

FONT_FAMILY = "Times New Roman"
# Single-panel size for half-width export.
FIG_WIDTH_IN = 3.35
FIG_HEIGHT_IN = 3.25
# Two-panel size for full-width 1x2 exports with shared legends.
PAIR_FIG_WIDTH_IN = 6.35
PAIR_FIG_HEIGHT_IN = 5.10
POINT_SIZE = 26
PAIR_POINT_SIZE = 34
X_AXIS_MARGIN_FRAC = 0.28
Y_AXIS_MARGIN_FRAC = 0.16

DERIVED_COLS = {"delta_pos_pct", "possession_accuracy_pct"}

OBJECT_STYLES = [
    ("obj26m_p80_fp16_", "26m (80%, FP16)", "#1f77b4"),
    ("obj26l_fp16_", "26l (FP16)", "#ff7f0e"),
    ("obj26s_baseline_", "26s (Uncompressed)", "#2ca02c"),
    ("obj26s_fp16_", "26s (FP16)", "#d62728"),
]

POSE_STYLES = [
    ("_pose26n_p80_fp16", "26n (80%, FP16)", "s"),
    ("_pose26n_fp16", "26n (FP16)", "o"),
    ("_pose26n_kd_p80_int8", "26n (KD-26m, 80%, INT8)", "^"),
    ("_pose26n_kd26x_p80_fp16", "26n (KD-26x, 80%, FP16)", "D"),
]


def _object_style(combo_name: str) -> tuple[str, str]:
    for prefix, label, color in OBJECT_STYLES:
        if combo_name.startswith(prefix):
            return label, color
    return "Other", "#7f7f7f"


def _pose_style(combo_name: str) -> tuple[str, str]:
    for suffix, label, marker in POSE_STYLES:
        if combo_name.endswith(suffix):
            return label, marker
    return "Other", "o"


def _to_float(row: dict[str, object], key: str) -> float | None:
    try:
        return float(row[key])
    except Exception:
        return None


def _build_legend_handles():
    from matplotlib.lines import Line2D

    object_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=5,
            label=label,
        )
        for _, label, color in OBJECT_STYLES
    ]
    pose_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            linewidth=0,
            label=label,
        )
        for _, label, marker in POSE_STYLES
    ]
    return object_handles, pose_handles


def _build_pair_legend_handles():
    from matplotlib.lines import Line2D

    pair_object_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=5,
            label=label,
        )
        for label, color in [
            ("26m p80 FP16", "#1f77b4"),
            ("26l FP16", "#ff7f0e"),
            ("26s Base", "#2ca02c"),
            ("26s FP16", "#d62728"),
        ]
    ]
    pair_pose_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            linewidth=0,
            label=label,
        )
        for label, marker in [
            ("26n p80 FP16", "s"),
            ("26n FP16", "o"),
            ("26n KD26m p80 INT8", "^"),
            ("26n KD26x p80 FP16", "D"),
        ]
    ]
    return pair_object_handles, pair_pose_handles


def _apply_axis_limits(
    ax,
    rows: list[dict[str, object]],
    *,
    col: str,
    axis: str,
    margin_frac: float,
    min_pad: float = 0.0,
) -> None:
    values = [float(row[col]) for row in rows]
    if not values:
        return

    value_min = min(values)
    value_max = max(values)
    value_span = value_max - value_min
    pad = max(value_span * margin_frac, min_pad)
    if value_span == 0:
        fallback_pad = 0.5 if axis == "x" else 0.05
        pad = max(pad, max(abs(value_min) * margin_frac, fallback_pad))

    if axis == "x":
        ax.set_xlim(value_min - pad, value_max + pad)
    elif axis == "y":
        ax.set_ylim(value_min - pad, value_max + pad)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def load_data(*, metric_cols: set[str] | None = None, csv_path: Path = CSV_PATH) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    metric_cols = metric_cols or set()
    extra_required = {col for col in metric_cols if col not in DERIVED_COLS}

    required = {
        "combo_name",
        "pred_passes_club1",
        "pred_passes_total",
        "gt_passes_club1",
        "gt_passes_total",
    } | extra_required
    if not rows:
        raise ValueError("Batch summary is empty.")

    missing = [col for col in required if col not in rows[0]]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    usable: list[dict[str, object]] = []
    for row in rows:
        pred_club1 = _to_float(row, "pred_passes_club1")
        pred_total = _to_float(row, "pred_passes_total")
        gt_club1 = _to_float(row, "gt_passes_club1")
        gt_total = _to_float(row, "gt_passes_total")
        combo_name = str(row["combo_name"])

        if None in (pred_club1, pred_total, gt_club1, gt_total):
            continue
        if not pred_total or not gt_total:
            continue

        parsed_metrics: dict[str, float] = {}
        failed_metric = False
        for col in extra_required:
            value = _to_float(row, col)
            if value is None:
                failed_metric = True
                break
            parsed_metrics[col] = value
        if failed_metric:
            continue

        delta_pos_pct = abs((pred_club1 / pred_total) - (gt_club1 / gt_total)) * 100.0
        possession_accuracy_pct = 100.0 - delta_pos_pct
        object_label, object_color = _object_style(combo_name)
        pose_label, pose_marker = _pose_style(combo_name)

        usable.append(
            {
                "combo_name": combo_name,
                **parsed_metrics,
                "delta_pos_pct": delta_pos_pct,
                "possession_accuracy_pct": possession_accuracy_pct,
                "object_label": object_label,
                "object_color": object_color,
                "pose_label": pose_label,
                "pose_marker": pose_marker,
            }
        )

    if not usable:
        raise ValueError("No usable rows found in batch summary.")

    if "mAP1-5" in extra_required:
        usable.sort(key=lambda row: float(row["mAP1-5"]), reverse=True)
    return usable


def plot_metric(
    rows: list[dict[str, object]],
    *,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    title: str,
    x_tick_fmt: str | None = None,
    y_tick_fmt: str | None = None,
) -> tuple[object, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    for row in rows:
        ax.scatter(
            float(row[x_col]),
            float(row[y_col]),
            s=POINT_SIZE,
            color=str(row["object_color"]),
            marker=str(row["pose_marker"]),
            edgecolors="black",
            linewidths=0.35,
            alpha=0.88,
        )

    _apply_axis_limits(ax, rows, col=x_col, axis="x", margin_frac=X_AXIS_MARGIN_FRAC)
    _apply_axis_limits(ax, rows, col=y_col, axis="y", margin_frac=Y_AXIS_MARGIN_FRAC)

    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)

    if x_tick_fmt:
        ax.xaxis.set_major_formatter(FormatStrFormatter(x_tick_fmt))
    if y_tick_fmt:
        ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_fmt))

    object_handles, pose_handles = _build_legend_handles()

    object_legend = ax.legend(
        handles=object_handles,
        title="Object model",
        frameon=False,
        fontsize=5,
        title_fontsize=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    ax.add_artist(object_legend)

    ax.legend(
        handles=pose_handles,
        title="Pose model",
        frameon=False,
        fontsize=5,
        title_fontsize=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.54, top=0.90)
    return fig, ax


def plot_metric_pair(
    rows: list[dict[str, object]],
    *,
    left_panel: dict[str, object],
    right_panel: dict[str, object],
    y_col: str,
    y_label: str,
    y_tick_fmt: str | None = None,
) -> tuple[object, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.ticker import MaxNLocator

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(PAIR_FIG_WIDTH_IN, PAIR_FIG_HEIGHT_IN),
        dpi=300,
        sharey=True,
    )

    for ax, panel in zip(axes, (left_panel, right_panel)):
        x_col = str(panel["x_col"])
        for row in rows:
            ax.scatter(
                float(row[x_col]),
                float(row[y_col]),
                s=PAIR_POINT_SIZE,
                color=str(row["object_color"]),
                marker=str(row["pose_marker"]),
                edgecolors="black",
                linewidths=0.45,
                alpha=0.9,
            )

        _apply_axis_limits(
            ax,
            rows,
            col=x_col,
            axis="x",
            margin_frac=float(panel.get("x_margin_frac", X_AXIS_MARGIN_FRAC)),
            min_pad=float(panel.get("x_min_pad", 0.0)),
        )

        ax.set_xlabel(str(panel["x_label"]), fontsize=10)
        ax.set_title(str(panel["title"]), fontsize=10, pad=4)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, linestyle="--", alpha=0.4)

        x_tick_fmt = panel.get("x_tick_fmt")
        if x_tick_fmt:
            ax.xaxis.set_major_formatter(FormatStrFormatter(str(x_tick_fmt)))

    axes[0].set_ylabel(y_label, fontsize=10)
    _apply_axis_limits(axes[0], rows, col=y_col, axis="y", margin_frac=Y_AXIS_MARGIN_FRAC)
    if y_tick_fmt:
        axes[0].yaxis.set_major_formatter(FormatStrFormatter(y_tick_fmt))

    pair_object_handles, pair_pose_handles = _build_pair_legend_handles()
    object_legend = fig.legend(
        handles=pair_object_handles,
        title="Object model",
        frameon=False,
        fontsize=6.8,
        title_fontsize=6.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=4,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.add_artist(object_legend)
    fig.legend(
        handles=pair_pose_handles,
        title="Pose model",
        frameon=False,
        fontsize=6.8,
        title_fontsize=6.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    fig.tight_layout(pad=0.12)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.28, top=0.89, wspace=0.22)
    return fig, axes


def _save_direct(fig, name: str) -> Path:
    cleaned = sanitize_filename(name)
    if not cleaned.lower().endswith(".png"):
        cleaned = f"{cleaned}.png"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / cleaned
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    return out_path


def run_metric_plot(
    *,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    title: str,
    default_name: str,
    x_tick_fmt: str | None = None,
    y_tick_fmt: str | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-name", help="Save directly to Figures/produced_images without prompting.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    args = parser.parse_args()

    try:
        rows = load_data(metric_cols={x_col, y_col})
        fig, _ = plot_metric(
            rows,
            x_col=x_col,
            x_label=x_label,
            y_col=y_col,
            y_label=y_label,
            title=title,
            x_tick_fmt=x_tick_fmt,
            y_tick_fmt=y_tick_fmt,
        )
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Rows used:", len(rows))
    print("CSV:", CSV_PATH)
    print("X metric:", x_col)
    print("Y metric:", y_col)

    if args.save_name:
        _save_direct(fig, args.save_name)

    if not args.no_show:
        plt.show()
        if not args.save_name:
            prompt_save_figure(fig, default_name=default_name)

    return 0


def run_metric_pair_plot(
    *,
    left_panel: dict[str, object],
    right_panel: dict[str, object],
    y_col: str,
    y_label: str,
    default_name: str,
    y_tick_fmt: str | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-name", help="Save directly to Figures/produced_images without prompting.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    args = parser.parse_args()

    metric_cols = {str(left_panel["x_col"]), str(right_panel["x_col"]), y_col}

    try:
        rows = load_data(metric_cols=metric_cols)
        fig, _ = plot_metric_pair(
            rows,
            left_panel=left_panel,
            right_panel=right_panel,
            y_col=y_col,
            y_label=y_label,
            y_tick_fmt=y_tick_fmt,
        )
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Rows used:", len(rows))
    print("CSV:", CSV_PATH)
    print("X metrics:", left_panel["x_col"], right_panel["x_col"])
    print("Y metric:", y_col)

    if args.save_name:
        _save_direct(fig, args.save_name)

    if not args.no_show:
        plt.show()
        if not args.save_name:
            prompt_save_figure(fig, default_name=default_name)

    return 0
