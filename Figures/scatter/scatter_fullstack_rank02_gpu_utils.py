#!/usr/bin/env python
"""
Shared plotting logic for rank-02 full-stack combo scatter charts.

Data source:
  research/sweep_replay/replay_20260319_204802/rank_02/batch_20260319_215159/batch_summary.csv

Two metrics are supported:
  - Pass mAP1-5
  - Possession accuracy, derived as 100 - DeltaPosPct

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
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 4.2
POINT_SIZE = 38

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


def load_data(csv_path: Path = CSV_PATH) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "combo_name",
        "mAP1-5",
        "GPU (%)",
        "pred_passes_club1",
        "pred_passes_total",
        "gt_passes_club1",
        "gt_passes_total",
    }
    if not rows:
        raise ValueError("Batch summary is empty.")

    missing = [col for col in required if col not in rows[0]]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    usable: list[dict[str, object]] = []
    for row in rows:
        map_1_5 = _to_float(row, "mAP1-5")
        gpu_pct = _to_float(row, "GPU (%)")
        pred_club1 = _to_float(row, "pred_passes_club1")
        pred_total = _to_float(row, "pred_passes_total")
        gt_club1 = _to_float(row, "gt_passes_club1")
        gt_total = _to_float(row, "gt_passes_total")
        combo_name = str(row["combo_name"])

        if None in (map_1_5, gpu_pct, pred_club1, pred_total, gt_club1, gt_total):
            continue
        if not pred_total or not gt_total:
            continue

        delta_pos_pct = abs((pred_club1 / pred_total) - (gt_club1 / gt_total)) * 100.0
        possession_accuracy_pct = 100.0 - delta_pos_pct
        object_label, object_color = _object_style(combo_name)
        pose_label, pose_marker = _pose_style(combo_name)

        usable.append(
            {
                "combo_name": combo_name,
                "mAP1-5": map_1_5,
                "GPU (%)": gpu_pct,
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

    usable.sort(key=lambda row: float(row["mAP1-5"]), reverse=True)
    return usable


def plot_metric(
    rows: list[dict[str, object]],
    *,
    y_col: str,
    y_label: str,
    title: str,
) -> tuple[object, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    for row in rows:
        ax.scatter(
            float(row["GPU (%)"]),
            float(row[y_col]),
            s=POINT_SIZE,
            color=str(row["object_color"]),
            marker=str(row["pose_marker"]),
            edgecolors="black",
            linewidths=0.35,
            alpha=0.88,
        )

    ax.set_xlabel("GPU Usage (%)", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)

    if y_col == "mAP1-5":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

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

    object_legend = ax.legend(
        handles=object_handles,
        title="Object model",
        frameon=False,
        fontsize=6,
        title_fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    ax.add_artist(object_legend)

    ax.legend(
        handles=pose_handles,
        title="Pose model",
        frameon=False,
        fontsize=6,
        title_fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.33),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.43, top=0.92)
    return fig, ax


def _save_direct(fig, name: str) -> Path:
    cleaned = sanitize_filename(name)
    if not cleaned.lower().endswith(".png"):
        cleaned = f"{cleaned}.png"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / cleaned
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    return out_path


def run_metric_plot(
    *,
    y_col: str,
    y_label: str,
    title: str,
    default_name: str,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-name", help="Save directly to Figures/produced_images without prompting.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    args = parser.parse_args()

    try:
        rows = load_data()
        fig, _ = plot_metric(rows, y_col=y_col, y_label=y_label, title=title)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Rows used:", len(rows))
    print("CSV:", CSV_PATH)

    if args.save_name:
        _save_direct(fig, args.save_name)

    if not args.no_show:
        plt.show()
        if not args.save_name:
            prompt_save_figure(fig, default_name=default_name)

    return 0
