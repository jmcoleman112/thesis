#!/usr/bin/env python
"""
1x2 comparison of YOLO11 pruning stages using individual n/s/m/l .pt rows.

The figure shows separate n/s/m/l lines within each group:

- YOLO11 Obj
- YOLO11 KP

Lines connect baseline -> P90 -> P80 -> P70 when those stages exist.

Run:
  python Figures/line/line_map50_95_vs_latency_pruning_pairs_nsmlx.py
  python Figures/line/line_map50_95_vs_latency_pruning_pairs_nsmlx.py --no-show
  python Figures/line/line_map50_95_vs_latency_pruning_pairs_nsmlx.py --output Figures/produced_images/map50_95_vs_latency_pruning_pairs_nsmlx.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

from figure_save_dialog import prompt_save_figure

MAP_COLUMN = "mAP50-95"
LATENCY_COLUMN = "Latency ms"
REQUIRED_COLUMNS = ["Model", "Location", MAP_COLUMN, LATENCY_COLUMN]
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
MM_PER_INCH = 25.4
A4_LANDSCAPE_WIDTH_MM = 297.0
A4_LANDSCAPE_HEIGHT_MM = 210.0
HORIZONTAL_MARGIN_MM = 24.0
VERTICAL_MARGIN_MM = 24.0
FIGURE_WIDTH_IN = (A4_LANDSCAPE_WIDTH_MM - (2 * HORIZONTAL_MARGIN_MM)) / MM_PER_INCH
FIGURE_HEIGHT_IN = (A4_LANDSCAPE_HEIGHT_MM - (2 * VERTICAL_MARGIN_MM)) / MM_PER_INCH

OBJECT_BASELINE_RE = re.compile(r"^(?P<family>11[nsmlx])_ds3_baseline\.pt$", re.IGNORECASE)
OBJECT_PRUNE_RE = re.compile(r"^(?P<family>11[nsmlx])_ds3_p(?P<pct>70|80|90)\.pt$", re.IGNORECASE)
POSE_BASELINE_RE = re.compile(r"^(?P<family>11[nsmlx])_pose_baseline\.pt$", re.IGNORECASE)
POSE_PRUNE_RE = re.compile(r"^(?P<family>11[nsmlx])_pose_p(?P<pct>70|80|90)\.pt$", re.IGNORECASE)

GROUP_ORDER = [
    ("11", "object"),
    ("11", "pose"),
]

GROUP_LABELS = {
    ("11", "object"): "YOLO11 Obj",
    ("11", "pose"): "YOLO11 KP",
}

SUBPLOT_TITLES = {
    ("11", "object"): "YOLO11 Object Pruning",
    ("11", "pose"): "YOLO11 Pose Pruning",
}

SIZE_ORDER = ["n", "s", "m", "l"]

SIZE_LABELS = {
    "n": "N",
    "s": "S",
    "m": "M",
    "l": "L",
}

SIZE_COLORS = {
    "n": "#1f77b4",
    "s": "#ff7f0e",
    "m": "#2ca02c",
    "l": "#d62728",
}

STAGE_ORDER = ["baseline", "p90", "p80", "p70"]

STAGE_STYLES = {
    "baseline": {"label": "baseline", "marker": "o"},
    "p90": {"label": "P90", "marker": "s"},
    "p80": {"label": "P80", "marker": "^"},
    "p70": {"label": "P70", "marker": "D"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot YOLO11 mAP50-95 vs latency for baseline and pruning .pt stages by size family."
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to the model summary CSV.")
    parser.add_argument("--output", type=Path, help="Optional PNG output path.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening a plot window.")
    return parser.parse_args()


def parse_model(model: str) -> Optional[dict[str, str]]:
    text = str(model).strip()
    if not text:
        return None

    match = OBJECT_BASELINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {"family": family, "size": family[-1], "domain": "object", "stage": "baseline"}

    match = OBJECT_PRUNE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {"family": family, "size": family[-1], "domain": "object", "stage": f"p{match.group('pct')}"}

    match = POSE_BASELINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {"family": family, "size": family[-1], "domain": "pose", "stage": "baseline"}

    match = POSE_PRUNE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {"family": family, "size": family[-1], "domain": "pose", "stage": f"p{match.group('pct')}"}

    return None


def load_plot_rows(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        usable: list[dict[str, object]] = []
        for row in reader:
            parsed = parse_model(row.get("Model", ""))
            if parsed is None:
                continue

            try:
                map_value = float(str(row[MAP_COLUMN]).strip())
                latency_value = float(str(row[LATENCY_COLUMN]).strip())
            except (TypeError, ValueError):
                continue

            usable.append({**row, **parsed, MAP_COLUMN: map_value, LATENCY_COLUMN: latency_value})

    if not usable:
        raise ValueError("No matching pruning .pt rows contain numeric mAP50-95 and latency values.")

    return usable


def build_group_rows(
    rows: list[dict[str, object]]
) -> dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]] = {
        group_key: {size: {} for size in SIZE_ORDER} for group_key in GROUP_ORDER
    }

    for row in rows:
        group_key = (str(row["family"])[:2], str(row["domain"]))
        size = str(row["size"])
        stage = str(row["stage"])
        if group_key in grouped and size in grouped[group_key] and stage in STAGE_ORDER:
            grouped[group_key][size][stage] = row

    missing_baselines: list[str] = []
    for group_key in GROUP_ORDER:
        for size in SIZE_ORDER:
            if "baseline" not in grouped[group_key][size]:
                missing_baselines.append(f"{GROUP_LABELS[group_key]} {SIZE_LABELS[size]}")

    if missing_baselines:
        raise ValueError("Missing one or more required pruning baselines: " + ", ".join(missing_baselines))

    return grouped


def print_group_summary(
    grouped_rows: dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]]
) -> None:
    for group_key in GROUP_ORDER:
        size_summaries = []
        for size in SIZE_ORDER:
            present_stages = [stage for stage in STAGE_ORDER if stage in grouped_rows[group_key][size]]
            size_summaries.append(f"{SIZE_LABELS[size]}={','.join(present_stages)}")
        print(f"- {GROUP_LABELS[group_key]}: {'; '.join(size_summaries)}")


def build_figure(grouped_rows: dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]]):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN * 0.50), sharex=True, sharey=True)

    for index, group_key in enumerate(GROUP_ORDER):
        ax = axes[index]

        for size in SIZE_ORDER:
            size_rows = grouped_rows[group_key][size]
            available_stages = [stage for stage in STAGE_ORDER if stage in size_rows]
            latencies = [float(size_rows[stage][LATENCY_COLUMN]) for stage in available_stages]
            map_values = [float(size_rows[stage][MAP_COLUMN]) for stage in available_stages]

            ax.plot(latencies, map_values, color=SIZE_COLORS[size], linewidth=1.1, zorder=1)

            for stage in available_stages:
                stage_latency = float(size_rows[stage][LATENCY_COLUMN])
                stage_map = float(size_rows[stage][MAP_COLUMN])
                ax.scatter(
                    stage_latency,
                    stage_map,
                    color=SIZE_COLORS[size],
                    marker=STAGE_STYLES[stage]["marker"],
                    s=72,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )

        ax.set_title(SUBPLOT_TITLES[group_key], fontsize=10.5)
        ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.tick_params(labelsize=9)

        ax.set_xlabel("Latency ms", fontsize=11)
        if index == 0:
            ax.set_ylabel(MAP_COLUMN, fontsize=11)

    size_handles = [
        Line2D([0], [0], color=SIZE_COLORS[size], linewidth=1.6, label=SIZE_LABELS[size])
        for size in SIZE_ORDER
    ]
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=STAGE_STYLES[stage]["marker"],
            color="black",
            linestyle="None",
            markersize=7,
            label=STAGE_STYLES[stage]["label"],
        )
        for stage in STAGE_ORDER
    ]

    fig.legend(
        handles=size_handles + stage_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=9,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.6,
        columnspacing=0.9,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def save_figure(fig, output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def main() -> int:
    args = parse_args()

    if args.no_show or args.output:
        try:
            import matplotlib
        except Exception as exc:
            print(f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.", file=sys.stderr)
            return 1
        matplotlib.use("Agg")

    try:
        rows = load_plot_rows(args.csv)
        grouped_rows = build_group_rows(rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_group_summary(grouped_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.", file=sys.stderr)
        return 1

    fig = build_figure(grouped_rows)

    saved_path: Optional[Path] = None
    try:
        if args.output is not None:
            saved_path = save_figure(fig, args.output)
            print(f"Saved figure to {saved_path}")
        elif not args.no_show:
            plt.show()
            saved_path = prompt_save_figure(fig, default_name="map50_95_vs_latency_pruning_pairs_nsmlx")
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
