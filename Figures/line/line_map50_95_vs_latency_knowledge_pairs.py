#!/usr/bin/env python
"""
Single-plot comparison of averaged distillation jump stages using .engine rows.

The figure averages the available n/s/m/l/x models within each group and overlays:

- YOLO26 Obj
- YOLO11 KP
- YOLO26 KP

Jump size is defined by teacher-to-student family distance:

- Small: one size step larger
- Medium: two size steps larger
- Big: three or more size steps larger

Lines connect baseline -> small -> medium -> big when those stages exist.

Run:
  python Figures/line/line_map50_95_vs_latency_knowledge_pairs.py
  python Figures/line/line_map50_95_vs_latency_knowledge_pairs.py --no-show
  python Figures/line/line_map50_95_vs_latency_knowledge_pairs.py --output Figures/produced_images/map50_95_vs_latency_knowledge_pairs.png
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

SIZE_ORDER = ["n", "s", "m", "l", "x"]
SIZE_INDEX = {size: idx for idx, size in enumerate(SIZE_ORDER)}

OBJECT_BASELINE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_baseline\.engine$", re.IGNORECASE)
OBJECT_DISTILL_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_from_(?P<src>(11|26)[nsmlx])\.engine$", re.IGNORECASE)
POSE_BASELINE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_baseline\.engine$", re.IGNORECASE)
POSE_DISTILL_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_from_(?P<src>(11|26)[nsmlx])\.engine$", re.IGNORECASE)

GROUP_ORDER = [
    ("26", "object"),
    ("11", "pose"),
    ("26", "pose"),
]

GROUP_LABELS = {
    ("26", "object"): "YOLO26 Obj",
    ("11", "pose"): "YOLO11 KP",
    ("26", "pose"): "YOLO26 KP",
}

GROUP_COLORS = {
    ("26", "object"): "#2ca02c",
    ("11", "pose"): "#ff7f0e",
    ("26", "pose"): "#d62728",
}

STAGE_ORDER = ["baseline", "small", "medium", "big"]

STAGE_STYLES = {
    "baseline": {"label": "baseline", "marker": "o"},
    "small": {"label": "Small", "marker": "s"},
    "medium": {"label": "Medium", "marker": "^"},
    "big": {"label": "Big", "marker": "D"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot averaged mAP50-95 vs latency for baseline and distillation jump .engine stages."
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to the model summary CSV.")
    parser.add_argument("--output", type=Path, help="Optional PNG output path.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening a plot window.")
    return parser.parse_args()


def infer_jump_stage(target_family: str, source_family: str) -> Optional[str]:
    if target_family[:2] != source_family[:2]:
        return None

    target_idx = SIZE_INDEX.get(target_family[-1])
    source_idx = SIZE_INDEX.get(source_family[-1])
    if target_idx is None or source_idx is None:
        return None

    jump = source_idx - target_idx
    if jump <= 0:
        return None
    if jump == 1:
        return "small"
    if jump == 2:
        return "medium"
    return "big"


def parse_model(model: str) -> Optional[dict[str, str]]:
    text = str(model).strip()
    if not text:
        return None

    match = OBJECT_BASELINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {"family": family, "domain": "object", "stage": "baseline"}

    match = OBJECT_DISTILL_RE.match(text)
    if match:
        family = match.group("family").lower()
        stage = infer_jump_stage(family, match.group("src").lower())
        if stage is None:
            return None
        return {"family": family, "domain": "object", "stage": stage}

    match = POSE_BASELINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {"family": family, "domain": "pose", "stage": "baseline"}

    match = POSE_DISTILL_RE.match(text)
    if match:
        family = match.group("family").lower()
        stage = infer_jump_stage(family, match.group("src").lower())
        if stage is None:
            return None
        return {"family": family, "domain": "pose", "stage": stage}

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
        raise ValueError("No matching distillation rows contain numeric mAP50-95 and latency values.")

    return usable


def build_group_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, list[dict[str, object]]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = {
        group_key: {stage: [] for stage in STAGE_ORDER} for group_key in GROUP_ORDER
    }

    for row in rows:
        group_key = (str(row["family"])[:2], str(row["domain"]))
        if group_key in grouped:
            grouped[group_key][str(row["stage"])].append(row)

    missing_baselines = [GROUP_LABELS[group_key] for group_key in GROUP_ORDER if not grouped[group_key]["baseline"]]
    if missing_baselines:
        raise ValueError("Missing one or more required distillation baselines: " + ", ".join(missing_baselines))

    return grouped


def build_group_averages(
    grouped_rows: dict[tuple[str, str], dict[str, list[dict[str, object]]]]
) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    averages: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}

    for group_key, stage_rows in grouped_rows.items():
        averages[group_key] = {}
        for stage in STAGE_ORDER:
            if not stage_rows[stage]:
                continue
            latencies = [float(row[LATENCY_COLUMN]) for row in stage_rows[stage]]
            map_values = [float(row[MAP_COLUMN]) for row in stage_rows[stage]]
            averages[group_key][stage] = (
                sum(latencies) / len(latencies),
                sum(map_values) / len(map_values),
            )

    return averages


def print_group_summary(grouped_rows: dict[tuple[str, str], dict[str, list[dict[str, object]]]]) -> None:
    for group_key in GROUP_ORDER:
        counts = ", ".join(f"{stage}={len(grouped_rows[group_key][stage])}" for stage in STAGE_ORDER)
        print(f"- {GROUP_LABELS[group_key]}: {counts}")


def build_figure(group_averages: dict[tuple[str, str], dict[str, tuple[float, float]]]):
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

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN * 0.72))

    for group_key in GROUP_ORDER:
        group_color = GROUP_COLORS[group_key]
        available_stages = [stage for stage in STAGE_ORDER if stage in group_averages[group_key]]
        latencies = [group_averages[group_key][stage][0] for stage in available_stages]
        map_values = [group_averages[group_key][stage][1] for stage in available_stages]

        ax.plot(latencies, map_values, color=group_color, linewidth=1.2, zorder=1)

        for stage in available_stages:
            stage_latency, stage_map = group_averages[group_key][stage]
            ax.scatter(
                stage_latency,
                stage_map,
                color=group_color,
                marker=STAGE_STYLES[stage]["marker"],
                s=82,
                zorder=3,
            )

    ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Latency ms", fontsize=11)
    ax.set_ylabel(MAP_COLUMN, fontsize=11)

    group_handles = [
        Line2D([0], [0], color=GROUP_COLORS[group_key], linewidth=1.6, label=GROUP_LABELS[group_key])
        for group_key in GROUP_ORDER
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

    ax.legend(
        handles=group_handles + stage_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=7,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.5,
        columnspacing=0.8,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.91])
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
        group_averages = build_group_averages(grouped_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_group_summary(grouped_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.", file=sys.stderr)
        return 1

    fig = build_figure(group_averages)

    saved_path: Optional[Path] = None
    try:
        if args.output is not None:
            saved_path = save_figure(fig, args.output)
            print(f"Saved figure to {saved_path}")
        elif not args.no_show:
            plt.show()
            saved_path = prompt_save_figure(fig, default_name="map50_95_vs_latency_knowledge_pairs")
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
