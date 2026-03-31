#!/usr/bin/env python
"""
Single-plot comparison of averaged pruning stages for YOLO11 .pt rows.

The figure averages the available n/s/m/l/x models within each YOLO11 group and overlays:

- YOLO11 Obj
- YOLO11 KP

Lines connect baseline -> P90 -> P80 when those stages exist.

Run:
  python Figures/line/line_map50_95_vs_latency_pruning_pairs.py
  python Figures/line/line_map50_95_vs_latency_pruning_pairs.py --no-show
  python Figures/line/line_map50_95_vs_latency_pruning_pairs.py --output Figures/produced_images/map50_95_vs_latency_pruning_pairs.png
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

OBJECT_BASELINE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_baseline\.pt$", re.IGNORECASE)
OBJECT_PRUNE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_p(?P<pct>80|90)\.pt$", re.IGNORECASE)
POSE_BASELINE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_baseline\.pt$", re.IGNORECASE)
POSE_PRUNE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_p(?P<pct>80|90)\.pt$", re.IGNORECASE)

GROUP_ORDER = [
    ("11", "object"),
    ("11", "pose"),
]

GROUP_LABELS = {
    ("11", "object"): "YOLO11 Obj",
    ("11", "pose"): "YOLO11 KP",
}

GROUP_COLORS = {
    ("11", "object"): "#1f77b4",
    ("11", "pose"): "#ff7f0e",
}

STAGE_ORDER = ["baseline", "p90", "p80"]

STAGE_STYLES = {
    "baseline": {"label": "baseline", "marker": "o"},
    "p90": {"label": "P90", "marker": "s"},
    "p80": {"label": "P80", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot averaged mAP50-95 vs latency for YOLO11 baseline and pruning .pt stages."
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
) -> dict[tuple[str, str], dict[str, list[dict[str, object]]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = {
        group_key: {stage: [] for stage in STAGE_ORDER} for group_key in GROUP_ORDER
    }

    for row in rows:
        group_key = (str(row["family"])[:2], str(row["domain"]))
        stage = str(row["stage"])
        if group_key in grouped and stage in grouped[group_key]:
            grouped[group_key][stage].append(row)

    missing_baselines = [GROUP_LABELS[group_key] for group_key in GROUP_ORDER if not grouped[group_key]["baseline"]]
    if missing_baselines:
        raise ValueError("Missing one or more required pruning baselines: " + ", ".join(missing_baselines))

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


def print_group_summary(
    grouped_rows: dict[tuple[str, str], dict[str, list[dict[str, object]]]]
) -> None:
    for group_key in GROUP_ORDER:
        counts = ", ".join(f"{stage}={len(grouped_rows[group_key][stage])}" for stage in STAGE_ORDER)
        print(f"- {GROUP_LABELS[group_key]}: {counts}")


def build_figure(group_averages: dict[tuple[str, str], dict[str, tuple[float, float]]]):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    group_label_offsets = {
        ("11", "object"): (8, 10),
        ("11", "pose"): (8, 8),
    }
    stage_label_offsets = {
        ("11", "object"): {
            "p90": (8, 10),
            "p80": (8, -12),
        },
        ("11", "pose"): {
            "p90": (8, -12),
            "p80": (8, 10),
        },
    }

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

            if stage != "baseline":
                offset_x, offset_y = stage_label_offsets[group_key][stage]
                ax.annotate(
                    STAGE_STYLES[stage]["label"],
                    xy=(stage_latency, stage_map),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    color="#444444",
                    fontsize=8.5,
                    ha="left",
                    va="center",
                )

        baseline_latency, baseline_map = group_averages[group_key]["baseline"]
        label_offset_x, label_offset_y = group_label_offsets[group_key]
        ax.annotate(
            GROUP_LABELS[group_key],
            xy=(baseline_latency, baseline_map),
            xytext=(label_offset_x, label_offset_y),
            textcoords="offset points",
            color=group_color,
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="center",
        )

    ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Latency ms", fontsize=11)
    ax.set_ylabel(MAP_COLUMN, fontsize=11)
    fig.tight_layout()
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
            saved_path = prompt_save_figure(fig, default_name="map50_95_vs_latency_pruning_pairs")
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
