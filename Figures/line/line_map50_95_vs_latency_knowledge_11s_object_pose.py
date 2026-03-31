#!/usr/bin/env python
"""
Focused comparison of YOLO11 small-model transform combinations.

The figure shows two horizontal plots for YOLO11s object models only.

Each plot shows averaged transformation paths:

- Baseline
- Accel
- Quantized
- Pruned
- Pruned+Accel
- Pruned+Quantized
- Distilled
- Distilled+Accel
- Distilled+Quantized
- Distilled+Pruned
- Distilled+Pruned+Accel
- Distilled+Pruned+Quantized

Run:
  python Figures/line/line_map50_95_vs_latency_knowledge_11s_object_pose.py
  python Figures/line/line_map50_95_vs_latency_knowledge_11s_object_pose.py --no-show
  python Figures/line/line_map50_95_vs_latency_knowledge_11s_object_pose.py --output Figures/produced_images/map50_95_vs_latency_combinations_11s_object_paths.png
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

OBJECT_GROUP = ("11", "object")
OBJECT_LABEL = "YOLO11s Obj"

COMBO_ORDER = [
    "baseline",
    "accel",
    "quantized",
    "pruned",
    "accel_pruned",
    "accel_pruned_quantized",
    "distilled",
    "distilled_accel",
    "distilled_quantized",
    "distilled_pruned",
    "distilled_pruned_accel",
    "distilled_pruned_quantized",
]

COMBO_STYLES = {
    "baseline": {
        "label": "Uncompressed",
        "annotation": "Uncompressed",
        "marker": "o",
        "color": "#1f77b4",
    },
    "accel": {"label": "Accel", "annotation": "Accel", "marker": "o", "color": "#ff7f0e"},
    "quantized": {"label": "Quantized", "annotation": "Quantized", "marker": "o", "color": "#d62728"},
    "pruned": {"label": "Pruned", "annotation": "Pruned", "marker": "o", "color": "#2ca02c"},
    "accel_pruned": {
        "label": "Pruned+Accel",
        "annotation": "Pruned+Accel",
        "marker": "o",
        "color": "#9467bd",
    },
    "accel_pruned_quantized": {
        "label": "Pruned+Quantized",
        "annotation": "Pruned+Quantized",
        "marker": "o",
        "color": "#7f7f7f",
    },
    "distilled": {"label": "Distilled", "annotation": "Distilled", "marker": "o", "color": "#17becf"},
    "distilled_accel": {
        "label": "Distilled+Accel",
        "annotation": "Distilled+Accel",
        "marker": "o",
        "color": "#1b9e9a",
    },
    "distilled_quantized": {
        "label": "Distilled+Quantized",
        "annotation": "Distilled+Quantized",
        "marker": "o",
        "color": "#bcbd22",
    },
    "distilled_pruned": {
        "label": "Distilled+Pruned",
        "annotation": "Distilled+Pruned",
        "marker": "o",
        "color": "#8c564b",
    },
    "distilled_pruned_accel": {
        "label": "Distilled+Pruned+Accel",
        "annotation": "Distilled+Pruned+Accel",
        "marker": "o",
        "color": "#c49c94",
    },
    "distilled_pruned_quantized": {
        "label": "Distilled+Pruned+Quantized",
        "annotation": "Distilled+Pruned+Quantized",
        "marker": "o",
        "color": "#e377c2",
    },
}

PANEL_PATHS = [
    [
        ["baseline", "accel", "quantized"],
        ["baseline", "pruned", "accel_pruned", "accel_pruned_quantized"],
    ],
    [
        ["baseline", "distilled", "distilled_accel", "distilled_quantized"],
        ["baseline", "distilled", "distilled_pruned", "distilled_pruned_accel", "distilled_pruned_quantized"],
    ],
]

PANEL_TITLES = [
    "Acceleration and Pruning Based Combos",
    "Distillation and Distilled Pruning Combos",
]

DISTILL_TOKENS = ("distill", "distillation", "student", "teacher", "kd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot YOLO11s object transformation paths as two horizontal averaged graphs."
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to the model summary CSV.")
    parser.add_argument("--output", type=Path, help="Optional PNG output path.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening a plot window.")
    return parser.parse_args()


def _normalize_location(value: str) -> str:
    return str(value).replace("\\", "/").lower()


def _normalize_model(value: str) -> str:
    return str(value).strip().lower()


def parse_model(model: str, location: str) -> Optional[dict[str, str]]:
    text = _normalize_model(model)
    loc = _normalize_location(location)
    if not text:
        return None

    if not text.startswith("11s_ds3"):
        return None

    is_engine = ".engine" in text
    is_distilled = "_from_" in text or any(token in text or token in loc for token in DISTILL_TOKENS)
    in_baseline = "/baseline/" in loc or "baseline" in text
    in_distillation_pruning_quantization = "/distillation_pruning_quantization/" in loc
    in_distillation_pruning = "/distillation_pruning/" in loc
    in_pruning_quantization = "/pruning_quantization/" in loc
    in_pruning = "/pruning/" in loc or bool(re.search(r"_p(70|80|90)", text))
    in_quantization = (
        "/quantization/" in loc
        or "/distilled-quantized/" in loc
        or "fp16" in text
        or "int8" in text
    )

    if in_baseline:
        combo = "accel" if is_engine else "baseline"
    elif in_distillation_pruning_quantization:
        combo = "distilled_pruned_quantized"
    elif in_distillation_pruning:
        combo = "distilled_pruned_accel" if is_engine else "distilled_pruned"
    elif is_distilled and in_quantization:
        combo = "distilled_quantized"
    elif is_distilled:
        combo = "distilled_accel" if is_engine else "distilled"
    elif in_pruning_quantization:
        combo = "accel_pruned_quantized"
    elif in_pruning:
        combo = "accel_pruned" if is_engine else "pruned"
    elif in_quantization:
        combo = "quantized"
    else:
        return None

    return {
        "family": "11s",
        "domain": "object",
        "combo": combo,
    }


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
            parsed = parse_model(row.get("Model", ""), row.get("Location", ""))
            if parsed is None:
                continue

            try:
                map_value = float(str(row[MAP_COLUMN]).strip())
                latency_value = float(str(row[LATENCY_COLUMN]).strip())
            except (TypeError, ValueError):
                continue

            usable.append(
                {
                    **row,
                    **parsed,
                    MAP_COLUMN: map_value,
                    LATENCY_COLUMN: latency_value,
                }
            )

    if not usable:
        raise ValueError("No matching YOLO11s object transform rows contain numeric mAP50-95 and latency values.")

    return usable


def build_group_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, list[dict[str, object]]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = {
        OBJECT_GROUP: {combo: [] for combo in COMBO_ORDER}
    }

    for row in rows:
        group_key = (str(row["family"])[:2], str(row["domain"]))
        combo = str(row["combo"])
        if group_key in grouped and combo in grouped[group_key]:
            grouped[group_key][combo].append(row)

    missing = []
    for combo in COMBO_ORDER:
        if not grouped[OBJECT_GROUP][combo]:
            missing.append(f"{OBJECT_LABEL} / {COMBO_STYLES[combo]['label']}")

    if missing:
        raise ValueError("Missing one or more required transform combinations: " + ", ".join(missing))

    return grouped


def build_group_averages(
    grouped_rows: dict[tuple[str, str], dict[str, list[dict[str, object]]]]
) -> dict[tuple[str, str], dict[str, tuple[float, float, int]]]:
    averages: dict[tuple[str, str], dict[str, tuple[float, float, int]]] = {}

    for group_key, combo_rows in grouped_rows.items():
        averages[group_key] = {}
        for combo in COMBO_ORDER:
            rows = combo_rows[combo]
            if not rows:
                continue
            latencies = [float(row[LATENCY_COLUMN]) for row in rows]
            map_values = [float(row[MAP_COLUMN]) for row in rows]
            averages[group_key][combo] = (
                sum(latencies) / len(latencies),
                sum(map_values) / len(map_values),
                len(rows),
            )

    return averages


def print_group_summary(grouped_rows: dict[tuple[str, str], dict[str, list[dict[str, object]]]]) -> None:
    combo_counts = ", ".join(
        f"{COMBO_STYLES[combo]['label']}={len(grouped_rows[OBJECT_GROUP][combo])}" for combo in COMBO_ORDER
    )
    print(f"- {OBJECT_LABEL}: {combo_counts}")


def build_figure(group_averages: dict[tuple[str, str], dict[str, tuple[float, float, int]]]):
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

    point_offsets = {
        "baseline": (-7, 7),
        "accel": (-7, 7),
        "quantized": (7, -7),
        "pruned": (-7, 7),
        "accel_pruned": (7, -7),
        "accel_pruned_quantized": (7, 1),
        "distilled": (7, -7),
        "distilled_accel": (7, -7),
        "distilled_quantized": (7, 7),
        "distilled_pruned": (-7, 7),
        "distilled_pruned_accel": (-7, 7),
        "distilled_pruned_quantized": (7, 7),
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN * 0.50),
        sharex=False,
        sharey=False,
    )

    object_averages = group_averages[OBJECT_GROUP]

    for index, panel_paths in enumerate(PANEL_PATHS):
        ax = axes[index]
        plotted_latencies: list[float] = []
        plotted_maps: list[float] = []
        combo_points: dict[str, tuple[float, float]] = {}
        panel_combos: list[str] = []
        for path in panel_paths:
            for combo in path:
                if combo not in panel_combos:
                    panel_combos.append(combo)

        for combo in panel_combos:
            if combo not in object_averages:
                continue
            combo_color = str(COMBO_STYLES[combo]["color"])
            stage_latency, stage_map, _ = object_averages[combo]
            combo_points[combo] = (stage_latency, stage_map)
            plotted_latencies.append(stage_latency)
            plotted_maps.append(stage_map)
            ax.scatter(
                stage_latency,
                stage_map,
                color=combo_color,
                marker=COMBO_STYLES[combo]["marker"],
                s=94,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
            offset_x, offset_y = point_offsets[combo]
            text_ha = "left" if offset_x > 0 else "right"
            text_va = "bottom" if offset_y > 0 else "top"
            ax.annotate(
                str(COMBO_STYLES[combo]["annotation"]),
                xy=(stage_latency, stage_map),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                color=combo_color,
                fontsize=7.4,
                ha=text_ha,
                va=text_va,
            )

        for path in panel_paths:
            path_points = [combo_points[combo] for combo in path if combo in combo_points]
            if len(path_points) >= 2:
                ax.plot(
                    [point[0] for point in path_points],
                    [point[1] for point in path_points],
                    color="#5a5a5a",
                    linewidth=1.2,
                    alpha=0.75,
                    zorder=1,
                )

        ax.set_title(PANEL_TITLES[index], fontsize=8.8, color="black")
        ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Latency ms", fontsize=11)
        if index == 0:
            ax.set_ylabel(MAP_COLUMN, fontsize=11)

        if plotted_latencies and plotted_maps:
            x_min = min(plotted_latencies)
            x_max = max(plotted_latencies)
            y_min = min(plotted_maps)
            y_max = max(plotted_maps)
            x_pad = max(1.0, (x_max - x_min) * 0.18) if x_max > x_min else max(1.0, x_max * 0.10)
            y_pad = max(0.005, (y_max - y_min) * 0.22) if y_max > y_min else max(0.005, y_max * 0.02)
            ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
            ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    fig.tight_layout(rect=[0, 0, 1, 1])
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
            saved_path = prompt_save_figure(fig, default_name="map50_95_vs_latency_combinations_11s_object_paths")
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
