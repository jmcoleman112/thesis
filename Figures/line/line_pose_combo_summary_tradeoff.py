#!/usr/bin/env python
"""
Compact mAP50-95 vs latency plot for the pose combo summary.

The table in Thesis/tables/pose_combo_summary.tex remains percentage-based.
This companion plot keeps the absolute baseline anchors, then projects the
family-matched baseline .engine deltas for each combo stage back into the same
absolute mAP50-95 vs latency space using:
  research/model_summaries.csv

so it can sit beside the table as a direct mAP50-95 vs latency view.

Run:
  python Figures/line/line_pose_combo_summary_tradeoff.py
  python Figures/line/line_pose_combo_summary_tradeoff.py --no-show
  python Figures/line/line_pose_combo_summary_tradeoff.py --output Figures/produced_images/pose_combo_summary_tradeoff.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

from figure_save_dialog import prompt_save_figure

CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "Figures" / "produced_images" / "pose_combo_summary_tradeoff.png"
MAP_COLUMN = "mAP50-95"
LATENCY_COLUMN = "Latency ms"
REQUIRED_COLUMNS = ["Model", "Location", MAP_COLUMN, LATENCY_COLUMN]
DISTILL_TOKENS = ("distill", "distillation", "student", "teacher", "kd")
INPUT_SEGMENTS = (
    "/input_reduction/",
    "/quantization_input/",
    "/pruning_input/",
    "/pruning_quantization_input/",
    "/distilled_input/",
    "/distilled_quantized_input/",
)

GROUP_COLORS = {
    "Baseline": "#111111",
    "Quantized": "#1f78b4",
    "Pruned": "#d95f02",
    "Distilled": "#33a02c",
    "Distilled + Pruned": "#7570b3",
}

POINT_ORDER = [
    "Uncompressed",
    "Baseline engine",
    "Baseline FP16",
    "Quantized + 960",
    "Quantized + 768",
    "Pruned + Accelerated",
    "Pruned + 960",
    "Pruned + FP16",
    "Pruned + 960 + FP16",
    "Pruned + INT8",
    "Distilled + Pruned + Accel.",
    "Distilled + Pruned + FP16",
    "Distilled + Accelerated",
    "Distilled + 960",
    "Distilled + FP16",
]

POINT_LABELS = {
    "Uncompressed": "PT",
    "Baseline engine": "Accel.",
    "Baseline FP16": "FP16",
    "Quantized + 960": "75%px",
    "Quantized + 768": "60%px",
    "Pruned + Accelerated": "Accel.",
    "Pruned + 960": "75%px",
    "Pruned + FP16": "FP16",
    "Pruned + 960 + FP16": "75%px+FP16",
    "Pruned + INT8": "INT8",
    "Distilled + Pruned + Accel.": "Accel.",
    "Distilled + Pruned + FP16": "FP16",
    "Distilled + Accelerated": "Accel.",
    "Distilled + 960": "75%px",
    "Distilled + FP16": "FP16",
}

LABEL_OFFSETS = {
    "Uncompressed": (-10, -10),
    "Baseline engine": (2, 8),
    "Baseline FP16": (2, 5),
    "Quantized + 960": (6, 0),
    "Quantized + 768": (6, 0),
    "Pruned + Accelerated": (6, 0),
    "Pruned + 960": (2, -6),
    "Pruned + FP16": (1, 5),
    "Pruned + 960 + FP16": (-4, 0),
    "Pruned + INT8": (-6, 0),
    "Distilled + Pruned + Accel.": (-6, 3),
    "Distilled + Pruned + FP16": (-6, 2),
    "Distilled + Accelerated": (8, -6),
    "Distilled + 960": (8, -4),
    "Distilled + FP16": (8, 4),
}

PATHS = [
    ["Uncompressed", "Baseline engine", "Baseline FP16"],
    ["Baseline FP16", "Quantized + 960", "Quantized + 768"],
    ["Baseline engine", "Pruned + Accelerated", "Pruned + 960", "Pruned + 960 + FP16"],
    ["Pruned + Accelerated", "Pruned + FP16", "Pruned + INT8"],
    ["Baseline engine", "Distilled + Pruned + Accel.", "Distilled + Pruned + FP16"],
    ["Baseline engine", "Distilled + Accelerated"],
    ["Baseline engine", "Distilled + 960"],
    ["Distilled + Accelerated", "Distilled + FP16"],
]

ROW_FILTERS = {
    "Uncompressed": [
        {"stage": "baseline", "artifact": "pt", "quant_mode": "none"},
    ],
    "Baseline engine": [
        {"stage": "baseline", "artifact": "engine", "quant_mode": "none"},
    ],
    "Baseline FP16": [
        {"stage": "quantized", "artifact": "engine", "quant_mode": "fp16"},
    ],
    "Quantized + 960": [
        {"stage": "quantized_input", "artifact": "engine", "quant_mode": "fp16", "input_stage": "960"},
    ],
    "Quantized + 768": [
        {"stage": "quantized_input", "artifact": "engine", "quant_mode": "fp16", "input_stage": "768"},
    ],
    "Quantized + 640": [
        {"stage": "quantized_input", "artifact": "engine", "quant_mode": "fp16", "input_stage": "640"},
    ],
    "Pruned + Accelerated": [
        {"stage": "pruned", "artifact": "engine", "quant_mode": "none"},
    ],
    "Pruned + 960": [
        {"stage": "pruned_input", "artifact": "engine", "quant_mode": "none", "input_stage": "960"},
    ],
    "Pruned + FP16": [
        {"stage": "pruned_quantized", "artifact": "engine", "quant_mode": "fp16"},
    ],
    "Pruned + 960 + FP16": [
        {"stage": "pruned_quantized_input", "artifact": "engine", "quant_mode": "fp16", "input_stage": "960"},
    ],
    "Pruned + INT8": [
        {"stage": "pruned_quantized", "artifact": "engine", "quant_mode": "int8"},
    ],
    "Distilled + Pruned + Accel.": [
        {"stage": "distilled_pruned", "artifact": "engine", "quant_mode": "none"},
    ],
    "Distilled + Pruned + FP16": [
        {"stage": "distilled_pruned_quantized", "artifact": "engine", "quant_mode": "fp16"},
    ],
    "Distilled + Accelerated": [
        {"stage": "distilled", "artifact": "engine", "quant_mode": "none"},
    ],
    "Distilled + 960": [
        {"stage": "distilled_input", "artifact": "engine", "quant_mode": "none", "input_stage": "960"},
    ],
    "Distilled + FP16": [
        {"stage": "distilled_quantized", "artifact": "engine", "quant_mode": "fp16"},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average pose combo mAP50-95 vs latency.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to the model summary CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path.")
    parser.add_argument(
        "--series",
        choices=["11", "26", "all"],
        default="26",
        help="Optionally restrict the plot to one model series.",
    )
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening a plot window.")
    return parser.parse_args()


def normalize(text: object) -> str:
    return str(text).replace("\\", "/").strip().lower()


def infer_task(location: object) -> str:
    loc = normalize(location)
    if "/object/" in loc:
        return "object"
    if "/pose/" in loc:
        return "pose"
    return "other"


def infer_family(model: object, location: object) -> str:
    model_text = normalize(model)
    location_text = normalize(location)

    model_match = re.search(r"^(\d+[a-z])(?:_|$)", model_text)
    if model_match:
        return model_match.group(1)

    location_match = re.search(r"/(?:object|pose)/(\d+[a-z])(?:_ds3)?(?:/|$)", location_text)
    if location_match:
        return location_match.group(1)

    return "unknown"


def infer_series(family: object) -> str:
    match = re.match(r"^(\d+)", normalize(family))
    return match.group(1) if match else "unknown"


def infer_artifact(model: object) -> str:
    model_text = normalize(model)
    if ".engine" in model_text:
        return "engine"
    if model_text.endswith(".pt"):
        return "pt"
    return "other"


def infer_quant_mode(model: object, location: object) -> str:
    model_text = normalize(model)
    location_text = normalize(location)
    if "int8" in model_text or "/int8/" in location_text:
        return "int8"
    if "fp16" in model_text or "/fp16/" in location_text:
        return "fp16"
    return "none"


def infer_input_stage(model: object, location: object) -> Optional[str]:
    model_text = normalize(model)
    location_text = normalize(location)

    if not any(segment in location_text for segment in INPUT_SEGMENTS):
        return None

    model_match = re.search(r"_(960|768|640)(?:_(?:fp16|int8))?\.engine$", model_text)
    if model_match:
        return model_match.group(1)

    location_match = re.search(r"/(960|768|640)(?:/|$)", location_text)
    if location_match:
        return location_match.group(1)

    return None


def infer_stage(model: object, location: object) -> str:
    model_text = normalize(model)
    location_text = normalize(location)
    combined = f"{model_text} {location_text}"

    has_distill = any(token in combined for token in DISTILL_TOKENS)
    has_pruning = "pruning" in combined or bool(re.search(r"_p\d+", model_text))
    has_quant = "quantization" in combined or "fp16" in combined or "int8" in combined
    has_baseline = "/baseline/" in location_text or "baseline" in model_text
    input_stage = infer_input_stage(model, location)

    if input_stage is not None:
        transform_stage = {
            (False, False, False): "input",
            (False, False, True): "quantized_input",
            (False, True, False): "pruned_input",
            (False, True, True): "pruned_quantized_input",
            (True, False, False): "distilled_input",
            (True, False, True): "distilled_quantized_input",
            (True, True, False): "distilled_pruned_input",
            (True, True, True): "distilled_pruned_quantized_input",
        }
        return transform_stage[(has_distill, has_pruning, has_quant)]

    transform_stage = {
        (True, False, False): "distilled",
        (False, True, False): "pruned",
        (False, False, True): "quantized",
        (True, True, False): "distilled_pruned",
        (True, False, True): "distilled_quantized",
        (False, True, True): "pruned_quantized",
        (True, True, True): "distilled_pruned_quantized",
    }
    stage = transform_stage.get((has_distill, has_pruning, has_quant))
    if stage:
        return stage
    if has_baseline:
        return "baseline"
    return "other"


def load_pose_rows(csv_path: Path, *, series_filter: str = "all") -> list[dict[str, object]]:
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
            task = infer_task(row.get("Location", ""))
            family = infer_family(row.get("Model", ""), row.get("Location", ""))
            series = infer_series(family)

            if task != "pose" or series not in {"11", "26"} or family[-1] not in {"n", "s", "m", "l"}:
                continue
            if series_filter != "all" and series != series_filter:
                continue

            try:
                map_value = float(str(row[MAP_COLUMN]).strip())
                latency_value = float(str(row[LATENCY_COLUMN]).strip())
            except (TypeError, ValueError):
                continue

            usable.append(
                {
                    "family": family,
                    "stage": infer_stage(row.get("Model", ""), row.get("Location", "")),
                    "artifact": infer_artifact(row.get("Model", "")),
                    "quant_mode": infer_quant_mode(row.get("Model", ""), row.get("Location", "")),
                    "input_stage": infer_input_stage(row.get("Model", ""), row.get("Location", "")),
                    MAP_COLUMN: map_value,
                    LATENCY_COLUMN: latency_value,
                }
            )

    if not usable:
        raise ValueError("No matching pose rows contain numeric mAP50-95 and latency values.")

    return usable


def filtered_rows(rows: list[dict[str, object]], filters: list[dict[str, str]]) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    for row in rows:
        for row_filter in filters:
            if all(str(row.get(key)) == value for key, value in row_filter.items()):
                matched.append(row)
                break
    return matched


def average_absolute_point(group_rows: list[dict[str, object]]) -> tuple[float, float, int]:
    latencies = [float(row[LATENCY_COLUMN]) for row in group_rows]
    map_values = [float(row[MAP_COLUMN]) for row in group_rows]
    return (
        sum(latencies) / len(latencies),
        sum(map_values) / len(map_values),
        len(group_rows),
    )


def build_baseline_lookup(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    baseline_rows = filtered_rows(rows, ROW_FILTERS["Baseline engine"])
    baseline_lookup: dict[str, dict[str, object]] = {}

    for row in baseline_rows:
        family = str(row["family"])
        if family in baseline_lookup:
            raise ValueError(f"Duplicate baseline .engine rows found for family: {family}")
        baseline_lookup[family] = row

    if not baseline_lookup:
        raise ValueError("Missing baseline .engine rows for combo normalization.")

    return baseline_lookup


def average_family_matched_point(
    group_rows: list[dict[str, object]],
    baseline_lookup: dict[str, dict[str, object]],
    baseline_anchor: tuple[float, float, int],
) -> tuple[float, float, int]:
    family_rows: dict[str, list[dict[str, object]]] = {}
    for row in group_rows:
        family_rows.setdefault(str(row["family"]), []).append(row)

    map_deltas: list[float] = []
    latency_reductions: list[float] = []

    for family, rows_for_family in sorted(family_rows.items()):
        baseline_row = baseline_lookup.get(family)
        if baseline_row is None:
            raise ValueError(f"Missing baseline .engine row for family: {family}")

        baseline_map = float(baseline_row[MAP_COLUMN])
        baseline_latency = float(baseline_row[LATENCY_COLUMN])
        if baseline_map <= 0.0 or baseline_latency <= 0.0:
            raise ValueError(f"Baseline .engine metrics must be positive for family: {family}")

        avg_family_map = sum(float(row[MAP_COLUMN]) for row in rows_for_family) / len(rows_for_family)
        avg_family_latency = sum(float(row[LATENCY_COLUMN]) for row in rows_for_family) / len(rows_for_family)

        map_deltas.append((avg_family_map - baseline_map) / baseline_map)
        latency_reductions.append(1.0 - (avg_family_latency / baseline_latency))

    baseline_latency, baseline_map, _ = baseline_anchor
    avg_map_delta = sum(map_deltas) / len(map_deltas)
    avg_latency_reduction = sum(latency_reductions) / len(latency_reductions)

    return (
        baseline_latency * (1.0 - avg_latency_reduction),
        baseline_map * (1.0 + avg_map_delta),
        len(family_rows),
    )


def build_points(rows: list[dict[str, object]]) -> dict[str, tuple[float, float, int]]:
    points: dict[str, tuple[float, float, int]] = {}
    baseline_lookup = build_baseline_lookup(rows)

    for key in POINT_ORDER[:2]:
        group_rows = filtered_rows(rows, ROW_FILTERS[key])
        if not group_rows:
            raise ValueError(f"Missing rows for plot point: {key}")
        points[key] = average_absolute_point(group_rows)

    baseline_anchor = points["Baseline engine"]

    for key in POINT_ORDER[2:]:
        group_rows = filtered_rows(rows, ROW_FILTERS[key])
        if not group_rows:
            raise ValueError(f"Missing rows for plot point: {key}")
        points[key] = average_family_matched_point(group_rows, baseline_lookup, baseline_anchor)

    return points


def group_key(label: str) -> str:
    if label in {"Uncompressed", "Baseline engine", "Baseline FP16"}:
        return "Baseline"
    if label.startswith("Quantized +"):
        return "Quantized"
    if label.startswith("Distilled + Pruned"):
        return "Distilled + Pruned"
    if label.startswith("Distilled +"):
        return "Distilled"
    return "Pruned"


def build_figure(points: dict[str, tuple[float, float, int]]):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(4.3, 3.7), dpi=300)

    all_latencies = [point[0] for point in points.values()]
    all_maps = [point[1] for point in points.values()]

    for path in PATHS:
        path_points = [points[label] for label in path]
        path_color_source = path[1] if len(path) > 1 else path[0]
        line_color = GROUP_COLORS[group_key(path_color_source)]
        ax.plot(
            [point[0] for point in path_points],
            [point[1] for point in path_points],
            color=line_color,
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )

    for label in POINT_ORDER:
        latency, map_value, _ = points[label]
        color = GROUP_COLORS[group_key(label)]
        size = 62 if label in {"Uncompressed", "Baseline engine"} else 58
        ax.scatter(
            latency,
            map_value,
            color=color,
            marker="o",
            s=size,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )

    # Individual labels for points with unique identifiers.
    individual_labels = [
        "Uncompressed",
        "Quantized + 960",
        "Quantized + 768",
        "Pruned + 960 + FP16",
        "Pruned + INT8",
    ]

    for label in individual_labels:
        latency, map_value, _ = points[label]
        color = GROUP_COLORS[group_key(label)]
        text = POINT_LABELS[label]
        offset = LABEL_OFFSETS[label]
        annotation = ax.annotate(
            text,
            xy=(latency, map_value),
            xytext=offset,
            textcoords="offset points",
            ha="center" if offset[0] == 0 else ("left" if offset[0] > 0 else "right"),
            va="bottom" if offset[1] > 0 else "top",
            fontsize=6.9,
            color=color,
            weight="bold",
            zorder=5,
        )
        annotation.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

    # Shared labels with leader lines to all related points.
    # Move text_xy values to fine-tune placement.
    shared_label_groups = [
        {
            "text": "Accel.",
            "members": [
                "Baseline engine",
                "Pruned + Accelerated",
                "Distilled + Pruned + Accel.",
                "Distilled + Accelerated",
            ],
            "text_xy": (140, 0.882),
            "color": "#666666",
        },
        {
            "text": "FP16",
            "members": [
                "Baseline FP16",
                "Pruned + FP16",
                "Distilled + Pruned + FP16",
                "Distilled + FP16",
            ],
            "text_xy": (42, 0.900),
            "color": "#666666",
        },
        {
            "text": "75%px",
            "members": [
                "Pruned + 960",
                "Distilled + 960",
            ],
            "text_xy": (92, 0.882),
            "color": "#666666",
        },
    ]

    for group in shared_label_groups:
        text_x, text_y = group["text_xy"]
        text = group["text"]
        color = group["color"]

        shared_text = ax.text(
            text_x,
            text_y,
            text,
            fontsize=7.0,
            color=color,
            weight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        shared_text.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

        for member in group["members"]:
            px, py, _ = points[member]
            ax.plot(
                [text_x, px],
                [text_y, py],
                color=color,
                linewidth=0.7,
                alpha=0.8,
                zorder=2,
            )

    x_min = min(all_latencies)
    x_max = max(all_latencies)
    y_min = min(all_maps)
    y_max = max(all_maps)
    x_pad = max(2.5, (x_max - x_min) * 0.10)
    y_pad = max(0.008, (y_max - y_min) * 0.16)

    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))
    ax.set_xlabel("Average latency (ms)", fontsize=10)
    ax.set_ylabel("Average mAP50-95", fontsize=10)
    ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], color=GROUP_COLORS["Baseline"], linewidth=1.6, label="Baseline"),
        Line2D([0], [0], color=GROUP_COLORS["Quantized"], linewidth=1.6, label="Quantized"),
        Line2D([0], [0], color=GROUP_COLORS["Pruned"], linewidth=1.6, label="Pruned"),
        Line2D([0], [0], color=GROUP_COLORS["Distilled"], linewidth=1.6, label="Distilled"),
        Line2D([0], [0], color=GROUP_COLORS["Distilled + Pruned"], linewidth=1.6, label="Distilled + Pruned"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=6.4,
        frameon=False,
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.3,
    )

    fig.tight_layout()
    return fig


def save_figure(fig, output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
        rows = load_pose_rows(args.csv, series_filter=args.series)
        points = build_points(rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for label in POINT_ORDER:
        latency, map_value, count = points[label]
        print(f"{label}: n={count}, avg mAP50-95={map_value:.4f}, avg latency={latency:.2f} ms")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.", file=sys.stderr)
        return 1

    fig = build_figure(points)

    saved_path: Optional[Path] = None
    try:
        if args.output is not None:
            saved_path = save_figure(fig, args.output)
            print(f"Saved figure to {saved_path}")
        elif not args.no_show:
            plt.show()
            saved_path = prompt_save_figure(fig, default_name="pose_combo_summary_tradeoff")
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
