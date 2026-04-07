#!/usr/bin/env python
"""
Compact mAP50-95 vs latency plot for the pose technique summary.

The table in Thesis/tables/pose_technique_summary.tex remains percentage-based.
This companion plot keeps the absolute uncompressed and accelerated anchors.
Quantisation is projected from family-matched baseline .engine deltas, while
pruning and distillation are projected from family-matched baseline .pt deltas,
back into the same absolute mAP50-95 vs latency space using:
  research/model_summaries.csv

Run:
  python Figures/line/line_pose_technique_summary_tradeoff.py
  python Figures/line/line_pose_technique_summary_tradeoff.py --no-show
  python Figures/line/line_pose_technique_summary_tradeoff.py --output Figures/produced_images/pose_technique_summary_tradeoff.png
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
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "Figures" / "produced_images" / "pose_technique_summary_tradeoff.png"
)
MAP_COLUMN = "mAP50-95"
LATENCY_COLUMN = "Latency ms"
REQUIRED_COLUMNS = ["Model", "Location", MAP_COLUMN, LATENCY_COLUMN]
DISTILL_TOKENS = ("distill", "distillation", "student", "teacher", "kd")
SIZE_ORDER = ["n", "s", "m", "l", "x"]
SIZE_INDEX = {size: index for index, size in enumerate(SIZE_ORDER)}

GROUP_COLORS = {
    "Baseline": "#111111",
    "Input reduction": "#1f78b4",
    "Quantisation": "#33a02c",
    "Pruning": "#ff7f00",
    "Distillation": "#6a3d9a",
}

MARKERS = {
    "base": "o",
    "accelerated": "s",
    "first": "s",
    "second": "^",
    "third": "D",
}

POINT_ORDER = [
    "Uncompressed",
    "Baseline engine",
    "Input 960",
    "Input 768",
    "Input 640",
    "Quant FP16",
    "Quant INT8",
    "Prune P90",
    "Prune P80",
    "Prune P70",
    "Distill Small",
    "Distill Medium",
    "Distill Large",
]

ANNOTATIONS = {
    "Uncompressed": ("Uncomp.", (-10, -10)),
    "Baseline engine": ("Accel.", (0, 10)),
}

PATHS = [
    ["Uncompressed", "Baseline engine"],
    ["Baseline engine", "Input 960", "Input 768", "Input 640"],
    ["Baseline engine", "Quant FP16", "Quant INT8"],
    ["Uncompressed", "Prune P90", "Prune P80", "Prune P70"],
    ["Uncompressed", "Distill Small", "Distill Medium", "Distill Large"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average pose technique mAP50-95 vs latency.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to the model summary CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path.")
    parser.add_argument(
        "--series",
        choices=["11", "26", "all"],
        default="all",
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

    if "/input_reduction/" not in location_text:
        return None

    model_match = re.search(r"_(960|768|640)\.(?:pt|engine)$", model_text)
    if model_match:
        return model_match.group(1)

    location_match = re.search(r"/input_reduction/(960|768|640)(?:/|$)", location_text)
    if location_match:
        return location_match.group(1)

    return None


def infer_stage(model: object, location: object) -> str:
    input_stage = infer_input_stage(model, location)
    if input_stage is not None:
        return input_stage

    model_text = normalize(model)
    location_text = normalize(location)
    combined = f"{model_text} {location_text}"

    has_distill = any(token in combined for token in DISTILL_TOKENS)
    has_pruning = "pruning" in combined or bool(re.search(r"_p\d+", model_text))
    has_quant = "quantization" in combined or "fp16" in combined or "int8" in combined
    has_baseline = "/baseline/" in location_text or "baseline" in model_text

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


def infer_pruning_ratio(model: object, location: object) -> Optional[str]:
    model_text = normalize(model)
    location_text = normalize(location)

    model_match = re.search(r"_p(70|80|90)(?:_|\.|$)", model_text)
    if model_match:
        return model_match.group(1)

    location_match = re.search(r"/pruning(?:_quantization)?/(70|80|90)(?:/|$)", location_text)
    if location_match:
        return location_match.group(1)

    return None


def infer_distill_jump_group(model: object, family: str) -> Optional[str]:
    model_text = normalize(model)
    source_match = re.search(r"from_(\d+[a-z])", model_text)
    if source_match is None:
        return None

    source_family = source_match.group(1)
    if source_family[:2] != family[:2]:
        return None

    source_index = SIZE_INDEX.get(source_family[-1])
    family_index = SIZE_INDEX.get(family[-1])
    if source_index is None or family_index is None:
        return None

    jump = source_index - family_index
    if jump == 1:
        return "small"
    if jump == 2:
        return "medium"
    if jump >= 3:
        return "large"
    return None


def is_ds3(model: object, location: object) -> bool:
    return "ds3" in normalize(model) or "/ds3" in normalize(location)


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
                    "pruning_ratio": infer_pruning_ratio(row.get("Model", ""), row.get("Location", "")),
                    "distill_jump_group": infer_distill_jump_group(row.get("Model", ""), family),
                    MAP_COLUMN: map_value,
                    LATENCY_COLUMN: latency_value,
                }
            )

    if not usable:
        raise ValueError("No matching pose rows contain numeric mAP50-95 and latency values.")

    return usable


def filtered_rows(rows: list[dict[str, object]], predicate) -> list[dict[str, object]]:
    return [row for row in rows if predicate(row)]


def average_absolute_point(group_rows: list[dict[str, object]]) -> tuple[float, float, int]:
    latencies = [float(row[LATENCY_COLUMN]) for row in group_rows]
    map_values = [float(row[MAP_COLUMN]) for row in group_rows]
    return (
        sum(latencies) / len(latencies),
        sum(map_values) / len(map_values),
        len(group_rows),
    )


def build_baseline_lookup(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return build_baseline_lookup_for_artifact(rows, artifact="engine")


def build_baseline_lookup_for_artifact(
    rows: list[dict[str, object]],
    *,
    artifact: str,
) -> dict[str, dict[str, object]]:
    baseline_rows = filtered_rows(
        rows,
        lambda row: str(row["stage"]) == "baseline"
        and str(row["artifact"]) == artifact
        and str(row["quant_mode"]) == "none",
    )
    baseline_lookup: dict[str, dict[str, object]] = {}

    for row in baseline_rows:
        family = str(row["family"])
        if family in baseline_lookup:
            raise ValueError(f"Duplicate baseline .{artifact} rows found for family: {family}")
        baseline_lookup[family] = row

    if not baseline_lookup:
        raise ValueError(f"Missing baseline .{artifact} rows for technique normalization.")

    return baseline_lookup


def average_matched_delta_point(
    group_rows: list[dict[str, object]],
    baseline_lookup: dict[str, dict[str, object]],
    baseline_anchor: tuple[float, float, int],
) -> tuple[float, float, int]:
    map_deltas: list[float] = []
    latency_reductions: list[float] = []

    for row in group_rows:
        family = str(row["family"])
        baseline_row = baseline_lookup.get(family)
        if baseline_row is None:
            raise ValueError(f"Missing baseline .engine row for family: {family}")

        baseline_map = float(baseline_row[MAP_COLUMN])
        baseline_latency = float(baseline_row[LATENCY_COLUMN])
        if baseline_map <= 0.0 or baseline_latency <= 0.0:
            raise ValueError(f"Baseline .engine metrics must be positive for family: {family}")

        map_deltas.append((float(row[MAP_COLUMN]) - baseline_map) / baseline_map)
        latency_reductions.append(1.0 - (float(row[LATENCY_COLUMN]) / baseline_latency))

    baseline_latency, baseline_map, _ = baseline_anchor
    avg_map_delta = sum(map_deltas) / len(map_deltas)
    avg_latency_reduction = sum(latency_reductions) / len(latency_reductions)

    return (
        baseline_latency * (1.0 - avg_latency_reduction),
        baseline_map * (1.0 + avg_map_delta),
        len(group_rows),
    )


def build_points(rows: list[dict[str, object]]) -> dict[str, tuple[float, float, int]]:
    points: dict[str, tuple[float, float, int]] = {}
    engine_baseline_lookup = build_baseline_lookup_for_artifact(rows, artifact="engine")
    pt_baseline_lookup = build_baseline_lookup_for_artifact(rows, artifact="pt")

    uncompressed_rows = filtered_rows(
        rows,
        lambda row: str(row["stage"]) == "baseline"
        and str(row["artifact"]) == "pt"
        and str(row["quant_mode"]) == "none",
    )
    baseline_engine_rows = filtered_rows(
        rows,
        lambda row: str(row["stage"]) == "baseline"
        and str(row["artifact"]) == "engine"
        and str(row["quant_mode"]) == "none",
    )
    if not uncompressed_rows or not baseline_engine_rows:
        raise ValueError("Missing uncompressed or accelerated anchor rows.")

    points["Uncompressed"] = average_absolute_point(uncompressed_rows)
    points["Baseline engine"] = average_absolute_point(baseline_engine_rows)
    engine_anchor = points["Baseline engine"]
    pt_anchor = points["Uncompressed"]

    technique_filters = {
        "Input 960": lambda row: str(row["stage"]) == "960" and str(row["artifact"]) == "engine",
        "Input 768": lambda row: str(row["stage"]) == "768" and str(row["artifact"]) == "engine",
        "Input 640": lambda row: str(row["stage"]) == "640" and str(row["artifact"]) == "engine",
        "Quant FP16": lambda row: str(row["stage"]) == "quantized"
        and str(row["artifact"]) == "engine"
        and str(row["quant_mode"]) == "fp16",
        "Quant INT8": lambda row: str(row["stage"]) == "quantized"
        and str(row["artifact"]) == "engine"
        and str(row["quant_mode"]) == "int8",
        "Prune P90": lambda row: str(row["stage"]) == "pruned"
        and str(row["artifact"]) == "pt"
        and str(row["pruning_ratio"]) == "90",
        "Prune P80": lambda row: str(row["stage"]) == "pruned"
        and str(row["artifact"]) == "pt"
        and str(row["pruning_ratio"]) == "80",
        "Prune P70": lambda row: str(row["stage"]) == "pruned"
        and str(row["artifact"]) == "pt"
        and str(row["pruning_ratio"]) == "70",
        "Distill Small": lambda row: str(row["stage"]) == "distilled"
        and str(row["artifact"]) == "pt"
        and str(row["quant_mode"]) == "none"
        and str(row["distill_jump_group"]) == "small",
        "Distill Medium": lambda row: str(row["stage"]) == "distilled"
        and str(row["artifact"]) == "pt"
        and str(row["quant_mode"]) == "none"
        and str(row["distill_jump_group"]) == "medium",
        "Distill Large": lambda row: str(row["stage"]) == "distilled"
        and str(row["artifact"]) == "pt"
        and str(row["quant_mode"]) == "none"
        and str(row["distill_jump_group"]) == "large",
    }

    for key in POINT_ORDER[2:]:
        group_rows = filtered_rows(rows, technique_filters[key])
        if not group_rows:
            raise ValueError(f"Missing rows for plot point: {key}")
        baseline_lookup = engine_baseline_lookup
        baseline_anchor = engine_anchor
        if key.startswith("Prune") or key.startswith("Distill"):
            baseline_lookup = pt_baseline_lookup
            baseline_anchor = pt_anchor

        points[key] = average_matched_delta_point(group_rows, baseline_lookup, baseline_anchor)

    return points


def group_key(label: str) -> str:
    if label in {"Uncompressed", "Baseline engine"}:
        return "Baseline"
    if label.startswith("Input"):
        return "Input reduction"
    if label.startswith("Quant"):
        return "Quantisation"
    if label.startswith("Prune"):
        return "Pruning"
    return "Distillation"


def marker_key(label: str) -> str:
    if label == "Uncompressed":
        return "base"
    if label == "Baseline engine":
        return "accelerated"
    if label in {"Input 960", "Quant FP16", "Prune P90", "Distill Small"}:
        return "first"
    if label in {"Input 768", "Quant INT8", "Prune P80", "Distill Medium"}:
        return "second"
    return "third"


def build_figure(points: dict[str, tuple[float, float, int]]):
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

    fig, ax = plt.subplots(figsize=(4.3, 3.75), dpi=300)

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
            linewidth=1.3,
            alpha=0.82,
            zorder=1,
        )

    for label in POINT_ORDER:
        latency, map_value, _ = points[label]
        color = GROUP_COLORS[group_key(label)]
        marker = MARKERS[marker_key(label)]
        size = 64 if label in {"Uncompressed", "Baseline engine"} else 58
        ax.scatter(
            latency,
            map_value,
            color=color,
            marker=marker,
            s=size,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

        if label in ANNOTATIONS:
            text, offset = ANNOTATIONS[label]
            ax.annotate(
                text,
                xy=(latency, map_value),
                xytext=offset,
                textcoords="offset points",
                ha="center" if offset[0] == 0 else ("left" if offset[0] > 0 else "right"),
                va="bottom" if offset[1] > 0 else "top",
                fontsize=6.9,
                color=color,
                weight="bold",
                zorder=4,
            )

    x_min = min(all_latencies)
    x_max = max(all_latencies)
    y_min = min(all_maps)
    y_max = max(all_maps)
    x_pad = max(2.5, (x_max - x_min) * 0.08)
    y_pad = max(0.008, (y_max - y_min) * 0.16)

    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))
    ax.set_xlabel("Average latency (ms)", fontsize=10)
    ax.set_ylabel("Average mAP50-95", fontsize=10)
    ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], color=GROUP_COLORS["Input reduction"], linewidth=1.6, label="Input reduction"),
        Line2D([0], [0], color=GROUP_COLORS["Quantisation"], linewidth=1.6, label="Quantisation"),
        Line2D([0], [0], color=GROUP_COLORS["Pruning"], linewidth=1.6, label="Pruning"),
        Line2D([0], [0], color=GROUP_COLORS["Distillation"], linewidth=1.6, label="Distillation"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=6.4,
        frameon=False,
        ncol=2,
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
            saved_path = prompt_save_figure(fig, default_name="pose_technique_summary_tradeoff")
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
