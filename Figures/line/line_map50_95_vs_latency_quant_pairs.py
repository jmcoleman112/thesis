#!/usr/bin/env python
"""
Line plot of mAP50-95 vs latency for baseline to fp32 to int8 pairs.

One combined plot shows four model family/task chains:

- 11 DS3 object baseline .pt -> fp32 -> int8
- 26 DS3 object baseline .pt -> fp32 -> int8
- 11 pose baseline .pt -> fp32 -> int8
- 26 pose baseline .pt -> fp32 -> int8

Run:
  python Figures/line/line_map50_95_vs_latency_quant_pairs.py
  python Figures/line/line_map50_95_vs_latency_quant_pairs.py --no-show
  python Figures/line/line_map50_95_vs_latency_quant_pairs.py --output Figures/produced_images/map50_95_vs_latency_quant_pairs.png
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

OBJECT_BASELINE_PT_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_baseline\.pt$", re.IGNORECASE)
OBJECT_BASELINE_ENGINE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_baseline\.engine$", re.IGNORECASE)
OBJECT_INT8_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_int8\.engine$", re.IGNORECASE)
POSE_BASELINE_PT_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_baseline\.pt$", re.IGNORECASE)
POSE_BASELINE_ENGINE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_baseline\.engine$", re.IGNORECASE)
POSE_INT8_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_int8\.engine$", re.IGNORECASE)

SUBPLOT_ORDER = [
    ("11", "object"),
    ("26", "object"),
    ("11", "pose"),
    ("26", "pose"),
]

SUBPLOT_TITLES = {
    ("11", "object"): "11 DS3 Object Quantization",
    ("26", "object"): "26 DS3 Object Quantization",
    ("11", "pose"): "11 Pose Quantization",
    ("26", "pose"): "26 Pose Quantization",
}

GROUP_LABELS = {
    ("11", "object"): "YOLO11 Obj",
    ("26", "object"): "YOLO26 Obj",
    ("11", "pose"): "YOLO11 KP",
    ("26", "pose"): "YOLO26 KP",
}

PANEL_COLORS = {
    ("11", "object"): "#1f77b4",
    ("26", "object"): "#2ca02c",
    ("11", "pose"): "#ff7f0e",
    ("26", "pose"): "#d62728",
}

POINT_STYLES = {
    "pt": {
        "label": "baseline",
        "marker": "o",
        "linestyle": "None",
    },
    "fp32": {
        "label": "FP32",
        "marker": "s",
        "linestyle": "None",
    },
    "int8": {
        "label": "INT8",
        "marker": "D",
        "linestyle": "None",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mAP50-95 vs latency for baseline .pt to fp32 to int8 model pairs."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to the model summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional PNG output path. If omitted, the save dialog is used in interactive mode.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the figure without opening a plot window.",
    )
    return parser.parse_args()


def parse_model(model: str) -> Optional[dict[str, str]]:
    text = str(model).strip()
    if not text:
        return None

    match = OBJECT_BASELINE_PT_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "size": family[-1],
            "domain": "object",
            "artifact": "pt",
        }

    match = OBJECT_BASELINE_ENGINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "size": family[-1],
            "domain": "object",
            "artifact": "fp32",
        }

    match = OBJECT_INT8_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "size": family[-1],
            "domain": "object",
            "artifact": "int8",
        }

    match = POSE_BASELINE_PT_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "size": family[-1],
            "domain": "pose",
            "artifact": "pt",
        }

    match = POSE_BASELINE_ENGINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "size": family[-1],
            "domain": "pose",
            "artifact": "fp32",
        }

    match = POSE_INT8_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "size": family[-1],
            "domain": "pose",
            "artifact": "int8",
        }

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

            usable.append(
                {
                    **row,
                    **parsed,
                    MAP_COLUMN: map_value,
                    LATENCY_COLUMN: latency_value,
                    "size_order": SIZE_INDEX.get(parsed["size"], 999),
                }
            )

    if not usable:
        raise ValueError("No matching rows contain numeric mAP50-95 and latency values.")

    return usable


def build_subplot_triplets(
    rows: list[dict[str, object]]
) -> dict[tuple[str, str], list[dict[str, dict[str, object]]]]:
    row_lookup = {
        (str(row["family"]), str(row["domain"]), str(row["artifact"])): row for row in rows
    }

    subplot_triplets: dict[tuple[str, str], list[dict[str, dict[str, object]]]] = {
        subplot_key: [] for subplot_key in SUBPLOT_ORDER
    }
    missing_rows: list[str] = []

    for family_prefix, domain in SUBPLOT_ORDER:
        for size in SIZE_ORDER:
            family = f"{family_prefix}{size}"
            pt_row = row_lookup.get((family, domain, "pt"))
            fp32_row = row_lookup.get((family, domain, "fp32"))
            int8_row = row_lookup.get((family, domain, "int8"))

            if pt_row is None or fp32_row is None or int8_row is None:
                missing_artifacts: list[str] = []
                if pt_row is None:
                    missing_artifacts.append(".pt")
                if fp32_row is None:
                    missing_artifacts.append("fp32")
                if int8_row is None:
                    missing_artifacts.append("int8")
                missing_rows.append(
                    f"{family.upper()} {domain} ({', '.join(missing_artifacts)})"
                )
                continue

            subplot_triplets[(family_prefix, domain)].append(
                {
                    "pt": pt_row,
                    "fp32": fp32_row,
                    "int8": int8_row,
                }
            )

    if missing_rows:
        raise ValueError("Missing one or more quantization rows: " + ", ".join(missing_rows))

    return subplot_triplets


def print_triplet_summary(
    subplot_triplets: dict[tuple[str, str], list[dict[str, dict[str, object]]]]
) -> None:
    total_triplets = sum(len(groups) for groups in subplot_triplets.values())
    print(f"Prepared {total_triplets} .pt/fp32/int8 triplets across {len(subplot_triplets)} subplots.")
    for subplot_key in SUBPLOT_ORDER:
        sizes = ", ".join(
            str(group["pt"]["family"]).upper() for group in subplot_triplets[subplot_key]
        )
        print(f"- {SUBPLOT_TITLES[subplot_key]}: {sizes}")


def build_subplot_averages(
    subplot_triplets: dict[tuple[str, str], list[dict[str, dict[str, object]]]]
) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    subplot_averages: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}

    for subplot_key, groups in subplot_triplets.items():
        artifact_means: dict[str, tuple[float, float]] = {}
        for artifact in ("pt", "fp32", "int8"):
            latencies = [float(group[artifact][LATENCY_COLUMN]) for group in groups]
            map_values = [float(group[artifact][MAP_COLUMN]) for group in groups]
            artifact_means[artifact] = (
                sum(latencies) / len(latencies),
                sum(map_values) / len(map_values),
            )
        subplot_averages[subplot_key] = artifact_means

    return subplot_averages


def build_figure(subplot_triplets: dict[tuple[str, str], list[dict[str, dict[str, object]]]]):
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
    subplot_averages = build_subplot_averages(subplot_triplets)

    for subplot_key in SUBPLOT_ORDER:
        panel_color = PANEL_COLORS[subplot_key]
        pt_latency, pt_map = subplot_averages[subplot_key]["pt"]
        fp32_latency, fp32_map = subplot_averages[subplot_key]["fp32"]
        int8_latency, int8_map = subplot_averages[subplot_key]["int8"]

        ax.plot(
            [pt_latency, fp32_latency, int8_latency],
            [pt_map, fp32_map, int8_map],
            color=panel_color,
            linewidth=1.2,
            zorder=1,
        )

        ax.scatter(
            pt_latency,
            pt_map,
            color=panel_color,
            marker=POINT_STYLES["pt"]["marker"],
            s=82,
            zorder=3,
        )
        ax.scatter(
            fp32_latency,
            fp32_map,
            color=panel_color,
            marker=POINT_STYLES["fp32"]["marker"],
            s=82,
            zorder=3,
        )
        ax.scatter(
            int8_latency,
            int8_map,
            color=panel_color,
            marker=POINT_STYLES["int8"]["marker"],
            s=82,
            zorder=3,
        )

    ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Latency ms", fontsize=11)
    ax.set_ylabel(MAP_COLUMN, fontsize=11)

    group_handles = [
        Line2D(
            [0],
            [0],
            color=PANEL_COLORS[subplot_key],
            linewidth=1.6,
            label=GROUP_LABELS[subplot_key],
        )
        for subplot_key in SUBPLOT_ORDER
    ]
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=POINT_STYLES[artifact]["marker"],
            color="black",
            linestyle="None",
            markersize=7,
            label=POINT_STYLES[artifact]["label"],
        )
        for artifact in ("pt", "fp32", "int8")
    ]

    ax.legend(
        handles=group_handles + stage_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=7,
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
        columnspacing=1.0,
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
            print(
                f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.",
                file=sys.stderr,
            )
            return 1

        matplotlib.use("Agg")

    try:
        rows = load_plot_rows(args.csv)
        subplot_triplets = build_subplot_triplets(rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_triplet_summary(subplot_triplets)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.",
            file=sys.stderr,
        )
        return 1

    fig = build_figure(subplot_triplets)

    saved_path: Optional[Path] = None
    try:
        if args.output is not None:
            saved_path = save_figure(fig, args.output)
            print(f"Saved figure to {saved_path}")
        elif not args.no_show:
            plt.show()
            saved_path = prompt_save_figure(
                fig,
                default_name="map50_95_vs_latency_quant_pairs",
            )
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
