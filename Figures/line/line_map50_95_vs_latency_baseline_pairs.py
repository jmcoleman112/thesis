#!/usr/bin/env python
"""
Line plot of mAP50-95 vs latency for baseline to fp32 pairs by size.

The figure shows two side-by-side plots:

- Object models
- Pose models

Within each plot, the X/L/M/S/N lines average the corresponding 11- and 26-series
baseline rows for that size and artifact.

Run:
  python Figures/line/line_map50_95_vs_latency_baseline_pairs.py
  python Figures/line/line_map50_95_vs_latency_baseline_pairs.py --no-show
  python Figures/line/line_map50_95_vs_latency_baseline_pairs.py --output Figures/produced_images/map50_95_vs_latency_baseline_pairs.png
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

SIZE_ORDER = ["x", "l", "m", "s", "n"]
DOMAIN_ORDER = ["object", "pose"]

OBJECT_BASELINE_PT_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_baseline\.pt$", re.IGNORECASE)
OBJECT_BASELINE_ENGINE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_ds3_baseline\.engine$", re.IGNORECASE)
POSE_BASELINE_PT_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_baseline\.pt$", re.IGNORECASE)
POSE_BASELINE_ENGINE_RE = re.compile(r"^(?P<family>(11|26)[nsmlx])_pose_baseline\.engine$", re.IGNORECASE)

DOMAIN_TITLES = {
    "object": "Object Baseline vs FP32",
    "pose": "Pose Baseline vs FP32",
}

SIZE_LABELS = {
    "x": "X",
    "l": "L",
    "m": "M",
    "s": "S",
    "n": "N",
}

SIZE_COLORS = {
    "x": "#9467bd",
    "l": "#d62728",
    "m": "#2ca02c",
    "s": "#ff7f0e",
    "n": "#1f77b4",
}

STAGE_ORDER = ["pt", "fp32"]

STAGE_STYLES = {
    "pt": {
        "label": "Uncompressed",
        "marker": "o",
    },
    "fp32": {
        "label": "FP32",
        "marker": "s",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mAP50-95 vs latency for baseline .pt to fp32 model pairs by size."
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
            "series": family[:2],
            "size": family[-1],
            "domain": "object",
            "artifact": "pt",
        }

    match = OBJECT_BASELINE_ENGINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "series": family[:2],
            "size": family[-1],
            "domain": "object",
            "artifact": "fp32",
        }

    match = POSE_BASELINE_PT_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "series": family[:2],
            "size": family[-1],
            "domain": "pose",
            "artifact": "pt",
        }

    match = POSE_BASELINE_ENGINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "series": family[:2],
            "size": family[-1],
            "domain": "pose",
            "artifact": "fp32",
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
                }
            )

    if not usable:
        raise ValueError("No matching baseline/fp32 rows contain numeric mAP50-95 and latency values.")

    return usable


def build_domain_size_rows(
    rows: list[dict[str, object]]
) -> dict[str, dict[str, dict[str, list[dict[str, object]]]]]:
    grouped: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {
        domain: {size: {artifact: [] for artifact in STAGE_ORDER} for size in SIZE_ORDER} for domain in DOMAIN_ORDER
    }

    for row in rows:
        domain = str(row["domain"])
        size = str(row["size"])
        artifact = str(row["artifact"])
        if domain in grouped and size in grouped[domain] and artifact in grouped[domain][size]:
            grouped[domain][size][artifact].append(row)

    missing: list[str] = []
    for domain in DOMAIN_ORDER:
        for size in SIZE_ORDER:
            for artifact in STAGE_ORDER:
                if not grouped[domain][size][artifact]:
                    missing.append(f"{domain} {SIZE_LABELS[size]} ({artifact})")

    if missing:
        raise ValueError("Missing one or more required size/domain baseline groups: " + ", ".join(missing))

    return grouped


def build_domain_size_averages(
    grouped_rows: dict[str, dict[str, dict[str, list[dict[str, object]]]]]
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    averages: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}

    for domain, size_rows in grouped_rows.items():
        averages[domain] = {}
        for size, artifact_rows in size_rows.items():
            averages[domain][size] = {}
            for artifact in STAGE_ORDER:
                latencies = [float(row[LATENCY_COLUMN]) for row in artifact_rows[artifact]]
                map_values = [float(row[MAP_COLUMN]) for row in artifact_rows[artifact]]
                averages[domain][size][artifact] = (
                    sum(latencies) / len(latencies),
                    sum(map_values) / len(map_values),
                )

    return averages


def print_group_summary(grouped_rows: dict[str, dict[str, dict[str, list[dict[str, object]]]]]) -> None:
    for domain in DOMAIN_ORDER:
        size_summaries = []
        for size in SIZE_ORDER:
            counts = ", ".join(f"{artifact}={len(grouped_rows[domain][size][artifact])}" for artifact in STAGE_ORDER)
            size_summaries.append(f"{SIZE_LABELS[size]}({counts})")
        print(f"- {DOMAIN_TITLES[domain]}: {'; '.join(size_summaries)}")


def build_figure(domain_size_averages: dict[str, dict[str, dict[str, tuple[float, float]]]]):
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

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN * 0.50), sharey=False)

    for index, domain in enumerate(DOMAIN_ORDER):
        ax = axes[index]

        for size in SIZE_ORDER:
            size_color = SIZE_COLORS[size]
            latencies = [domain_size_averages[domain][size][artifact][0] for artifact in STAGE_ORDER]
            map_values = [domain_size_averages[domain][size][artifact][1] for artifact in STAGE_ORDER]

            ax.plot(
                latencies,
                map_values,
                color=size_color,
                linewidth=1.1,
                zorder=1,
            )

            for artifact in STAGE_ORDER:
                stage_latency, stage_map = domain_size_averages[domain][size][artifact]
                ax.scatter(
                    stage_latency,
                    stage_map,
                    color=size_color,
                    marker=STAGE_STYLES[artifact]["marker"],
                    s=72,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )

                annotate_point = artifact == "pt"
                x_offset = -7
                y_offset = 6
                horizontal_alignment = "right"
                vertical_alignment = "bottom"

                if domain == "object" and size == "l" and artifact == "pt":
                    x_offset = 7
                    y_offset = 0
                    horizontal_alignment = "left"
                    vertical_alignment = "center"
                elif domain == "object" and size == "m":
                    annotate_point = artifact == "fp32"
                    x_offset = -7
                    y_offset = 0
                    horizontal_alignment = "right"
                    vertical_alignment = "center"
                elif domain == "pose" and size in {"s", "l"} and artifact == "pt":
                    x_offset = 0
                    y_offset = -8
                    horizontal_alignment = "center"
                    vertical_alignment = "top"

                if annotate_point:
                    ax.annotate(
                        SIZE_LABELS[size],
                        xy=(stage_latency, stage_map),
                        xytext=(x_offset, y_offset),
                        textcoords="offset points",
                        ha=horizontal_alignment,
                        va=vertical_alignment,
                        fontsize=9,
                        color=size_color,
                        weight="bold",
                        zorder=4,
                    )

        ax.set_title(DOMAIN_TITLES[domain], fontsize=10.5)
        ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Latency ms", fontsize=11)
        ax.set_ylabel(MAP_COLUMN, fontsize=11)

    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=STAGE_STYLES[artifact]["marker"],
            color="black",
            linestyle="None",
            markersize=7,
            label=STAGE_STYLES[artifact]["label"],
        )
        for artifact in STAGE_ORDER
    ]

    fig.legend(
        handles=stage_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
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
            print(
                f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.",
                file=sys.stderr,
            )
            return 1

        matplotlib.use("Agg")

    try:
        rows = load_plot_rows(args.csv)
        grouped_rows = build_domain_size_rows(rows)
        domain_size_averages = build_domain_size_averages(grouped_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_group_summary(grouped_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.",
            file=sys.stderr,
        )
        return 1

    fig = build_figure(domain_size_averages)

    saved_path: Optional[Path] = None
    try:
        if args.output is not None:
            saved_path = save_figure(fig, args.output)
            print(f"Saved figure to {saved_path}")
        elif not args.no_show:
            plt.show()
            saved_path = prompt_save_figure(
                fig,
                default_name="map50_95_vs_latency_baseline_pairs",
            )
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
