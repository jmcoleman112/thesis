#!/usr/bin/env python
"""
Horizontal comparison of knowledge-distillation chains for n- and s-model students using .engine rows.

The figure shows two side-by-side plots:

- n-student chains
- s-student chains

Each plot overlays three group lines:

- YOLO26 object
- YOLO11 pose
- YOLO26 pose

Run:
  python Figures/line/line_map50_95_vs_latency_knowledge_n_chains.py
  python Figures/line/line_map50_95_vs_latency_knowledge_n_chains.py --no-show
  python Figures/line/line_map50_95_vs_latency_knowledge_n_chains.py --output Figures/produced_images/map50_95_vs_latency_knowledge_n_chains.png
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

TARGET_ORDER = ["n", "s"]

OBJECT_BASELINE_RE = re.compile(r"^(?P<family>(11|26)[ns])_ds3_baseline\.engine$", re.IGNORECASE)
OBJECT_DISTILL_RE = re.compile(r"^(?P<family>(11|26)[ns])_ds3_from_(?P<src>(11|26)[slmx])\.engine$", re.IGNORECASE)
POSE_BASELINE_RE = re.compile(r"^(?P<family>(11|26)[ns])_pose_baseline\.engine$", re.IGNORECASE)
POSE_DISTILL_RE = re.compile(r"^(?P<family>(11|26)[ns])_pose_from_(?P<src>(11|26)[slmx])\.engine$", re.IGNORECASE)

GROUP_ORDER = [
    ("11", "object"),
    ("26", "object"),
    ("11", "pose"),
    ("26", "pose"),
]

GROUP_LABELS = {
    ("26", "object"): "YOLO26 Obj",
    ("11", "object"): "YOLO11 Obj",
    ("11", "pose"): "YOLO11 KP",
    ("26", "pose"): "YOLO26 KP",
}

GROUP_COLORS = {
    ("11", "object"): "#1f77b4",
    ("26", "object"): "#2ca02c",
    ("11", "pose"): "#ff7f0e",
    ("26", "pose"): "#d62728",
}

SUBPLOT_TITLES = {
    "n": "n-Student Chains",
    "s": "s-Student Chains",
}

STAGE_ORDER_BY_TARGET = {
    "n": ["baseline", "x", "l", "m", "s"],
    "s": ["baseline", "x", "l", "m"],
}

STAGE_STYLES = {
    "baseline": {"label": "Baseline", "marker": "o"},
    "x": {"label": "From X", "marker": "D"},
    "l": {"label": "From L", "marker": "s"},
    "m": {"label": "From M", "marker": "^"},
    "s": {"label": "From S", "marker": "P"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mAP50-95 vs latency for 11/26 n- and s-student distillation .engine chains with baselines."
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
        return {
            "family": family,
            "target": family[-1],
            "domain": "object",
            "stage": "baseline",
        }

    match = OBJECT_DISTILL_RE.match(text)
    if match:
        family = match.group("family").lower()
        source_family = match.group("src").lower()
        if family[:2] != source_family[:2]:
            return None
        return {
            "family": family,
            "target": family[-1],
            "domain": "object",
            "stage": source_family[-1],
        }

    match = POSE_BASELINE_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "target": family[-1],
            "domain": "pose",
            "stage": "baseline",
        }

    match = POSE_DISTILL_RE.match(text)
    if match:
        family = match.group("family").lower()
        source_family = match.group("src").lower()
        if family[:2] != source_family[:2]:
            return None
        return {
            "family": family,
            "target": family[-1],
            "domain": "pose",
            "stage": source_family[-1],
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

            usable.append({**row, **parsed, MAP_COLUMN: map_value, LATENCY_COLUMN: latency_value})

    if not usable:
        raise ValueError("No matching distillation .engine rows contain numeric mAP50-95 and latency values.")

    return usable


def build_group_rows(
    rows: list[dict[str, object]]
) -> dict[str, dict[tuple[str, str], dict[str, dict[str, object]]]]:
    grouped: dict[str, dict[tuple[str, str], dict[str, dict[str, object]]]] = {
        target: {group_key: {} for group_key in GROUP_ORDER} for target in TARGET_ORDER
    }

    for row in rows:
        target = str(row["target"])
        group_key = (str(row["family"])[:2], str(row["domain"]))
        stage = str(row["stage"])
        if target in grouped and group_key in grouped[target] and stage in STAGE_STYLES:
            grouped[target][group_key][stage] = row

    missing_baselines: list[str] = []
    for target in TARGET_ORDER:
        for group_key in GROUP_ORDER:
            if "baseline" not in grouped[target][group_key]:
                missing_baselines.append(f"{target} / {GROUP_LABELS[group_key]}")

    if missing_baselines:
        raise ValueError("Missing one or more required distillation baselines: " + ", ".join(missing_baselines))

    return grouped


def print_group_summary(grouped_rows: dict[str, dict[tuple[str, str], dict[str, dict[str, object]]]]) -> None:
    for target in TARGET_ORDER:
        for group_key in GROUP_ORDER:
            stages = [
                STAGE_STYLES[stage]["label"]
                for stage in STAGE_ORDER_BY_TARGET[target]
                if stage in grouped_rows[target][group_key]
            ]
            print(f"- {target} / {GROUP_LABELS[group_key]}: {', '.join(stages)}")


def build_figure(grouped_rows: dict[str, dict[tuple[str, str], dict[str, dict[str, object]]]]):
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

    label_offsets = {
        ("11", "object"): (8, -12),
        ("26", "object"): (8, -12),
        ("11", "pose"): (8, -12),
        ("26", "pose"): (8, 10),
    }

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN * 0.56), sharey=True)

    for index, target in enumerate(TARGET_ORDER):
        ax = axes[index]

        for group_key in GROUP_ORDER:
            available_stages = [
                stage for stage in STAGE_ORDER_BY_TARGET[target] if stage in grouped_rows[target][group_key]
            ]
            if len(available_stages) < 2:
                continue

            group_color = GROUP_COLORS[group_key]
            latencies = [float(grouped_rows[target][group_key][stage][LATENCY_COLUMN]) for stage in available_stages]
            map_values = [float(grouped_rows[target][group_key][stage][MAP_COLUMN]) for stage in available_stages]

            ax.plot(latencies, map_values, color=group_color, linewidth=1.2, zorder=1)

            for stage in available_stages:
                stage_latency = float(grouped_rows[target][group_key][stage][LATENCY_COLUMN])
                stage_map = float(grouped_rows[target][group_key][stage][MAP_COLUMN])
                ax.scatter(
                    stage_latency,
                    stage_map,
                    color=group_color,
                    marker=STAGE_STYLES[stage]["marker"],
                    s=82,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )

            end_stage = available_stages[-1]
            end_latency = float(grouped_rows[target][group_key][end_stage][LATENCY_COLUMN])
            end_map = float(grouped_rows[target][group_key][end_stage][MAP_COLUMN])
            offset_x, offset_y = label_offsets[group_key]
            ax.annotate(
                GROUP_LABELS[group_key],
                xy=(end_latency, end_map),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                color=group_color,
                fontsize=9.5,
                fontweight="bold",
                ha="left",
                va="center",
            )

        ax.set_title(SUBPLOT_TITLES[target], fontsize=11)
        ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Latency ms", fontsize=11)
        if index == 0:
            ax.set_ylabel(MAP_COLUMN, fontsize=11)

    legend_stages = ["baseline", "x", "l", "m", "s"]
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=STAGE_STYLES[stage]["marker"],
            color="black",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=STAGE_STYLES[stage]["label"],
        )
        for stage in legend_stages
    ]

    fig.legend(
        handles=stage_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
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
            saved_path = prompt_save_figure(fig, default_name="map50_95_vs_latency_knowledge_ns_chains")
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
