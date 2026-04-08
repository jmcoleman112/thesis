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

DEFAULT_MAP_COLUMN = "mAP50-95"
POSE_VALIDATION2_MAP_COLUMN = "Validation2 mAP50-95"
MAP_COLUMN = DEFAULT_MAP_COLUMN
LATENCY_COLUMN = "Latency ms"
REQUIRED_COLUMNS = ["Model", "Location", MAP_COLUMN, LATENCY_COLUMN]
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FIGURE_WIDTH_IN = 7.1
FIGURE_HEIGHT_IN = 6.35
TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 12.5
LEGEND_FONT_SIZE = 12
POSE_ROW_Y_MIN = 0.50
ANNOTATIONS = {
    ("11", "pose", "m", "int8"): {
        "target_xy": (150, 0.506),
        "text_xy": (140, 0.62),
        "ha": "left",
    },
    ("26", "pose", "m", "int8"): {
        "target_xy": (175, 0.506),
        "text_xy": (150, 0.6),
        "ha": "left",
    },
}

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
    ("11", "object"): "YOLO11 Object",
    ("26", "object"): "YOLO26 Object",
    ("11", "pose"): "YOLO11 Pose",
    ("26", "pose"): "YOLO26 Pose",
}

SIZE_LABELS = {
    "x": "X",
    "l": "L",
    "m": "M",
    "s": "S",
    "n": "N",
}

SIZE_COLORS = {
    "n": "#1f77b4",
    "s": "#ff7f0e",
    "m": "#2ca02c",
    "l": "#d62728",
    "x": "#9467bd",
}

STAGE_ORDER = ["pt", "fp32", "int8"]

STAGE_STYLES = {
    "pt": {
        "label": "Uncompressed",
        "marker": "o",
    },
    "fp32": {
        "label": "Accel.",
        "marker": "s",
    },
    "int8": {
        "label": "INT8",
        "marker": "D",
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

    match = OBJECT_INT8_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "series": family[:2],
            "size": family[-1],
            "domain": "object",
            "artifact": "int8",
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

    match = POSE_INT8_RE.match(text)
    if match:
        family = match.group("family").lower()
        return {
            "family": family,
            "series": family[:2],
            "size": family[-1],
            "domain": "pose",
            "artifact": "int8",
        }

    return None


def required_columns() -> list[str]:
    return ["Model", "Location", DEFAULT_MAP_COLUMN, POSE_VALIDATION2_MAP_COLUMN, LATENCY_COLUMN]


def load_plot_rows(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required_columns() if column not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        usable: list[dict[str, object]] = []
        for row in reader:
            parsed = parse_model(row.get("Model", ""))
            if parsed is None:
                continue

            map_column = POSE_VALIDATION2_MAP_COLUMN if parsed["domain"] == "pose" else DEFAULT_MAP_COLUMN
            try:
                map_value = float(str(row[map_column]).strip())
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


def build_panel_rows(
    rows: list[dict[str, object]]
) -> dict[tuple[str, str], dict[str, dict[str, list[dict[str, object]]]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, list[dict[str, object]]]]] = {
        panel: {size: {artifact: [] for artifact in STAGE_ORDER} for size in SIZE_ORDER}
        for panel in SUBPLOT_ORDER
    }

    for row in rows:
        panel = (str(row["series"]), str(row["domain"]))
        size = str(row["size"])
        artifact = str(row["artifact"])
        if panel in grouped and size in grouped[panel] and artifact in grouped[panel][size]:
            grouped[panel][size][artifact].append(row)

    missing: list[str] = []
    for series, domain in SUBPLOT_ORDER:
        panel = (series, domain)
        for size in SIZE_ORDER:
            for artifact in STAGE_ORDER:
                if not grouped[panel][size][artifact]:
                    missing.append(f"{SUBPLOT_TITLES[panel]} {SIZE_LABELS[size]} ({artifact})")

    if missing:
        raise ValueError("Missing one or more required quantization groups: " + ", ".join(missing))

    return grouped


def build_panel_size_values(
    grouped_rows: dict[tuple[str, str], dict[str, dict[str, list[dict[str, object]]]]]
) -> dict[tuple[str, str], dict[str, dict[str, tuple[float, float]]]]:
    values: dict[tuple[str, str], dict[str, dict[str, tuple[float, float]]]] = {}

    for panel, size_rows in grouped_rows.items():
        values[panel] = {}
        for size, artifact_rows in size_rows.items():
            values[panel][size] = {}
            for artifact in STAGE_ORDER:
                latencies = [float(row[LATENCY_COLUMN]) for row in artifact_rows[artifact]]
                map_values = [float(row[MAP_COLUMN]) for row in artifact_rows[artifact]]
                values[panel][size][artifact] = (
                    sum(latencies) / len(latencies),
                    sum(map_values) / len(map_values),
                )

    return values


def print_group_summary(
    grouped_rows: dict[tuple[str, str], dict[str, dict[str, list[dict[str, object]]]]]
) -> None:
    for panel in SUBPLOT_ORDER:
        size_summaries = []
        for size in SIZE_ORDER:
            counts = ", ".join(
                f"{artifact}={len(grouped_rows[panel][size][artifact])}" for artifact in STAGE_ORDER
            )
            size_summaries.append(f"{SIZE_LABELS[size]}({counts})")
        print(f"- {SUBPLOT_TITLES[panel]}: {'; '.join(size_summaries)}")


def build_figure(panel_size_values: dict[tuple[str, str], dict[str, dict[str, tuple[float, float]]]]):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), sharey="row")
    axes_map = {panel: axes.flat[index] for index, panel in enumerate(SUBPLOT_ORDER)}
    row_ys: dict[int, list[float]] = {0: [], 1: []}
    panel_xs: dict[tuple[str, str], list[float]] = {panel: [] for panel in SUBPLOT_ORDER}

    for panel in SUBPLOT_ORDER:
        ax = axes_map[panel]
        row_index = 0 if panel[1] == "object" else 1

        for size in SIZE_ORDER:
            size_color = SIZE_COLORS[size]
            latencies = [panel_size_values[panel][size][artifact][0] for artifact in STAGE_ORDER]
            map_values = [panel_size_values[panel][size][artifact][1] for artifact in STAGE_ORDER]
            panel_xs[panel].extend(latencies)
            row_ys[row_index].extend(map_values)

            ax.plot(
                latencies,
                map_values,
                color=size_color,
                linewidth=1.7,
                zorder=1,
            )

            for artifact in STAGE_ORDER:
                stage_latency, stage_map = panel_size_values[panel][size][artifact]
                ax.scatter(
                    stage_latency,
                    stage_map,
                    color=size_color,
                    marker=STAGE_STYLES[artifact]["marker"],
                    s=115,
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=3,
                )

                annotation = ANNOTATIONS.get((panel[0], panel[1], size, artifact))
                if annotation:
                    label = fr"INT8" "\n" fr"mAP$_{{50\mathrm{{-}}95}}$: {stage_map:.2f}"
                    text = ax.annotate(
                        label,
                        xy=tuple(annotation.get("target_xy", (stage_latency, stage_map))),
                        xytext=tuple(annotation.get("text_xy", (stage_latency, stage_map))),
                        textcoords="data",
                        ha=str(annotation["ha"]),
                        va="center",
                        fontsize=11,
                        color=size_color,
                        annotation_clip=False,
                        arrowprops={
                            "arrowstyle": "->",
                            "color": size_color,
                            "linewidth": 1.0,
                            "shrinkA": 0,
                            "shrinkB": 4,
                        },
                        zorder=5,
                    )
                    text.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

        ax.set_title(SUBPLOT_TITLES[panel], fontsize=TITLE_FONT_SIZE, pad=8)
        ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.tick_params(labelsize=TICK_FONT_SIZE, width=1.0, length=5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if panel_xs[panel]:
            x_min = min(panel_xs[panel])
            x_max = max(panel_xs[panel])
            x_pad = max((x_max - x_min) * 0.05, 0.9)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

    object_y_min = min(row_ys[0])
    object_y_max = max(row_ys[0])
    object_y_pad = max((object_y_max - object_y_min) * 0.10, 0.01)
    for ax in axes[0]:
        ax.set_ylim(object_y_min - object_y_pad, object_y_max + object_y_pad)

    pose_y_min = POSE_ROW_Y_MIN
    pose_y_max = max(row_ys[1])
    pose_y_pad = max((pose_y_max - pose_y_min) * 0.10, 0.01)
    for ax in axes[1]:
        ax.set_ylim(pose_y_min, pose_y_max + pose_y_pad)

    axes[0][0].set_ylabel(DEFAULT_MAP_COLUMN, fontsize=LABEL_FONT_SIZE)
    axes[1][0].set_ylabel(POSE_VALIDATION2_MAP_COLUMN, fontsize=LABEL_FONT_SIZE)
    axes[1][0].set_xlabel("Latency ms", fontsize=LABEL_FONT_SIZE)
    axes[1][1].set_xlabel("Latency ms", fontsize=LABEL_FONT_SIZE)

    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=STAGE_STYLES[artifact]["marker"],
            color="black",
            linestyle="None",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=STAGE_STYLES[artifact]["label"],
        )
        for artifact in STAGE_ORDER
    ]

    size_handles = [
        Line2D(
            [0],
            [0],
            color=SIZE_COLORS[size],
            linewidth=1.6,
            label=SIZE_LABELS[size],
        )
        for size in SIZE_ORDER
    ]

    combined_handles = size_handles + stage_handles

    fig.legend(
        handles=combined_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(combined_handles),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=0.9,
        handletextpad=0.55,
        columnspacing=0.95,
        labelspacing=0.8,
        borderaxespad=0.2,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1], w_pad=1.1, h_pad=1.8)
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
            print(
                f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.",
                file=sys.stderr,
            )
            return 1

        matplotlib.use("Agg")

    try:
        rows = load_plot_rows(args.csv)
        grouped_rows = build_panel_rows(rows)
        panel_size_values = build_panel_size_values(grouped_rows)
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

    fig = build_figure(panel_size_values)

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
