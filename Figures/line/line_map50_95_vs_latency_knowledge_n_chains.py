#!/usr/bin/env python
"""
Plot knowledge-distillation trade-offs for object and pose models.

The figure contains four panels:
- YOLO11 object
- YOLO26 object
- YOLO11 pose
- YOLO26 pose

Each panel shows N-, S-, M-, and L-student lines, connecting:
baseline -> from X -> from L -> from M -> from S

Artifact selection is configurable:
- `mixed`: object uses `.engine`, pose uses `.pt`
- `engine`: both object and pose use `.engine`
- `pt`: both object and pose use `.pt`

Object panels use `mAP50-95`, while the pose panels use `Validation2 mAP50-95`.

Run:
  python Figures/line/line_map50_95_vs_latency_knowledge_n_chains.py
  python Figures/line/line_map50_95_vs_latency_knowledge_n_chains.py --no-show
  python Figures/line/line_map50_95_vs_latency_knowledge_n_chains.py --output Figures/produced_images/map50_95_vs_latency_knowledge_ns_chains.png
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
DISPLAY_MAP_LABEL = DEFAULT_MAP_COLUMN
LATENCY_COLUMN = "Latency ms"
REQUIRED_COLUMNS = [
    "Model",
    "Location",
    DEFAULT_MAP_COLUMN,
    POSE_VALIDATION2_MAP_COLUMN,
    LATENCY_COLUMN,
]
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FIGURE_WIDTH_IN = 7.1
FIGURE_HEIGHT_IN = 6.35
TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 12.5
LEGEND_FONT_SIZE = 12

OBJECT_BASELINE_RE = re.compile(r"^(?P<family>(11|26)[nsml])_ds3_baseline\.(?P<artifact>pt|engine)$", re.IGNORECASE)
OBJECT_DISTILL_RE = re.compile(
    r"^(?P<family>(11|26)[nsml])_ds3_from_(?P<src>(11|26)[slmx])\.(?P<artifact>pt|engine)$",
    re.IGNORECASE,
)
POSE_BASELINE_RE = re.compile(r"^(?P<family>(11|26)[nsml])_pose_baseline\.(?P<artifact>pt|engine)$", re.IGNORECASE)
POSE_DISTILL_RE = re.compile(
    r"^(?P<family>(11|26)[nsml])_pose_from_(?P<src>(11|26)[slmx])\.(?P<artifact>pt|engine)$",
    re.IGNORECASE,
)

GROUP_ORDER = [
    ("11", "object"),
    ("26", "object"),
    ("11", "pose"),
    ("26", "pose"),
]

GROUP_TITLES = {
    ("11", "object"): "YOLO11 Object",
    ("26", "object"): "YOLO26 Object",
    ("11", "pose"): "YOLO11 Pose",
    ("26", "pose"): "YOLO26 Pose",
}

TARGET_ORDER = ["n", "s", "m", "l"]
GROUP_TARGETS = {
    ("11", "object"): ["n", "s", "l"],
    ("26", "object"): ["n", "s", "m", "l"],
    ("11", "pose"): ["n", "s", "l"],
    ("26", "pose"): ["n", "s", "m", "l"],
}

TARGET_LABELS = {
    "n": "N",
    "s": "S",
    "m": "M",
    "l": "L",
}

TARGET_COLORS = {
    "n": "#1f77b4",
    "s": "#ff7f0e",
    "m": "#2ca02c",
    "l": "#d62728",
}

STAGE_ORDER_BY_TARGET = {
    "n": ["baseline", "x", "l", "m", "s"],
    "s": ["baseline", "x", "l", "m"],
    "m": ["baseline", "x", "l"],
    "l": ["baseline", "x"],
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
        description="Plot knowledge-distillation trade-offs with configurable pt/engine artifact selection."
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to the model summary CSV.")
    parser.add_argument(
        "--artifact-mode",
        choices=["mixed", "engine", "pt"],
        default="mixed",
        help="Artifact selection: mixed keeps object on engine and pose on pt; engine/pt force both domains.",
    )
    parser.add_argument("--output", type=Path, help="Optional PNG output path.")
    parser.add_argument("--no-show", action="store_true", help="Build the figure without opening a plot window.")
    return parser.parse_args()


def allowed_artifact(domain: str, artifact_mode: str) -> str:
    if artifact_mode == "mixed":
        return "engine" if domain == "object" else "pt"
    return artifact_mode


def parse_model(model: str, *, artifact_mode: str) -> Optional[dict[str, str]]:
    text = str(model).strip()
    if not text:
        return None

    match = OBJECT_BASELINE_RE.match(text)
    if match:
        artifact = match.group("artifact").lower()
        if artifact != allowed_artifact("object", artifact_mode):
            return None
        family = match.group("family").lower()
        return {
            "family": family,
            "series": family[:2],
            "target": family[-1],
            "domain": "object",
            "stage": "baseline",
            "artifact": artifact,
        }

    match = OBJECT_DISTILL_RE.match(text)
    if match:
        artifact = match.group("artifact").lower()
        if artifact != allowed_artifact("object", artifact_mode):
            return None
        family = match.group("family").lower()
        source_family = match.group("src").lower()
        if family[:2] != source_family[:2]:
            return None
        return {
            "family": family,
            "series": family[:2],
            "target": family[-1],
            "domain": "object",
            "stage": source_family[-1],
            "artifact": artifact,
        }

    match = POSE_BASELINE_RE.match(text)
    if match:
        artifact = match.group("artifact").lower()
        if artifact != allowed_artifact("pose", artifact_mode):
            return None
        family = match.group("family").lower()
        return {
            "family": family,
            "series": family[:2],
            "target": family[-1],
            "domain": "pose",
            "stage": "baseline",
            "artifact": artifact,
        }

    match = POSE_DISTILL_RE.match(text)
    if match:
        artifact = match.group("artifact").lower()
        if artifact != allowed_artifact("pose", artifact_mode):
            return None
        family = match.group("family").lower()
        source_family = match.group("src").lower()
        return {
            "family": family,
            "series": family[:2],
            "target": family[-1],
            "domain": "pose",
            "stage": source_family[-1],
            "artifact": artifact,
        }

    return None


def load_plot_rows(csv_path: Path, *, artifact_mode: str) -> list[dict[str, object]]:
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
            parsed = parse_model(row.get("Model", ""), artifact_mode=artifact_mode)
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
                    "map_column": map_column,
                    "map_value": map_value,
                    LATENCY_COLUMN: latency_value,
                }
            )

    if not usable:
        raise ValueError("No matching distillation rows contain numeric mAP50-95 and latency values.")

    return usable


def build_group_rows(
    rows: list[dict[str, object]]
) -> dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]] = {
        group_key: {target: {} for target in TARGET_ORDER} for group_key in GROUP_ORDER
    }

    for row in rows:
        group_key = (str(row["series"]), str(row["domain"]))
        target = str(row["target"])
        stage = str(row["stage"])
        if group_key in grouped and target in grouped[group_key] and stage in STAGE_STYLES:
            grouped[group_key][target][stage] = row

    missing_baselines: list[str] = []
    for group_key in GROUP_ORDER:
        for target in GROUP_TARGETS[group_key]:
            if "baseline" not in grouped[group_key][target]:
                missing_baselines.append(f"{GROUP_TITLES[group_key]} {TARGET_LABELS[target]}")

    if missing_baselines:
        raise ValueError("Missing one or more required distillation baselines: " + ", ".join(missing_baselines))

    return grouped


def print_group_summary(
    grouped_rows: dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]]
) -> None:
    for group_key in GROUP_ORDER:
        target_summaries = []
        for target in GROUP_TARGETS[group_key]:
            present_stages = [stage for stage in STAGE_ORDER_BY_TARGET[target] if stage in grouped_rows[group_key][target]]
            target_summaries.append(f"{TARGET_LABELS[target]}={','.join(present_stages)}")
        print(f"- {GROUP_TITLES[group_key]}: {'; '.join(target_summaries)}")


def build_figure(grouped_rows: dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]]):
    import matplotlib.pyplot as plt
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
    axes_map = {group_key: axes.flat[index] for index, group_key in enumerate(GROUP_ORDER)}
    panel_xs: dict[tuple[str, str], list[float]] = {group_key: [] for group_key in GROUP_ORDER}
    row_ys: dict[str, list[float]] = {"object": [], "pose": []}

    for group_key in GROUP_ORDER:
        ax = axes_map[group_key]
        row_key = str(group_key[1])

        for target in GROUP_TARGETS[group_key]:
            target_rows = grouped_rows[group_key][target]
            available_stages = [stage for stage in STAGE_ORDER_BY_TARGET[target] if stage in target_rows]
            if not available_stages:
                continue

            latencies = [float(target_rows[stage][LATENCY_COLUMN]) for stage in available_stages]
            map_values = [float(target_rows[stage]["map_value"]) for stage in available_stages]
            panel_xs[group_key].extend(latencies)
            row_ys[row_key].extend(map_values)

            ax.plot(
                latencies,
                map_values,
                color=TARGET_COLORS[target],
                linewidth=1.7,
                zorder=1,
            )

            for stage in available_stages:
                stage_latency = float(target_rows[stage][LATENCY_COLUMN])
                stage_map = float(target_rows[stage]["map_value"])
                ax.scatter(
                    stage_latency,
                    stage_map,
                    color=TARGET_COLORS[target],
                    marker=STAGE_STYLES[stage]["marker"],
                    s=115,
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=3,
                )

        ax.set_title(GROUP_TITLES[group_key], fontsize=TITLE_FONT_SIZE, pad=8)
        ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.tick_params(labelsize=TICK_FONT_SIZE, width=1.0, length=5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        if panel_xs[group_key]:
            x_min = min(panel_xs[group_key])
            x_max = max(panel_xs[group_key])
            x_pad = max((x_max - x_min) * 0.06, 1.0)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

    object_axes = [axes_map[("11", "object")], axes_map[("26", "object")]]
    pose_axes = [axes_map[("11", "pose")], axes_map[("26", "pose")]]

    if row_ys["object"]:
        object_y_min = min(row_ys["object"])
        object_y_max = max(row_ys["object"])
        object_y_pad = max((object_y_max - object_y_min) * 0.10, 0.015)
        for ax in object_axes:
            ax.set_ylim(object_y_min - object_y_pad, object_y_max + object_y_pad)

    if row_ys["pose"]:
        pose_y_min = min(row_ys["pose"])
        pose_y_max = max(row_ys["pose"])
        pose_y_pad = max((pose_y_max - pose_y_min) * 0.10, 0.015)
        for ax in pose_axes:
            ax.set_ylim(pose_y_min - pose_y_pad, pose_y_max + pose_y_pad)

    axes[0][0].set_ylabel(DISPLAY_MAP_LABEL, fontsize=LABEL_FONT_SIZE)
    axes[1][0].set_ylabel(DISPLAY_MAP_LABEL, fontsize=LABEL_FONT_SIZE)
    axes[1][0].set_xlabel("Latency ms", fontsize=LABEL_FONT_SIZE)
    axes[1][1].set_xlabel("Latency ms", fontsize=LABEL_FONT_SIZE)

    target_handles = [
        Line2D(
            [0],
            [0],
            color=TARGET_COLORS[target],
            linewidth=1.6,
            label=TARGET_LABELS[target],
        )
        for target in TARGET_ORDER
    ]
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=STAGE_STYLES[stage]["marker"],
            color="black",
            linestyle="None",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=STAGE_STYLES[stage]["label"],
        )
        for stage in ["baseline", "x", "l", "m", "s"]
    ]

    combined_handles = target_handles + stage_handles
    fig.legend(
        handles=combined_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(combined_handles),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=0.9,
        handletextpad=0.55,
        columnspacing=0.9,
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
            print(f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.", file=sys.stderr)
            return 1
        matplotlib.use("Agg")

    try:
        rows = load_plot_rows(args.csv, artifact_mode=args.artifact_mode)
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
