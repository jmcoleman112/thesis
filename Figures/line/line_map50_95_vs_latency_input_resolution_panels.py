#!/usr/bin/env python
"""
Plot mAP50-95 vs latency paths for input-resolution variants.

The figure contains four panels:
- 11 DS3 object
- 26 DS3 object
- 11 pose
- 26 pose

Each panel shows five family lines (N/S/M/L/X), connecting:
baseline -> 75% -> 60% -> 50%

Artifact selection:
- By default, all stages prefer `.engine` rows.
- Use `--baseline-artifact pt` to compare a `.pt` baseline against
  `.engine` 75%/60%/50% rows when both are available.

Run:
  python Figures/line/line_map50_95_vs_latency_input_resolution_panels.py
  python Figures/line/line_map50_95_vs_latency_input_resolution_panels.py --no-show
  python Figures/line/line_map50_95_vs_latency_input_resolution_panels.py --baseline-artifact pt
  python Figures/line/line_map50_95_vs_latency_input_resolution_panels.py --output Figures/produced_images/map50_95_vs_latency_input_resolution_panels.png
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

SIZE_ORDER = ["n", "s", "m", "l", "x"]
STAGE_ORDER = ["baseline", "960", "768", "640"]

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

SIZE_COLORS = {
    "n": "#1f77b4",
    "s": "#ff7f0e",
    "m": "#2ca02c",
    "l": "#d62728",
    "x": "#9467bd",
}

STAGE_MARKERS = {
    "baseline": "o",
    "960": "s",
    "768": "^",
    "640": "D",
}

STAGE_LABELS = {
    "baseline": "Uncompressed",
    "960": "75%",
    "768": "60%",
    "640": "50%",
}

OBJECT_RE = re.compile(
    r"^(?P<family>(?:11|26)[nsmlx])_ds3_(?P<stage>baseline|960|768|640)\.(?P<artifact>pt|engine)$",
    re.IGNORECASE,
)
POSE_RE = re.compile(
    r"^(?P<family>(?:11|26)[nsmlx])_pose_(?P<stage>baseline|960|768|640)\.(?P<artifact>pt|engine)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mAP50-95 vs latency paths for baseline/960/768/640 input-resolution variants."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to the model summary CSV.",
    )
    parser.add_argument(
        "--baseline-artifact",
        choices=["engine", "pt"],
        default="engine",
        help="Artifact preference for the baseline stage (default: engine).",
    )
    parser.add_argument(
        "--pose-validation2",
        action="store_true",
        help="Use Validation2 mAP50-95 for the pose panels only.",
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


def stages_for_size(size: str) -> list[str]:
    if size == "n":
        return ["baseline", "960", "768"]
    return STAGE_ORDER


def parse_model(model: str) -> Optional[dict[str, str]]:
    model_text = str(model).strip()
    if not model_text:
        return None

    match = OBJECT_RE.match(model_text)
    if match:
        family = match.group("family").lower()
        return {
            "series": family[:2],
            "family": family,
            "size": family[-1],
            "domain": "object",
            "stage": match.group("stage").lower(),
            "artifact": match.group("artifact").lower(),
        }

    match = POSE_RE.match(model_text)
    if match:
        family = match.group("family").lower()
        return {
            "series": family[:2],
            "family": family,
            "size": family[-1],
            "domain": "pose",
            "stage": match.group("stage").lower(),
            "artifact": match.group("artifact").lower(),
        }

    return None


def required_columns(*, pose_validation2: bool) -> list[str]:
    columns = ["Model", "Location", DEFAULT_MAP_COLUMN, LATENCY_COLUMN]
    if pose_validation2:
        columns.append(POSE_VALIDATION2_MAP_COLUMN)
    return columns


def load_plot_rows(csv_path: Path, *, pose_validation2: bool) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required_columns(pose_validation2=pose_validation2) if column not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        usable: list[dict[str, object]] = []
        for row in reader:
            parsed = parse_model(row.get("Model", ""))
            if parsed is None:
                continue

            map_column = (
                POSE_VALIDATION2_MAP_COLUMN
                if pose_validation2 and parsed["domain"] == "pose"
                else DEFAULT_MAP_COLUMN
            )
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
                }
            )

    if not usable:
        raise ValueError(
            "No matching input-resolution rows were found. Expected model names ending in "
            "'baseline', '960', '768', or '640'."
        )

    return usable


def choose_preferred_rows(
    rows: list[dict[str, object]],
    *,
    baseline_artifact: str,
) -> dict[tuple[str, str, str, str], dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["series"]),
            str(row["domain"]),
            str(row["family"]),
            str(row["stage"]),
        )
        grouped.setdefault(key, []).append(row)

    preferred: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for key, candidates in grouped.items():
        stage = key[3]
        if stage == "baseline":
            artifact_order = [baseline_artifact, "engine", "pt"]
        else:
            artifact_order = ["engine", "pt"]

        # De-duplicate while preserving order.
        seen_artifacts: set[str] = set()
        artifact_order = [
            artifact for artifact in artifact_order if not (artifact in seen_artifacts or seen_artifacts.add(artifact))
        ]
        preference_index = {artifact: index for index, artifact in enumerate(artifact_order)}

        chosen = sorted(
            candidates,
            key=lambda row: (
                preference_index.get(str(row["artifact"]), len(preference_index)),
                str(row["Model"]).lower(),
            ),
        )[0]
        preferred[key] = chosen

    return preferred


def build_subplot_paths(
    preferred_rows: dict[tuple[str, str, str, str], dict[str, object]]
) -> dict[tuple[str, str], list[dict[str, object]]]:
    subplot_paths: dict[tuple[str, str], list[dict[str, object]]] = {
        subplot_key: [] for subplot_key in SUBPLOT_ORDER
    }
    missing_rows: list[str] = []

    for series, domain in SUBPLOT_ORDER:
        for size in SIZE_ORDER:
            family = f"{series}{size}"
            path_rows: dict[str, dict[str, object]] = {}
            missing_stages: list[str] = []
            stages = stages_for_size(size)

            for stage in stages:
                row = preferred_rows.get((series, domain, family, stage))
                if row is None:
                    missing_stages.append(stage)
                else:
                    path_rows[stage] = row

            if missing_stages:
                missing_rows.append(
                    f"{family.upper()} {domain} ({', '.join(missing_stages)})"
                )
                continue

            subplot_paths[(series, domain)].append(
                {
                    "family": family,
                    "size": size,
                    "stages": stages,
                    "rows": path_rows,
                }
            )

    if missing_rows:
        raise ValueError(
            "Missing one or more input-resolution rows: " + "; ".join(missing_rows)
        )

    return subplot_paths


def print_summary(
    subplot_paths: dict[tuple[str, str], list[dict[str, object]]],
    *,
    baseline_artifact: str,
) -> None:
    total_paths = sum(len(paths) for paths in subplot_paths.values())
    print(f"Prepared {total_paths} family paths across {len(subplot_paths)} panels.")
    print(f"Baseline artifact preference: {baseline_artifact}")
    for subplot_key in SUBPLOT_ORDER:
        families = ", ".join(
            str(path["family"]).upper() for path in subplot_paths[subplot_key]
        )
        print(f"- {SUBPLOT_TITLES[subplot_key]}: {families}")


def build_figure(
    subplot_paths: dict[tuple[str, str], list[dict[str, object]]],
    *,
    baseline_artifact: str,
    pose_validation2: bool,
):
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

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=300, sharey="row")
    axes_map = {subplot_key: axes.flat[index] for index, subplot_key in enumerate(SUBPLOT_ORDER)}
    row_ys: dict[int, list[float]] = {0: [], 1: []}
    panel_xs: dict[tuple[str, str], list[float]] = {subplot_key: [] for subplot_key in SUBPLOT_ORDER}

    for subplot_key in SUBPLOT_ORDER:
        ax = axes_map[subplot_key]
        row_index = 0 if subplot_key[1] == "object" else 1
        for path in subplot_paths[subplot_key]:
            size = str(path["size"])
            color = SIZE_COLORS.get(size, "#7f7f7f")
            rows = path["rows"]
            stages = list(path["stages"])
            xs = [float(rows[stage][LATENCY_COLUMN]) for stage in stages]
            ys = [float(rows[stage][MAP_COLUMN]) for stage in stages]
            row_ys[row_index].extend(ys)
            panel_xs[subplot_key].extend(xs)

            ax.plot(xs, ys, color=color, linewidth=1.35, alpha=0.95, zorder=1)

            for stage in stages:
                row = rows[stage]
                ax.scatter(
                    float(row[LATENCY_COLUMN]),
                    float(row[MAP_COLUMN]),
                    color=color,
                    marker=STAGE_MARKERS[stage],
                    s=115,
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=3,
                )

        ax.set_title(SUBPLOT_TITLES[subplot_key], fontsize=TITLE_FONT_SIZE, pad=8)
        ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.tick_params(labelsize=TICK_FONT_SIZE, width=1.0, length=5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if panel_xs[subplot_key]:
            x_min = min(panel_xs[subplot_key])
            x_max = max(panel_xs[subplot_key])
            x_pad = max((x_max - x_min) * 0.05, 0.9)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

    for row_index, axes_row in enumerate(axes):
        y_min = min(row_ys[row_index])
        y_max = max(row_ys[row_index])
        y_pad = max((y_max - y_min) * 0.10, 0.01)
        for ax in axes_row:
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    axes[0][0].set_ylabel(DISPLAY_MAP_LABEL, fontsize=LABEL_FONT_SIZE)
    axes[1][0].set_ylabel(DISPLAY_MAP_LABEL, fontsize=LABEL_FONT_SIZE)
    axes[1][0].set_xlabel("Latency ms", fontsize=LABEL_FONT_SIZE)
    axes[1][1].set_xlabel("Latency ms", fontsize=LABEL_FONT_SIZE)

    size_handles = [
        Line2D(
            [0],
            [0],
            color=SIZE_COLORS[size],
            linewidth=1.6,
            label=size.upper(),
        )
        for size in SIZE_ORDER
    ]
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=STAGE_MARKERS[stage],
            color="black",
            linestyle="None",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=STAGE_LABELS[stage],
        )
        for stage in STAGE_ORDER
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
        rows = load_plot_rows(args.csv, pose_validation2=args.pose_validation2)
        preferred_rows = choose_preferred_rows(
            rows,
            baseline_artifact=args.baseline_artifact,
        )
        subplot_paths = build_subplot_paths(preferred_rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(
        subplot_paths,
        baseline_artifact=args.baseline_artifact,
    )

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"Error importing matplotlib: {exc}. Install it with 'pip install matplotlib'.",
            file=sys.stderr,
        )
        return 1

    fig = build_figure(
        subplot_paths,
        baseline_artifact=args.baseline_artifact,
        pose_validation2=args.pose_validation2,
    )

    saved_path: Optional[Path] = None
    try:
        if args.output is not None:
            saved_path = save_figure(fig, args.output)
            print(f"Saved figure to {saved_path}")
        elif not args.no_show:
            plt.show()
            saved_path = prompt_save_figure(
                fig,
                default_name="map50_95_vs_latency_input_resolution_panels",
            )
            if saved_path is not None:
                print(f"Saved figure to {saved_path}")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
