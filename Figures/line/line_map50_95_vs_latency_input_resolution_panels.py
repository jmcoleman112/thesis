#!/usr/bin/env python
"""
Plot mAP50-95 vs latency paths for input-resolution variants.

The figure contains four panels:
- 11 DS3 object
- 26 DS3 object
- 11 pose
- 26 pose

Each panel shows five family lines (N/S/M/L/X), connecting:
baseline -> 960 -> 768 -> 640

Artifact selection:
- By default, all stages prefer `.engine` rows.
- Use `--baseline-artifact pt` to compare a `.pt` baseline against
  `.engine` 960/768/640 rows when both are available.

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

MAP_COLUMN = "mAP50-95"
LATENCY_COLUMN = "Latency ms"
REQUIRED_COLUMNS = ["Model", "Location", MAP_COLUMN, LATENCY_COLUMN]
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"

SIZE_ORDER = ["n", "s", "m", "l", "x"]
STAGE_ORDER = ["baseline", "960", "768", "640"]

SUBPLOT_ORDER = [
    ("11", "object"),
    ("26", "object"),
    ("11", "pose"),
    ("26", "pose"),
]

SUBPLOT_TITLES = {
    ("11", "object"): "11 DS3 Object",
    ("26", "object"): "26 DS3 Object",
    ("11", "pose"): "11 Pose",
    ("26", "pose"): "26 Pose",
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
    "baseline": "baseline",
    "960": "960",
    "768": "768",
    "640": "640",
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

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=300, sharey=True)
    axes_map = {subplot_key: axes.flat[index] for index, subplot_key in enumerate(SUBPLOT_ORDER)}
    all_ys: list[float] = []

    for subplot_key in SUBPLOT_ORDER:
        ax = axes_map[subplot_key]
        for path in subplot_paths[subplot_key]:
            size = str(path["size"])
            color = SIZE_COLORS.get(size, "#7f7f7f")
            rows = path["rows"]
            stages = list(path["stages"])
            xs = [float(rows[stage][LATENCY_COLUMN]) for stage in stages]
            ys = [float(rows[stage][MAP_COLUMN]) for stage in stages]
            all_ys.extend(ys)

            ax.plot(xs, ys, color=color, linewidth=1.35, alpha=0.95, zorder=1)

            for stage in stages:
                row = rows[stage]
                ax.scatter(
                    float(row[LATENCY_COLUMN]),
                    float(row[MAP_COLUMN]),
                    color=color,
                    marker=STAGE_MARKERS[stage],
                    s=42,
                    edgecolors="black",
                    linewidths=0.45,
                    zorder=3,
                )

        ax.set_title(SUBPLOT_TITLES[subplot_key], fontsize=10, pad=4)
        ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    y_min = min(all_ys)
    y_max = max(all_ys)
    y_pad = max((y_max - y_min) * 0.08, 0.005)

    for ax in axes.flat:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    axes[0][0].set_ylabel(MAP_COLUMN, fontsize=10)
    axes[1][0].set_ylabel(MAP_COLUMN, fontsize=10)
    axes[1][0].set_xlabel("Latency ms", fontsize=10)
    axes[1][1].set_xlabel("Latency ms", fontsize=10)

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
            markersize=6.2,
            label=STAGE_LABELS[stage],
        )
        for stage in STAGE_ORDER
    ]

    size_legend = fig.legend(
        handles=size_handles,
        title="Model size",
        loc="upper center",
        bbox_to_anchor=(0.32, 1.01),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        title_fontsize=8.5,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    fig.add_artist(size_legend)

    fig.legend(
        handles=stage_handles,
        title="Input size",
        loc="upper center",
        bbox_to_anchor=(0.77, 1.01),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        title_fontsize=8.5,
        columnspacing=0.9,
        handletextpad=0.45,
    )

    fig.suptitle(
        f"Input-Resolution Paths ({baseline_artifact.upper()} baseline preference)",
        fontsize=11.5,
        y=1.07,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.92], w_pad=1.0, h_pad=1.0)
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
