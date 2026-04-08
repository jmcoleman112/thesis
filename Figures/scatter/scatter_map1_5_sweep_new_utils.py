#!/usr/bin/env python
"""
Shared plotting logic for sweep_new full-stack scatter plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from figure_save_dialog import prompt_save_figure, sanitize_filename

JSON_PATH = Path(__file__).resolve().parents[2] / "research" / "sweep_new.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "produced_images"

FONT_FAMILY = "Times New Roman"
FIG_WIDTH_IN = 3.35
FIG_HEIGHT_IN = 3.25
POINT_SIZE = 28
X_AXIS_MARGIN_FRAC = 0.28
Y_AXIS_MIN_PCT = 30.0
Y_AXIS_MAX_PCT = 80.0
ANNOTATION_THRESHOLD_PCT = 60.0
ANNOTATION_FONT_SIZE = 4.5

REFERENCE_MAP_PCT = 71.2
REFERENCE_BAND_LOW_PCT = REFERENCE_MAP_PCT - 15
REFERENCE_LINE_COLOR = "#2f4f4f"
REFERENCE_BAND_COLOR = "#90ee90"
X_REFERENCE_LINE_COLOR = "#d62728"

OBJECT_STYLES = [
    ("26s from 26m FP16", "#1f77b4"),
    ("26s FP16", "#ff7f0e"),
    ("26s 960 FP16", "#2ca02c"),
    ("26s INT8", "#d62728"),
    ("Other", "#7f7f7f"),
]

POSE_STYLES = [
    ("26n FP16", "o"),
    ("26n P90 640", "s"),
    ("26n from 26x 768 FP16", "D"),
    ("26n from 26l 640 FP16", "^"),
    ("Other", "o"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=JSON_PATH,
        help="Path to the sweep JSON file.",
    )
    parser.add_argument(
        "--save-name",
        help="Save directly to Figures/produced_images without prompting.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the matplotlib window.",
    )
    return parser.parse_args()


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_text(value: object) -> str:
    return " ".join(str(value).replace("\\", "/").replace("_", " ").replace("-", " ").lower().split())


def _split_combo_name(combo_name: object) -> tuple[str, str]:
    raw_text = str(combo_name)
    if "+" in raw_text:
        left, right = raw_text.split("+", 1)
        return _normalize_text(left), _normalize_text(right)
    normalized = _normalize_text(raw_text)
    return normalized, normalized


def _object_style(row: dict[str, object]) -> tuple[str, str]:
    combo_name = str(row.get("combo_name", ""))
    object_path_raw = str(row.get("object_model_path", "")).replace("\\", "/").lower()
    object_path = _normalize_text(row.get("object_model_path", ""))
    combo_left, _ = _split_combo_name(combo_name)

    if (
        combo_left == "26s from 26m fp16"
        or object_path == "26s from 26m fp16"
        or combo_name.startswith("obj26m_p80_fp16")
        or "/object/26m/pruning_quantization/80/fp16/" in object_path_raw
    ):
        return "26s from 26m FP16", "#1f77b4"
    if (
        combo_left == "26s fp16"
        or object_path == "26s fp16"
        or combo_name.startswith("obj26l_fp16")
        or "/object/26l/quantization/fp16/" in object_path_raw
    ):
        return "26s FP16", "#ff7f0e"
    if (
        combo_left == "26s 960 fp16"
        or object_path == "26s 960 fp16"
        or "26s 960 fp16" in _normalize_text(combo_name)
    ):
        return "26s 960 FP16", "#2ca02c"
    if (
        combo_left == "26s int8"
        or object_path == "26s int8"
        or "26s int8" in _normalize_text(combo_name)
    ):
        return "26s INT8", "#d62728"
    return "Other", "#7f7f7f"


def _pose_style(row: dict[str, object]) -> tuple[str, str]:
    combo_name = str(row.get("combo_name", ""))
    pose_path_raw = str(row.get("pose_model_path", "")).replace("\\", "/").lower()
    pose_path = _normalize_text(row.get("pose_model_path", ""))
    _, combo_right = _split_combo_name(combo_name)

    if (
        combo_right == "26n pose fp16"
        or pose_path == "26n pose fp16"
        or combo_name.endswith("_pose26n_fp16")
        or "/pose/26n-pose/quantization/fp16/" in pose_path_raw
    ):
        return "26n FP16", "o"
    if (
        combo_right == "26n pose p90 640"
        or pose_path == "26n pose p90 640"
        or "26n pose p90 640" in _normalize_text(combo_name)
    ):
        return "26n P90 640", "s"
    if (
        combo_right == "26n pose from 26x 768 fp16"
        or pose_path == "26n pose from 26x 768 fp16"
        or combo_name.endswith("_pose26n_kd26x_p80_fp16")
        or (
            "/pose/26n-pose/distillation_pruning_quantization/80/fp16/" in pose_path_raw
            and "from_26x" in pose_path_raw
        )
    ):
        return "26n from 26x 768 FP16", "D"
    if (
        combo_right == "26n pose from 26l 640 fp16"
        or pose_path == "26n pose from 26l 640 fp16"
        or "26n pose from 26l 640 fp16" in _normalize_text(combo_name)
        or (
            "/pose/26n-pose/distillation_pruning_quantization/80/fp16/" in pose_path_raw
            and "from_26l" in pose_path_raw
        )
    ):
        return "26n from 26l 640 FP16", "^"
    return "Other", "o"


def load_rows(json_path: Path, *, x_col: str) -> list[dict[str, object]]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    if json_path.stat().st_size == 0:
        raise ValueError(f"JSON is empty on disk: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        raw_rows = data.get("rows")
    elif isinstance(data, list):
        raw_rows = data
    else:
        raise ValueError("Unsupported JSON structure. Expected a dict with 'rows' or a list of rows.")

    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("No rows found in JSON data.")

    required = {"combo_name", "object_model_path", "pose_model_path", "mAP1-5", x_col}
    first_row = raw_rows[0] if isinstance(raw_rows[0], dict) else None
    if not isinstance(first_row, dict):
        raise ValueError("JSON rows must be objects.")

    missing = [key for key in required if key not in first_row]
    if missing:
        raise ValueError(f"JSON rows missing required keys: {missing}")

    usable: list[dict[str, object]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue

        x_value = _to_float(row.get(x_col))
        map_value = _to_float(row.get("mAP1-5"))
        if x_value is None or map_value is None:
            continue

        map_pct = map_value * 100.0 if map_value <= 1.5 else map_value
        object_label, object_color = _object_style(row)
        pose_label, pose_marker = _pose_style(row)

        usable.append(
            {
                **row,
                x_col: x_value,
                "mAP1-5_pct": map_pct,
                "object_label": object_label,
                "object_color": object_color,
                "pose_label": pose_label,
                "pose_marker": pose_marker,
            }
        )

    if not usable:
        raise ValueError("No usable rows found in JSON data.")

    usable.sort(key=lambda row: float(row["mAP1-5_pct"]), reverse=True)
    return usable


def _apply_axis_limits(
    ax,
    values: list[float],
    *,
    axis: str,
    margin_frac: float,
    min_pad: float = 0.0,
) -> None:
    if not values:
        return

    value_min = min(values)
    value_max = max(values)
    value_span = value_max - value_min
    pad = max(value_span * margin_frac, min_pad)
    if value_span == 0:
        fallback_pad = 0.5 if axis == "x" else 1.0
        pad = max(pad, max(abs(value_min) * margin_frac, fallback_pad))

    if axis == "x":
        ax.set_xlim(value_min - pad, value_max + pad)
    elif axis == "y":
        ax.set_ylim(value_min - pad, value_max + pad)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def _build_legend_handles():
    from matplotlib.lines import Line2D

    object_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=5,
            label=label,
        )
        for label, color in OBJECT_STYLES[:-1]
    ]
    pose_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#8c8c8c",
            markeredgecolor="white",
            markersize=5,
            linewidth=0,
            label=label,
        )
        for label, marker in POSE_STYLES[:-1]
    ]
    return object_handles, pose_handles


def _format_annotation_text(label: str) -> str:
    return label.replace(" from ", " <- ")


def plot_rows(
    rows: list[dict[str, object]],
    *,
    x_col: str,
    x_label: str,
    x_tick_fmt: str = "%.1f",
    x_min_pad: float = 0.0,
    x_reference_value: float | None = None,
    x_reference_label: str | None = None,
    x_axis_max: float | None = None,
):
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    ax.axhspan(
        REFERENCE_BAND_LOW_PCT,
        REFERENCE_MAP_PCT,
        color=REFERENCE_BAND_COLOR,
        alpha=0.22,
        zorder=0,
    )
    ax.axhline(
        REFERENCE_MAP_PCT,
        color=REFERENCE_LINE_COLOR,
        linestyle=":",
        linewidth=1.15,
        zorder=1,
    )
    if x_reference_value is not None:
        ax.axvline(
            x_reference_value,
            color=X_REFERENCE_LINE_COLOR,
            linestyle=":",
            linewidth=1.15,
            zorder=1,
        )

    for row in rows:
        ax.scatter(
            float(row[x_col]),
            float(row["mAP1-5_pct"]),
            s=POINT_SIZE,
            color=str(row["object_color"]),
            marker=str(row["pose_marker"]),
            edgecolors="white",
            linewidths=0.35,
            alpha=0.9,
            zorder=2,
        )

    x_values = [float(row[x_col]) for row in rows]
    _apply_axis_limits(ax, x_values, axis="x", margin_frac=X_AXIS_MARGIN_FRAC, min_pad=x_min_pad)
    if x_axis_max is not None:
        current_left, current_right = ax.get_xlim()
        ax.set_xlim(current_left, max(current_right, x_axis_max))
    ax.set_ylim(Y_AXIS_MIN_PCT, Y_AXIS_MAX_PCT)

    x_min, x_max = ax.get_xlim()
    x_span = x_max - x_min
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_min = ax.get_ylim()[0]
    ax.text(
        x_min + x_span * 0.02,
        REFERENCE_MAP_PCT + y_span * 0.015,
        "71.2% SoccerNet",
        fontsize=5.8,
        color=REFERENCE_LINE_COLOR,
        va="bottom",
        ha="left",
    )
    if x_reference_value is not None and x_reference_label:
        ax.text(
            x_reference_value - x_span * 0.012,
            y_min + y_span * 0.015,
            x_reference_label,
            fontsize=6.3,
            color=X_REFERENCE_LINE_COLOR,
            ha="right",
            va="bottom",
        )

    for row in rows:
        map_pct = float(row["mAP1-5_pct"])
        if map_pct <= ANNOTATION_THRESHOLD_PCT:
            continue

        pose_label = str(row["pose_label"])
        label = (
            f"{_format_annotation_text(str(row['object_label']))}\n"
            f"{_format_annotation_text(pose_label)}"
        )
        place_bottom_left = pose_label in {"26n P90 640", "26n from 26l 640 FP16"}
        place_bottom_right = (
            str(row["object_label"]) == "26s FP16" and pose_label == "26n FP16"
        )
        if place_bottom_left:
            x_offset = -x_span * 0.015
            y_offset = -y_span * 0.012
            ha = "right"
            va = "top"
        elif place_bottom_right:
            x_offset = x_span * 0.015
            y_offset = -y_span * 0.012
            ha = "left"
            va = "top"
        else:
            x_offset = x_span * 0.015
            y_offset = y_span * 0.012
            ha = "left"
            va = "bottom"

        ax.text(
            float(row[x_col]) + x_offset,
            map_pct + y_offset,
            label,
            fontsize=ANNOTATION_FONT_SIZE,
            color=str(row["object_color"]),
            ha=ha,
            va=va,
            linespacing=0.9,
            path_effects=[pe.withStroke(linewidth=1.4, foreground="white")],
            zorder=3,
        )

    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel("Pass mAP$_{1-5}$ (%)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter(x_tick_fmt))
    ax.grid(True, linestyle="--", alpha=0.4)

    object_handles, pose_handles = _build_legend_handles()
    pose_legend = ax.legend(
        handles=pose_handles,
        title="Pose model",
        frameon=False,
        fontsize=5,
        title_fontsize=5,
        loc="upper center",
        bbox_to_anchor=(0.22, -0.22),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    ax.add_artist(pose_legend)
    ax.legend(
        handles=object_handles,
        title="Object model",
        frameon=False,
        fontsize=5,
        title_fontsize=5,
        loc="upper center",
        bbox_to_anchor=(0.78, -0.22),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(left=0.17, right=0.995, bottom=0.42, top=0.97)
    return fig, ax


def save_direct(fig, name: str) -> Path:
    cleaned = sanitize_filename(name)
    if not cleaned.lower().endswith(".png"):
        cleaned = f"{cleaned}.png"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / cleaned
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    return output_path


def run_plot(
    *,
    x_col: str,
    x_label: str,
    default_name: str,
    x_tick_fmt: str = "%.1f",
    x_min_pad: float = 0.0,
    x_reference_value: float | None = None,
    x_reference_label: str | None = None,
    x_axis_max: float | None = None,
) -> int:
    args = parse_args()

    try:
        rows = load_rows(args.json, x_col=x_col)
        fig, _ = plot_rows(
            rows,
            x_col=x_col,
            x_label=x_label,
            x_tick_fmt=x_tick_fmt,
            x_min_pad=x_min_pad,
            x_reference_value=x_reference_value,
            x_reference_label=x_reference_label,
            x_axis_max=x_axis_max,
        )
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Rows used:", len(rows))
    print("JSON:", args.json)
    print("X metric:", x_col)

    if args.save_name:
        save_direct(fig, args.save_name)

    if not args.no_show:
        plt.show()
        if not args.save_name:
            prompt_save_figure(fig, default_name=default_name)

    return 0
