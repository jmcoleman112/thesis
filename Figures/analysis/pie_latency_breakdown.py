#!/usr/bin/env python
"""
Build a latency-breakdown pie chart from the timing log in research/piechatdata.txt.

The script parses lines that contain:
  timings_avg_ms={...} frames=... obj_runs=... kp_runs=...

Keypoint detection is weighted by kp_runs / frames so the chart reflects its
effective per-frame contribution rather than its per-invocation cost.

Run:
  python Figures/analysis/pie_latency_breakdown.py
  python Figures/analysis/pie_latency_breakdown.py --no-show --save-name latency_breakdown_pie
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import OrderedDict
from pathlib import Path

SCATTER_DIR = Path(__file__).resolve().parents[1] / "scatter"
if str(SCATTER_DIR) not in sys.path:
    sys.path.insert(0, str(SCATTER_DIR))

from figure_save_dialog import prompt_save_figure, sanitize_filename

DATA_PATH = Path(__file__).resolve().parents[2] / "research" / "piechatdata.txt"
OUT_DIR = Path(__file__).resolve().parents[1] / "produced_images"
FONT_FAMILY = "Times New Roman"
FIG_WIDTH_IN = 4.2
FIG_HEIGHT_IN = 3.4
LABEL_DISTANCE = 1.24
PCT_DISTANCE = 0.80
OUTER_LABEL_FONT_SIZE = 6.6

LINE_RE = re.compile(
    r"timings_avg_ms=(\{.*?\}) frames=(\d+) fps=.*? obj_runs=(\d+) kp_runs=(\d+)"
)

RAW_COMPONENT_SPECS = OrderedDict(
    [
        ("club_assign_ms", ("Club assignment", "direct")),
        ("obj_detect_ms", ("Object detection", "obj_weighted")),
        ("kp_detect_ms", ("Keypoint detection", "kp_weighted")),
        ("obj_track_ms", ("Object tracking", "direct")),
        ("mapping_ms", ("Mapping", "direct")),
        ("kp_track_ms", ("Keypoint tracking", "direct")),
        ("ball_assign_ms", ("Ball assignment", "direct")),
        ("speed_ms", ("Speed", "direct")),
    ]
)

DISPLAY_GROUPS = OrderedDict(
    [
        ("Club assignment", ("#d95f02", ["club_assign_ms"])),
        ("Object detection", ("#1b9e77", ["obj_detect_ms"])),
        ("Keypoint detection", ("#7570b3", ["kp_detect_ms"])),
        ("Object model processing", ("#e7298a", ["obj_track_ms", "ball_assign_ms", "speed_ms"])),
        ("Keypoint model processing", ("#66a61e", ["mapping_ms", "kp_track_ms"])),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the latency log text file.",
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


def _autopct_with_threshold(threshold_pct: float):
    def formatter(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= threshold_pct else ""

    return formatter


def _format_slice_label(label: str) -> str:
    if label == "Club assignment":
        return "Club\nassignment"
    if label == "Keypoint detection":
        return "Keypoint\ndetection"
    if label == "Object model processing":
        return "Object model\nprocessing"
    if label == "Keypoint model processing":
        return "Keypoint model\nprocessing"
    return label


def load_component_means(data_path: Path) -> tuple[list[tuple[str, float, str]], float, int]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if data_path.stat().st_size == 0:
        raise ValueError(f"Data file is empty: {data_path}")

    accumulators = {key: [] for key in RAW_COMPONENT_SPECS}
    matched_rows = 0

    for raw_line in data_path.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.search(raw_line)
        if not match:
            continue

        timings = ast.literal_eval(match.group(1))
        frames = int(match.group(2))
        obj_runs = int(match.group(3))
        kp_runs = int(match.group(4))
        if frames <= 0:
            continue

        matched_rows += 1
        obj_weight = obj_runs / frames
        kp_weight = kp_runs / frames

        for raw_key, (_, mode) in RAW_COMPONENT_SPECS.items():
            value = float(timings.get(raw_key, 0.0))
            if mode == "obj_weighted":
                value *= obj_weight
            elif mode == "kp_weighted":
                value *= kp_weight
            accumulators[raw_key].append(value)

    if matched_rows == 0:
        raise ValueError("No timing rows matched the expected format in the data file.")

    raw_means = {
        raw_key: (sum(values) / len(values)) if values else 0.0
        for raw_key, values in accumulators.items()
    }

    components: list[tuple[str, float, str]] = []
    for label, (color, raw_keys) in DISPLAY_GROUPS.items():
        grouped_value = sum(raw_means.get(raw_key, 0.0) for raw_key in raw_keys)
        components.append((label, grouped_value, color))

    total_ms = sum(value for _, value, _ in components)
    return components, total_ms, matched_rows


def draw_donut(
    ax,
    components: list[tuple[str, float, str]],
    total_ms: float,
    *,
    title: str | None = None,
    center_text: str | None = None,
):
    values = [value for _, value, _ in components]
    colors = [color for _, _, color in components]
    labels = [
        _format_slice_label(label)
        for label, _, _ in components
    ]

    _, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        labeldistance=LABEL_DISTANCE,
        autopct=_autopct_with_threshold(3.0),
        pctdistance=PCT_DISTANCE,
        textprops={"fontsize": OUTER_LABEL_FONT_SIZE, "color": "black", "ha": "center"},
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 0.8},
    )

    label_offsets = {
        "Object detection": (-0.03, 0.03),
        "Object model processing": (-0.10, -0.06),
        "Keypoint model processing": (0.10, -0.06),
    }

    for text, (label, _, color) in zip(texts, components):
        text.set_color(color)
        text.set_fontsize(OUTER_LABEL_FONT_SIZE)
        text.set_fontweight("bold")
        if label in label_offsets:
            x, y = text.get_position()
            x_offset, y_offset = label_offsets[label]
            text.set_position((x + x_offset, y + y_offset))
            text.set_ha("right" if x_offset < 0 else "left")

    hidden_pct_labels = {"Object model processing", "Keypoint model processing"}
    for autotext, (label, _, _) in zip(autotexts, components):
        if label in hidden_pct_labels:
            autotext.set_text("")
            continue
        autotext.set_color("white")
        autotext.set_fontsize(7)
        autotext.set_fontweight("bold")

    ax.text(
        0,
        0,
        center_text if center_text is not None else f"Avg latency\n{total_ms:.1f} ms",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    if title:
        ax.set_title(title, fontsize=9, pad=6)
    ax.set_aspect("equal")


def build_figure(components: list[tuple[str, float, str]], total_ms: float):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)
    draw_donut(ax, components, total_ms)

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.06)
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


def main() -> int:
    args = parse_args()

    try:
        components, total_ms, matched_rows = load_component_means(args.data)
        fig, _ = build_figure(components, total_ms)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Rows used:", matched_rows)
    print("Data:", args.data)
    for label, value, _ in components:
        print(f"{label}: {value:.3f} ms ({value / total_ms * 100:.1f}%)")

    if args.save_name:
        save_direct(fig, args.save_name)

    if not args.no_show:
        plt.show()
        if not args.save_name:
            prompt_save_figure(fig, default_name="latency_breakdown_pie")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
