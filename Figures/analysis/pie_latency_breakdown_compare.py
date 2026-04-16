#!/usr/bin/env python
"""
Build a side-by-side latency-breakdown comparison using two timing logs.

Run:
  python Figures/analysis/pie_latency_breakdown_compare.py
  python Figures/analysis/pie_latency_breakdown_compare.py --no-show --save-name latency_breakdown_compare
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from matplotlib import patheffects

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pie_latency_breakdown import (
    FONT_FAMILY,
    load_component_means,
    prompt_save_figure,
    sanitize_filename,
)

LEFT_DATA_PATH = Path(__file__).resolve().parents[2] / "research" / "piechartdatatwo.txt"
RIGHT_DATA_PATH = Path(__file__).resolve().parents[2] / "research" / "piechatdata.txt"
OUT_DIR = Path(__file__).resolve().parents[1] / "produced_images"

FIG_WIDTH_IN = 10.8
FIG_HEIGHT_IN = 4.2
LABEL_THRESHOLD_PCT = 6.0
INNER_PCT_LABELS = {"Object detection", "Keypoint detection"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-data", type=Path, default=LEFT_DATA_PATH, help="Path to the left timing log.")
    parser.add_argument("--right-data", type=Path, default=RIGHT_DATA_PATH, help="Path to the right timing log.")
    parser.add_argument("--save-name", help="Save directly to Figures/produced_images without prompting.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    return parser.parse_args()


def draw_donut_clean(
    ax,
    components: list[tuple[str, float, str]],
    total_ms: float,
    center_text: str,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    skip_labels: set[str] | None = None,
    absolute_label_positions: dict[str, tuple[float, float]] | None = None,
    wedge_offsets: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict]:
    """Draw a donut chart and return wedge/label positions for above-threshold slices."""
    values = [value for _, value, _ in components]
    colors = [color for _, _, color in components]

    donut_radius = 1.08
    donut_width = 0.45
    pct_radius = donut_radius - (donut_width / 2.0)
    label_offsets = label_offsets or {}
    skip_labels = skip_labels or set()
    absolute_label_positions = absolute_label_positions or {}
    wedge_offsets = wedge_offsets or {}

    wedges, _ = ax.pie(
        values,
        labels=None,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=donut_radius,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.6},
    )

    ax.text(
        0,
        0,
        center_text,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    positions: dict[str, dict] = {}

    for wedge, (name, value, color) in zip(wedges, components):
        pct = 100 * value / total_ms if total_ms else 0.0
        if pct < LABEL_THRESHOLD_PCT:
            continue

        theta = 0.5 * (wedge.theta1 + wedge.theta2)
        theta_rad = math.radians(theta)

        x = donut_radius * math.cos(theta_rad)
        y = donut_radius * math.sin(theta_rad)
        wdx, wdy = wedge_offsets.get(name, (0.0, 0.0))
        x += wdx
        y += wdy

        if name in absolute_label_positions:
            lx, ly = absolute_label_positions[name]
        else:
            label_r = 1.32
            lx = label_r * math.cos(theta_rad)
            ly = label_r * math.sin(theta_rad)
            dx, dy = label_offsets.get(name, (0.0, 0.0))
            lx += dx
            ly += dy

        positions[name] = {"wedge_xy": (x, y), "label_xy": (lx, ly), "color": color, "annotation": None}

        # Draw outer label annotation unless skipped
        if name not in skip_labels:
            ha = "left" if lx >= 0 else "right"
            ann = ax.annotate(
                name,
                xy=(x, y),
                xytext=(lx, ly),
                ha=ha,
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=color,
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "lw": 1.0,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "connectionstyle": "arc3,rad=0.15",
                },
            )
            positions[name]["annotation"] = ann

        # Always draw inner percentage for designated labels
        if name in INNER_PCT_LABELS:
            pct_x = pct_radius * math.cos(theta_rad)
            pct_y = pct_radius * math.sin(theta_rad)
            ax.text(
                pct_x,
                pct_y,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="black",
                color="white",
            ).set_path_effects([
                patheffects.Stroke(linewidth=1.1, foreground=(0, 0, 0, 0.28)),
                patheffects.Normal(),
            ])

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    return positions


def add_shared_legend(fig, components: list[tuple[str, float, str]]):
    import matplotlib.lines as mlines

    filtered_components = [
        item for item in components
        if item[0] not in {"Object detection", "Keypoint detection"}
    ]

    handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=7,  # slightly bigger circles
            markerfacecolor=color,
            markeredgecolor=color,
            label=name,
        )
        for name, _, color in filtered_components
    ]

    legend = fig.legend(
        handles=handles,
        labels=[name for name, _, _ in filtered_components],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(filtered_components),
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    for text, (_, _, color) in zip(legend.get_texts(), filtered_components):
        text.set_color(color)
        text.set_fontweight("bold")


def build_figure(
    left_components: list[tuple[str, float, str]],
    left_total_ms: float,
    right_components: list[tuple[str, float, str]],
    right_total_ms: float,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    left_positions = draw_donut_clean(
        axes[0],
        left_components,
        left_total_ms,
        center_text=f"Uncompressed:\n{left_total_ms:.1f} ms",
        label_offsets={
            "Object detection": (-0.25, 0.35),
        },
        skip_labels={"Keypoint detection"},
    )

    right_positions = draw_donut_clean(
        axes[1],
        right_components,
        right_total_ms,
        center_text=f"Compressed:\n{right_total_ms:.1f} ms",
        label_offsets={
            "Object model processing": (0.25, 0.06),
            "Keypoint model processing": (0.27, 0.08),
            "Keypoint detection": (-0.55, -0.35),
        },
        absolute_label_positions={
            "Object detection": (0.59, 1.1),
        },
        wedge_offsets={
            "Object detection": (-0.1, 0.0),
        },
    )

    # Draw a line from the left chart's "Keypoint detection" wedge to the
    # left end ("K") of the shared label on the right chart.
    # Force a render pass first so we can get the annotation text bounding box.
    if "Keypoint detection" in right_positions and "Keypoint detection" in left_positions:
        r = right_positions["Keypoint detection"]
        l = left_positions["Keypoint detection"]
        r_ann = r.get("annotation")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        if r_ann is not None:
            bbox = r_ann.get_window_extent(renderer)
            # Left-centre of the text in display (pixel) coords, with vertical correction
            left_centre_display = (bbox.x0, (bbox.y0 + bbox.y1) / 2)
            xyA_raw = axes[1].transData.inverted().transform(left_centre_display)
            xyA = (xyA_raw[0] - 0.1, xyA_raw[1] - 0.15)  # x and vertical correction
        else:
            xyA = r["label_xy"]

        # Pull the left end back slightly from the wedge edge
        wx, wy = l["wedge_xy"]
        scale = 0.88  # <1 pulls endpoint toward centre
        xyB = (wx * scale, wy * scale)

        con = ConnectionPatch(
            xyA=xyA,
            xyB=xyB,
            coordsA=axes[1].transData,
            coordsB=axes[0].transData,
            color=r["color"],
            lw=1.0,
            arrowstyle="-",
        )
        con.set_clip_on(False)
        fig.add_artist(con)

    # shared legend using left component order/colors
    add_shared_legend(fig, left_components)

    axes[0].annotate(
        "",
        xy=(0.54, 0.50),
        xytext=(0.46, 0.50),
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops={"arrowstyle": "->", "linewidth": 2.4, "color": "#444444"},
    )

    fig.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.08, wspace=0.08)
    return fig, axes


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
        left_components, left_total_ms, left_rows = load_component_means(args.left_data)
        right_components, right_total_ms, right_rows = load_component_means(args.right_data)
        fig, _ = build_figure(
            left_components,
            left_total_ms,
            right_components,
            right_total_ms,
        )
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Left rows used:", left_rows)
    print("Left data:", args.left_data)
    print("Right rows used:", right_rows)
    print("Right data:", args.right_data)

    if args.save_name:
        save_direct(fig, args.save_name)

    if not args.no_show:
        plt.show()
        if not args.save_name:
            prompt_save_figure(fig, default_name="latency_breakdown_compare")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
