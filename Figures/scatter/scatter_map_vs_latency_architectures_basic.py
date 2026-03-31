#!/usr/bin/env python
"""
Basic 3-point scatter plot of mAP vs latency for object detection architectures.

Suggested alternatives to YOLO:
  - Faster R-CNN
  - SSD

Edit ARCH_POINTS with your measured values before using the figure in a report.

Run:
  python Figures/scatter_map_vs_latency_architectures_basic.py
"""

from __future__ import annotations

import sys

from figure_save_dialog import prompt_save_figure

IEEE_ONE_COL_WIDTH_IN = 3.5
IEEE_ONE_COL_HEIGHT_IN = 2.38
IEEE_SERIF_STACK = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
LAYOUT_LEFT = 0.145
LAYOUT_RIGHT = 0.995
LAYOUT_TOP = 0.992
LAYOUT_BOTTOM = 0.19
TIGHT_LAYOUT_PAD = 0.08

ARCH_POINTS = [
    {"architecture": "YOLO", "mAP50-95": 0.88, "Latency ms": 141, "color": "#1f77b4"},
    {"architecture": "Faster R-CNN", "mAP50-95": 0.9, "Latency ms": 268, "color": "#ff7f0e"},
    {"architecture": "SSD", "mAP50-95": 0.91, "Latency ms": 350, "color": "#2ca02c"},
]


def validate_points(points: list[dict[str, object]]) -> None:
    required = {"architecture", "mAP50-95", "Latency ms", "color"}
    for idx, point in enumerate(points):
        missing = required - set(point.keys())
        if missing:
            raise ValueError(f"Point {idx} is missing required keys: {sorted(missing)}")

        try:
            float(point["mAP50-95"])
            float(point["Latency ms"])
        except Exception as exc:
            raise ValueError(f"Point {idx} has non-numeric mAP/Latency: {point}") from exc


def main() -> int:
    try:
        validate_points(ARCH_POINTS)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = IEEE_SERIF_STACK

    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDTH_IN, IEEE_ONE_COL_HEIGHT_IN), dpi=300)

    for point in ARCH_POINTS:
        x = float(point["Latency ms"])
        y = float(point["mAP50-95"])
        label = str(point["architecture"])
        color = str(point["color"])

        ax.scatter(x, y, s=28, color=color, alpha=0.9, label=label)
        x_offset, y_offset = 4, 4
        ha, va = "left", "bottom"
        if label.lower() == "ssd":
            x_offset, y_offset = -15, 10
            ha, va = "left", "top"
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            ha=ha,
            va=va,
            fontsize=6,
        )

    ax.set_xlabel("Latency ms", fontsize=8)
    ax.set_ylabel("mAP50-95", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylim(0.8, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="lower right", fontsize=6, borderaxespad=0.2, handletextpad=0.3)

    fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
    fig.subplots_adjust(
        left=LAYOUT_LEFT,
        right=LAYOUT_RIGHT,
        top=LAYOUT_TOP,
        bottom=LAYOUT_BOTTOM,
    )
    plt.show()
    prompt_save_figure(fig, default_name="architecture_map_vs_latency_basic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
