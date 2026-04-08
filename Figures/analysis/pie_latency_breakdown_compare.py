#!/usr/bin/env python
"""
Build a side-by-side latency-breakdown comparison using two timing logs.

Default layout:
- left: research/piechartdatatwo.txt
- right: research/piechatdata.txt

An arrow is drawn from the left donut to the right donut.

Run:
  python Figures/analysis/pie_latency_breakdown_compare.py
  python Figures/analysis/pie_latency_breakdown_compare.py --no-show --save-name latency_breakdown_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pie_latency_breakdown import FONT_FAMILY, draw_donut, load_component_means, prompt_save_figure, sanitize_filename

LEFT_DATA_PATH = Path(__file__).resolve().parents[2] / "research" / "piechartdatatwo.txt"
RIGHT_DATA_PATH = Path(__file__).resolve().parents[2] / "research" / "piechatdata.txt"
OUT_DIR = Path(__file__).resolve().parents[1] / "produced_images"
FIG_WIDTH_IN = 8.2
FIG_HEIGHT_IN = 3.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-data", type=Path, default=LEFT_DATA_PATH, help="Path to the left timing log.")
    parser.add_argument("--right-data", type=Path, default=RIGHT_DATA_PATH, help="Path to the right timing log.")
    parser.add_argument("--save-name", help="Save directly to Figures/produced_images without prompting.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    return parser.parse_args()


def build_figure(
    left_components: list[tuple[str, float, str]],
    left_total_ms: float,
    right_components: list[tuple[str, float, str]],
    right_total_ms: float,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=300)

    draw_donut(
        axes[0],
        left_components,
        left_total_ms,
        center_text=f"Uncompressed:\n{left_total_ms:.1f} ms",
    )
    draw_donut(
        axes[1],
        right_components,
        right_total_ms,
        center_text=f"Compressed:\n{right_total_ms:.1f} ms",
    )

    axes[0].annotate(
        "",
        xy=(0.59, 0.52),
        xytext=(0.41, 0.52),
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#444444"},
    )

    fig.tight_layout(pad=0.15)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.06, wspace=0.35)
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
