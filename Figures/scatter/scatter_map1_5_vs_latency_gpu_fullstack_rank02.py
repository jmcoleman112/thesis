#!/usr/bin/env python
"""
Rank-02 full-stack combo scatter plot pair: Pass mAP1-5 vs latency and GPU usage.

Run:
  python Figures/scatter/scatter_map1_5_vs_latency_gpu_fullstack_rank02.py
"""

from __future__ import annotations

from scatter_fullstack_rank02_gpu_utils import run_metric_pair_plot


def main() -> int:
    return run_metric_pair_plot(
        left_panel={
            "x_col": "average_latency_ms",
            "x_label": "Latency (ms)",
            "title": "Latency",
            "x_tick_fmt": "%.0f",
            "x_margin_frac": 0.5,
            "x_min_pad": 10.0,
        },
        right_panel={
            "x_col": "GPU (%)",
            "x_label": "GPU Usage (%)",
            "title": "GPU Usage",
            "x_tick_fmt": "%.0f",
            "x_margin_frac": 0.5,
            "x_min_pad": 3.0,
        },
        y_col="mAP1-5",
        y_label="Pass mAP$_{1-5}$",
        default_name="fullstack_rank02_map1_5_vs_latency_gpu_pair",
        y_tick_fmt="%.2f",
    )


if __name__ == "__main__":
    raise SystemExit(main())
