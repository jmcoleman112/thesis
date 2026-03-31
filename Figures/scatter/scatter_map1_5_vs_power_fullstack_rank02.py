#!/usr/bin/env python
"""
Rank-02 full-stack combo scatter plot: Pass mAP1-5 vs power draw.

Run:
  python Figures/scatter/scatter_map1_5_vs_power_fullstack_rank02.py
"""

from __future__ import annotations

from scatter_fullstack_rank02_gpu_utils import run_metric_plot


def main() -> int:
    return run_metric_plot(
        x_col="PS Power (W)",
        x_label="Power (W)",
        y_col="mAP1-5",
        y_label="Pass mAP$_{1-5}$",
        title="Pass mAP1-5 vs Power (Rank-02 Full-Stack Combos)",
        default_name="fullstack_rank02_map1_5_vs_power",
        x_tick_fmt="%.2f",
        y_tick_fmt="%.2f",
    )


if __name__ == "__main__":
    raise SystemExit(main())
