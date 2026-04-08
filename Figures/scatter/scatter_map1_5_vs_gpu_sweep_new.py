#!/usr/bin/env python
"""
Sweep JSON scatter plot: Pass mAP1-5 vs GPU usage.

Run:
  python Figures/scatter/scatter_map1_5_vs_gpu_sweep_new.py
"""

from __future__ import annotations

from scatter_map1_5_sweep_new_utils import run_plot


def main() -> int:
    return run_plot(
        x_col="GPU (%)",
        x_label="GPU (%)",
        default_name="sweep_new_map1_5_vs_gpu",
        x_tick_fmt="%.1f",
        x_min_pad=1.0,
        x_reference_value=80.0,
        x_reference_label="80%",
        x_axis_max=85.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
