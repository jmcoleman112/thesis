#!/usr/bin/env python
"""
Sweep JSON scatter plot: Pass mAP1-5 vs power draw.

Run:
  python Figures/scatter/scatter_map1_5_vs_power_sweep_new.py
"""

from __future__ import annotations

from scatter_map1_5_sweep_new_utils import run_plot


def main() -> int:
    return run_plot(
        x_col="PS Power (W)",
        x_label="Power (W)",
        default_name="sweep_new_map1_5_vs_power",
        x_tick_fmt="%.1f",
        x_min_pad=0.2,
        x_reference_value=15.0,
        x_reference_label="15 W",
        x_axis_max=20.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
