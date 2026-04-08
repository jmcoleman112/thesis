#!/usr/bin/env python
"""
Sweep JSON scatter plot: Pass mAP1-5 vs temperature.

Run:
  python Figures/scatter/scatter_map1_5_vs_temp_sweep_new.py
"""

from __future__ import annotations

from scatter_map1_5_sweep_new_utils import run_plot


def main() -> int:
    return run_plot(
        x_col="Temp (C)",
        x_label="Temperature (C)",
        default_name="sweep_new_map1_5_vs_temp",
        x_tick_fmt="%.1f",
        x_min_pad=0.2,
        x_reference_value=75.0,
        x_reference_label="75 C",
        x_axis_max=80.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
