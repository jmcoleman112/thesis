#!/usr/bin/env python
"""
Sweep JSON scatter plot: Pass mAP1-5 vs latency.

Run:
  python Figures/scatter/scatter_map1_5_vs_latency_sweep_new.py
"""

from __future__ import annotations

from scatter_map1_5_sweep_new_utils import run_plot


def main() -> int:
    return run_plot(
        x_col="average_latency_ms",
        x_label="Latency (ms)",
        default_name="sweep_new_map1_5_vs_latency",
        x_tick_fmt="%.1f",
        x_min_pad=2.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
