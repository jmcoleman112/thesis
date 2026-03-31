#!/usr/bin/env python
"""
Rank-02 full-stack combo scatter plot: possession accuracy vs GPU usage.

Possession accuracy is derived as:
  100 - DeltaPosPct

Run:
  python Figures/scatter/scatter_possession_accuracy_vs_gpu_fullstack_rank02.py
"""

from __future__ import annotations

from scatter_fullstack_rank02_gpu_utils import run_metric_plot


def main() -> int:
    return run_metric_plot(
        x_col="GPU (%)",
        x_label="GPU Usage (%)",
        y_col="possession_accuracy_pct",
        y_label="Possession Accuracy (%)",
        title="Possession Accuracy vs GPU Usage (Rank-02 Full-Stack Combos)",
        default_name="fullstack_rank02_possession_accuracy_vs_gpu",
        x_tick_fmt="%.1f",
        y_tick_fmt="%.1f",
    )


if __name__ == "__main__":
    raise SystemExit(main())
