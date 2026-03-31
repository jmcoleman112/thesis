#!/usr/bin/env python
"""
Large scatter plot of mAP50-95 vs power draw for object DS3 models.

Run:
  python Figures/scatter_map_vs_power_object_large.py
"""

from __future__ import annotations

from scatter_map_vs_object_metric_large import run_metric_plot


def main() -> int:
    return run_metric_plot(
        metric_col="Power (W)",
        x_label="Power (W)",
        default_name="object_map_vs_power_large",
    )


if __name__ == "__main__":
    raise SystemExit(main())
