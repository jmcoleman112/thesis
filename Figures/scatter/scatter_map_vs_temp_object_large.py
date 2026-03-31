#!/usr/bin/env python
"""
Large scatter plot of mAP50-95 vs temperature for object DS3 models.

Run:
  python Figures/scatter_map_vs_temp_object_large.py
"""

from __future__ import annotations

from scatter_map_vs_object_metric_large import run_metric_plot


def main() -> int:
    return run_metric_plot(
        metric_col="Temp (?C)",
        x_label="Temp (C)",
        default_name="object_map_vs_temp_large",
    )


if __name__ == "__main__":
    raise SystemExit(main())
