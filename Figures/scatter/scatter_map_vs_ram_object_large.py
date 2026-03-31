#!/usr/bin/env python
"""
Large scatter plot of mAP50-95 vs RAM usage for object DS3 models.

Run:
  python Figures/scatter_map_vs_ram_object_large.py
"""

from __future__ import annotations

from scatter_map_vs_object_metric_large import run_metric_plot


def main() -> int:
    return run_metric_plot(
        metric_col="RAM (GB)",
        x_label="RAM (GB)",
        default_name="object_map_vs_ram_large",
    )


if __name__ == "__main__":
    raise SystemExit(main())
