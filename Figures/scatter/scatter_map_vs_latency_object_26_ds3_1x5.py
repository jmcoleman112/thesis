#!/usr/bin/env python
"""
Latency vs accuracy panels for DS3 object families 26n/26s/26m/26l.

Run:
  python Figures/scatter_map_vs_latency_object_26_ds3_1x5.py
"""

from __future__ import annotations

from scatter_map_vs_latency_object_series_1x5 import run_series_panel


def main() -> int:
    return run_series_panel(
        series_prefix="26",
        ds3_only=True,
        family_suffixes=("n", "s", "m", "l"),
        figure_title=None,
        default_name="object_26n_s_m_l_ds3_map_vs_latency",
    )


if __name__ == "__main__":
    raise SystemExit(main())
