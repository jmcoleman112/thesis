#!/usr/bin/env python
"""
Deprecated wrapper.

Use one of:
  python Figures/scatter_map_vs_latency_object.py
  python Figures/scatter_map_vs_latency_pose.py

Running this file will execute both sequentially.
"""

from __future__ import annotations

from scatter_map_vs_latency_object import main as main_object
from scatter_map_vs_latency_pose import main as main_pose


def main() -> int:
    print("Running object models plot...")
    rc = main_object()
    if rc != 0:
        return rc
    print("Running pose models plot...")
    return main_pose()


if __name__ == "__main__":
    raise SystemExit(main())
