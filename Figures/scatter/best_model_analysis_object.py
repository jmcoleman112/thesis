#!/usr/bin/env python
"""
Object scatter plot focused on a small set of highlighted "best" models.

Run:
  python Figures/scatter/best_model_analysis_object.py

Edit `MIN_MAP` and `MAX_LATENCY_MS` below to control filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_COLS = ["Model", "Location", "Latency ms"]
MAP_COLUMN = "mAP50-95"
MIN_MAP = 0.78
MAX_LATENCY_MS = 40
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FILTER_TOKEN = "/object/"
IEEE_ONE_COL_WIDTH_IN = 3.5
IEEE_ONE_COL_HEIGHT_IN = 2.5
DROP_PT_MODELS = False
FONT_FAMILY = "Times New Roman"

BASE_POINT_COLOR = "#8d8d8d"
HIGHLIGHT_COLOR = "#d95f02"
BASE_POINT_SIZE = 14
HIGHLIGHT_POINT_SIZE = 28

HIGHLIGHT_MODELS = [
    "26s_DS3_p90_768_fp16.engine",
    "26s_DS3_768_fp16.engine",
    "11n_DS3_p90_fp16.engine",
    "11n_DS3_from_11l_fp16.engine",
    "26s_DS3_int8.engine",
    "26s_DS3_p90_960_fp16.engine",
    "26s_DS3_960_fp16.engine",
    "26s_DS3_from_26m_fp16.engine",
    "26s_DS3_fp16.engine",
    "26m_DS3_p90_640_fp16.engine",
    "26s_DS3_640_fp16.engine",
]

ANNOTATION_OFFSETS = {
    "26s_DS3_p90_768_fp16.engine": (6, 0),
    "26s_DS3_768_fp16.engine": (6, 0),
    "11n_DS3_p90_fp16.engine": (-40, 0),
    "11n_DS3_from_11l_fp16.engine": (6, 0),
    "26s_DS3_int8.engine": (6, 0),
    "26s_DS3_p90_960_fp16.engine": (-45, 0),
    "26s_DS3_960_fp16.engine": (6, 0),
    "26s_DS3_from_26m_fp16.engine": (-50, 0),
    "26s_DS3_fp16.engine": (-28, 0),
    "26m_DS3_p90_640_fp16.engine": (6, 0),
    "26s_DS3_640_fp16.engine": (6, 0),
}


def _normalize_location(value: str) -> str:
    return str(value).replace("\\", "/").lower()


def is_ds3(model: str, location: str) -> bool:
    m = str(model).lower()
    loc = _normalize_location(location)
    return "ds3" in m or "/ds3" in loc


def is_pt_model(model: str) -> bool:
    return str(model).lower().endswith(".pt")


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS + [MAP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def build_label(model: str) -> str:
    return model.replace("_DS3_", " ").replace("_", " ").replace(".engine", "").replace(".pt", "")


def main() -> int:
    try:
        df = load_csv(CSV_PATH)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    subset = df[df["Location"].apply(lambda v: FILTER_TOKEN in _normalize_location(v))].copy()
    subset = subset[subset.apply(lambda r: is_ds3(r["Model"], r["Location"]), axis=1)]
    if DROP_PT_MODELS:
        subset = subset[~subset["Model"].apply(is_pt_model)].copy()
    if subset.empty:
        print("Error: no DS3 object models found (location contains /object/ and DS3).", file=sys.stderr)
        return 1

    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset["Latency ms"] = pd.to_numeric(subset["Latency ms"], errors="coerce")
    numeric = subset.dropna(subset=[MAP_COLUMN, "Latency ms"]).copy()
    usable = numeric[numeric[MAP_COLUMN] >= MIN_MAP].copy()

    dropped_non_numeric = len(subset) - len(numeric)
    if dropped_non_numeric:
        print(f"Warning: dropped {dropped_non_numeric} rows with non-numeric {MAP_COLUMN} or Latency ms.")

    dropped_low_map = len(numeric) - len(usable)
    if dropped_low_map:
        print(f"Warning: dropped {dropped_low_map} rows with {MAP_COLUMN} < {MIN_MAP:.2f}.")

    if MAX_LATENCY_MS is not None:
        before_latency_cap = len(usable)
        usable = usable[usable["Latency ms"] <= MAX_LATENCY_MS].copy()
        dropped_high_latency = before_latency_cap - len(usable)
        if dropped_high_latency:
            print(f"Warning: dropped {dropped_high_latency} rows with Latency ms > {MAX_LATENCY_MS:.2f}.")

    if usable.empty:
        print("Error: no object rows remain after applying numeric, mAP, and latency filters.", file=sys.stderr)
        return 1

    highlight = usable[usable["Model"].isin(HIGHLIGHT_MODELS)].copy()
    missing_highlights = [model for model in HIGHLIGHT_MODELS if model not in set(highlight["Model"])]
    if missing_highlights:
        print("Warning: some highlight models were not found after filtering:")
        for model in missing_highlights:
            print(f"  - {model}")

    try:
        import matplotlib as mpl
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    fig, ax = plt.subplots(figsize=(IEEE_ONE_COL_WIDTH_IN, IEEE_ONE_COL_HEIGHT_IN), dpi=300)
    ax.scatter(
        usable["Latency ms"],
        usable[MAP_COLUMN],
        alpha=0.65,
        s=BASE_POINT_SIZE,
        color=BASE_POINT_COLOR,
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        highlight["Latency ms"],
        highlight[MAP_COLUMN],
        alpha=0.95,
        s=HIGHLIGHT_POINT_SIZE,
        color=HIGHLIGHT_COLOR,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    for _, row in highlight.iterrows():
        model = str(row["Model"])
        dx, dy = ANNOTATION_OFFSETS.get(model, (6, 6))
        annotation = ax.annotate(
            build_label(model),
            xy=(row["Latency ms"], row[MAP_COLUMN]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6,
            color=HIGHLIGHT_COLOR,
            zorder=4,
        )
        annotation.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    ax.set_xlabel("Latency ms", fontsize=8)
    ax.set_ylabel(MAP_COLUMN, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(pad=0.08)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.14, top=0.995)
    plt.show()
    prompt_save_figure(fig, default_name="best_model_analysis_object")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
