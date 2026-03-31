#!/usr/bin/env python
"""
Scatter comparison for DS3 object family variants.

This plot is designed to compare how latency and mAP change as
transformations are applied (baseline, pruning, quantization, etc.).
If distillation rows are present in the CSV, they are included too.

Run:
  python Figures/scatter_map_vs_latency_object_11n_11l_ds3_transforms.py

Set SERIES_MODE in this file to one of: "11", "26", "both".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_COLS = ["Model", "Location", "Latency ms"]
MAP_COLUMN = "mAP50-95"
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FILTER_TOKEN = "/object/"
SERIES_FAMILIES = {
    "11": ("11n", "11s", "11m", "11l"),
    "26": ("26n", "26s", "26m", "26l"),
    "both": ("11n", "11s", "11m", "11l", "26n", "26s", "26m", "26l"),
}
SERIES_MODE = "both"
IEEE_ONE_COL_WIDTH_IN = 3.5
IEEE_ONE_COL_HEIGHT_IN = 3.9

ANNOTATE_POINTS = False

STAGE_ORDER = [
    "baseline",
    "distilled",
    "pruned",
    "quantized",
    "pruned_quantized",
    "distilled_pruned",
    "distilled_quantized",
    "distilled_pruned_quantized",
    "other",
]

STAGE_LABELS = {
    "baseline": "Baseline",
    "distilled": "Distill",
    "pruned": "Pruned",
    "quantized": "Quantized",
    "pruned_quantized": "Pruned+Quantized",
    "distilled_pruned": "Distill+Pruned",
    "distilled_quantized": "Distill+Quant",
    "distilled_pruned_quantized": "Distill+P+Q",
    "other": "Other",
}

STAGE_COLORS = {
    "baseline": "#1f77b4",
    "distilled": "#17becf",
    "pruned": "#2ca02c",
    "quantized": "#d62728",
    "pruned_quantized": "#9467bd",
    "distilled_pruned": "#8c564b",
    "distilled_quantized": "#bcbd22",
    "distilled_pruned_quantized": "#e377c2",
    "other": "#7f7f7f",
}

POINT_MARKER = "o"
POINT_SIZE = 16

DISTILL_TOKENS = ("distill", "distillation", "student", "teacher", "kd")


def _normalize_location(value: str) -> str:
    return str(value).replace("\\", "/").lower()


def _normalize_model(value: str) -> str:
    return str(value).strip().lower()


def is_ds3(model: str, location: str) -> bool:
    m = _normalize_model(model)
    loc = _normalize_location(location)
    return "ds3" in m or "/ds3" in loc


def extract_family(model: str, location: str) -> str | None:
    m = _normalize_model(model)
    loc = _normalize_location(location)

    model_match = re.search(r"^((?:11|26)[nmls])(?:_|$)", m)
    if model_match:
        return model_match.group(1)

    loc_match = re.search(r"/object/((?:11|26)[nmls])(?:_ds3)?/", loc)
    if loc_match:
        return loc_match.group(1)

    return None


def extract_artifact(model: str) -> str:
    m = _normalize_model(model)
    if ".engine" in m:
        return "engine"
    if m.endswith(".pt"):
        return "pt"
    return "other"


def extract_pruning_ratio(model: str, location: str) -> int | None:
    m = _normalize_model(model)
    loc = _normalize_location(location)

    loc_match = re.search(r"/pruning(?:_quantization)?/(\d+)(?:/|$)", loc)
    if loc_match:
        return int(loc_match.group(1))

    model_match = re.search(r"_p(\d+)(?:_|\\.|$)", m)
    if model_match:
        return int(model_match.group(1))

    return None


def extract_quant_mode(model: str, location: str) -> str | None:
    m = _normalize_model(model)
    loc = _normalize_location(location)

    if "/int8/" in loc or "int8" in m:
        return "int8"
    if "/fp16/" in loc or "fp16" in m:
        return "fp16"
    return None


def infer_stage(model: str, location: str) -> str:
    m = _normalize_model(model)
    loc = _normalize_location(location)

    has_distill = any(token in loc or token in m for token in DISTILL_TOKENS)
    has_pruning = "/pruning/" in loc or bool(re.search(r"_p\d+", m))
    has_quant = (
        "/quantization/" in loc
        or "/pruning_quantization/" in loc
        or "int8" in m
        or "fp16" in m
    )
    has_baseline = "/baseline/" in loc or "baseline" in m

    if "/pruning_quantization/" in loc:
        has_pruning = True
        has_quant = True

    if has_distill and has_pruning and has_quant:
        return "distilled_pruned_quantized"
    if has_distill and has_pruning:
        return "distilled_pruned"
    if has_distill and has_quant:
        return "distilled_quantized"
    if has_pruning and has_quant:
        return "pruned_quantized"
    if has_distill:
        return "distilled"
    if has_pruning:
        return "pruned"
    if has_quant:
        return "quantized"
    if has_baseline:
        return "baseline"
    return "other"


def build_setting_label(row: pd.Series) -> str:
    stage_label = STAGE_LABELS.get(str(row["stage"]), str(row["stage"]).title())
    parts = [str(row["family"]).upper(), stage_label]

    artifact = str(row["artifact"])
    if artifact in {"pt", "engine"}:
        parts.append(f".{artifact}")

    pruning_ratio = row.get("pruning_ratio")
    if pd.notna(pruning_ratio):
        parts.append(f"p{int(pruning_ratio)}")

    quant_mode = row.get("quant_mode")
    if isinstance(quant_mode, str) and quant_mode:
        parts.append(quant_mode.upper())

    return " ".join(parts)


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS + [MAP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def prepare_subset(df: pd.DataFrame, family_allowlist: tuple[str, ...]) -> pd.DataFrame:
    subset = df[df["Location"].apply(lambda v: FILTER_TOKEN in _normalize_location(v))].copy()
    subset = subset[subset.apply(lambda r: is_ds3(r["Model"], r["Location"]), axis=1)]

    subset["family"] = subset.apply(lambda r: extract_family(r["Model"], r["Location"]), axis=1)
    subset = subset[subset["family"].isin(family_allowlist)].copy()
    if subset.empty:
        raise ValueError(f"No DS3 object rows found for: {', '.join(family_allowlist)}.")

    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset["Latency ms"] = pd.to_numeric(subset["Latency ms"], errors="coerce")
    usable = subset.dropna(subset=[MAP_COLUMN, "Latency ms"]).copy()
    if usable.empty:
        raise ValueError(f"No rows with numeric mAP and Latency for filtered families: {', '.join(family_allowlist)}.")

    usable["artifact"] = usable["Model"].apply(extract_artifact)
    usable["stage"] = usable.apply(lambda r: infer_stage(r["Model"], r["Location"]), axis=1)
    usable["pruning_ratio"] = usable.apply(lambda r: extract_pruning_ratio(r["Model"], r["Location"]), axis=1)
    usable["quant_mode"] = usable.apply(lambda r: extract_quant_mode(r["Model"], r["Location"]), axis=1)
    usable["stage_rank"] = usable["stage"].apply(lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else len(STAGE_ORDER))
    usable["setting_label"] = usable.apply(build_setting_label, axis=1)

    return usable


def build_panel_groups(series_mode: str) -> list[tuple[str, tuple[str, ...]]]:
    if series_mode == "both":
        return [
            ("11n + 26n", ("11n", "26n")),
            ("11s + 26s", ("11s", "26s")),
            ("11m + 26m", ("11m", "26m")),
            ("11l + 26l", ("11l", "26l")),
        ]

    return [(family, (family,)) for family in SERIES_FAMILIES[series_mode]]


def _plot_family_scatter(ax, family_df: pd.DataFrame, family: str) -> None:
    if family_df.empty:
        ax.set_title(family, fontsize=7, pad=1)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return

    for _, row in family_df.iterrows():
        color = STAGE_COLORS.get(str(row["stage"]), STAGE_COLORS["other"])
        ax.scatter(
            row["Latency ms"],
            row[MAP_COLUMN],
            s=POINT_SIZE,
            marker=POINT_MARKER,
            color=color,
            alpha=0.9,
            edgecolors="none",
            zorder=3,
        )

        if ANNOTATE_POINTS:
            ax.annotate(
                str(row["setting_label"]),
                (row["Latency ms"], row[MAP_COLUMN]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=5,
                alpha=0.8,
            )

    ax.set_title(family, fontsize=7, pad=1)
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

    # Use a per-family mAP range so each panel has a useful vertical scale.
    map_min = float(family_df[MAP_COLUMN].min())
    map_max = float(family_df[MAP_COLUMN].max())
    if map_max > map_min:
        pad = max(0.003, (map_max - map_min) * 0.12)
    else:
        pad = max(0.003, map_max * 0.02)
    y0 = max(0.0, map_min - pad)
    y1 = min(1.0, map_max + pad)
    if y1 <= y0:
        y1 = min(1.0, y0 + 0.01)
    ax.set_ylim(y0, y1)


def plot_scatter(usable: pd.DataFrame, panel_groups: list[tuple[str, tuple[str, ...]]]) -> tuple[object, object]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    count = len(panel_groups)
    if count <= 4:
        rows, cols = 2, 2
        fig_w, fig_h = IEEE_ONE_COL_WIDTH_IN, IEEE_ONE_COL_HEIGHT_IN
    else:
        rows, cols = 2, 4
        fig_w, fig_h = IEEE_ONE_COL_WIDTH_IN * 2.0, IEEE_ONE_COL_HEIGHT_IN

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_w, fig_h),
        dpi=300,
        sharex=False,
        sharey=False,
    )
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for i, (panel_title, panel_families) in enumerate(panel_groups):
        ax = axes_flat[i]
        panel_df = usable[usable["family"].isin(panel_families)].copy()
        _plot_family_scatter(ax, panel_df, panel_title)
        if i < cols:
            ax.tick_params(axis="x", labelbottom=False)

    for j in range(count, len(axes_flat)):
        axes_flat[j].set_visible(False)

    present_stages = [stage for stage in STAGE_ORDER if (usable["stage"] == stage).any()]
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STAGE_COLORS[stage],
            markeredgecolor="none",
            markeredgewidth=0,
            markersize=5,
            label=STAGE_LABELS.get(stage, stage),
        )
        for stage in present_stages
    ]

    if stage_handles:
        fig.legend(
            handles=stage_handles,
            loc="lower center",
            ncol=len(stage_handles),
            frameon=False,
            fontsize=4.6,
            bbox_to_anchor=(0.5, 0.07),
            columnspacing=0.35,
            handletextpad=0.22,
        )

    fig.supxlabel("Latency ms", fontsize=7, y=0.12)
    fig.supylabel(MAP_COLUMN, fontsize=7, x=0.04)
    fig.tight_layout(rect=[0.08, 0.16, 1, 1], pad=0.2)
    return fig, axes


def main() -> int:
    if SERIES_MODE not in SERIES_FAMILIES:
        print(f"Error: invalid SERIES_MODE '{SERIES_MODE}'. Use one of: {', '.join(SERIES_FAMILIES)}.", file=sys.stderr)
        return 1
    series_mode = SERIES_MODE
    family_allowlist = SERIES_FAMILIES[series_mode]
    panel_groups = build_panel_groups(series_mode)

    try:
        df = load_csv(CSV_PATH)
        usable = prepare_subset(df, family_allowlist)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Series:", series_mode)
    print("Families:", ", ".join(family_allowlist))
    print("Panels:", ", ".join(title for title, _ in panel_groups))
    print("Filtered rows:", len(usable))
    print("\nCounts by stage:")
    print(usable["stage"].value_counts().reindex(STAGE_ORDER, fill_value=0))
    print("\nCounts by family/artifact:")
    print(usable.groupby(["family", "artifact"]).size())

    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception as exc:
        print(f"Error importing matplotlib: {exc}", file=sys.stderr)
        return 1

    fig, _ = plot_scatter(usable, panel_groups)
    import matplotlib.pyplot as plt

    plt.show()
    series_suffix = "11_26" if series_mode == "both" else series_mode
    prompt_save_figure(fig, default_name=f"object_{series_suffix}_ds3_transforms_scatter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
