#!/usr/bin/env python
"""
Shared utilities for object-model latency vs accuracy panel layouts.

Used by:
  python Figures/scatter_map_vs_latency_object_11_ds3_1x5.py
  python Figures/scatter_map_vs_latency_object_26_ds3_1x5.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

REQUIRED_COLS = ["Model", "Location", "Latency ms"]
MAP_COLUMN = "mAP50-95"
LATENCY_COLUMN = "Latency ms"
CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
FILTER_TOKEN = "/object/"
FAMILY_SUFFIXES = ("n", "s", "m", "l", "x")
FONT_FAMILY = "Times New Roman"
IEEE_TWO_COL_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 1.9

STAGE_ORDER = [
    "uncompressed",
    "baseline_accelerated",
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
    "uncompressed": "Uncompressed",
    "baseline_accelerated": "Accelerated",
    "distilled": "Distilled",
    "pruned": "Pruned",
    "quantized": "Quantized",
    "pruned_quantized": "Pruned+Quantized",
    "distilled_pruned": "Distilled+Pruned",
    "distilled_quantized": "Distilled+Quantized",
    "distilled_pruned_quantized": "Distilled+P+Q",
    "other": "Other",
}

STAGE_COLORS = {
    "uncompressed": "#1f77b4",
    "baseline_accelerated": "#ff7f0e",
    "distilled": "#17becf",
    "pruned": "#2ca02c",
    "quantized": "#d62728",
    "pruned_quantized": "#9467bd",
    "distilled_pruned": "#8c564b",
    "distilled_quantized": "#bcbd22",
    "distilled_pruned_quantized": "#e377c2",
    "other": "#7f7f7f",
}

DISTILL_TOKENS = ("distill", "distillation", "student", "teacher", "kd")


def _norm(value: object) -> str:
    return str(value).replace("\\", "/").strip().lower()


def is_ds3(model: object, location: object) -> bool:
    model_text = _norm(model)
    loc = _norm(location)
    return "ds3" in model_text or "/ds3" in loc


def _extract_family(model: object, location: object) -> str | None:
    model_text = _norm(model)
    loc = _norm(location)

    model_match = re.search(r"^((?:11|26)[nsmlx])(?:_|$)", model_text)
    if model_match:
        return model_match.group(1)

    loc_match = re.search(r"/object/((?:11|26)[nsmlx])(?:_ds3)?/", loc)
    if loc_match:
        return loc_match.group(1)

    return None


def _infer_stage(model: object, location: object) -> str:
    model_text = _norm(model)
    loc = _norm(location)

    has_distill = any(token in loc or token in model_text for token in DISTILL_TOKENS)
    has_pruning = "/pruning/" in loc or bool(re.search(r"_p\d+", model_text))
    has_quant = (
        "/quantization/" in loc
        or "/pruning_quantization/" in loc
        or "int8" in model_text
        or "fp16" in model_text
    )
    has_baseline = "/baseline/" in loc or "baseline" in model_text

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
        if ".engine" in model_text:
            return "baseline_accelerated"
        return "uncompressed"
    return "other"


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLS + [MAP_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def _series_families(series_prefix: str, family_suffixes: tuple[str, ...] = FAMILY_SUFFIXES) -> tuple[str, ...]:
    if series_prefix not in {"11", "26"}:
        raise ValueError("series_prefix must be '11' or '26'.")
    if not family_suffixes:
        raise ValueError("family_suffixes cannot be empty.")
    return tuple(f"{series_prefix}{suffix}" for suffix in family_suffixes)


def prepare_rows(
    df: pd.DataFrame,
    *,
    series_prefix: str,
    ds3_only: bool,
    family_suffixes: tuple[str, ...] = FAMILY_SUFFIXES,
) -> pd.DataFrame:
    subset = df[df["Location"].apply(lambda v: FILTER_TOKEN in _norm(v))].copy()
    if ds3_only:
        subset = subset[subset.apply(lambda row: is_ds3(row["Model"], row["Location"]), axis=1)]

    families = _series_families(series_prefix, family_suffixes=family_suffixes)
    subset["family"] = subset.apply(lambda row: _extract_family(row["Model"], row["Location"]), axis=1)
    subset = subset[subset["family"].isin(families)].copy()
    if subset.empty:
        scope = "DS3 object rows" if ds3_only else "object rows"
        raise ValueError(f"No {scope} found for families: {', '.join(families)}.")

    subset[MAP_COLUMN] = pd.to_numeric(subset[MAP_COLUMN], errors="coerce")
    subset[LATENCY_COLUMN] = pd.to_numeric(subset[LATENCY_COLUMN], errors="coerce")
    usable = subset.dropna(subset=[MAP_COLUMN, LATENCY_COLUMN]).copy()
    if usable.empty:
        raise ValueError(
            f"No rows with numeric {MAP_COLUMN} and {LATENCY_COLUMN} for families: {', '.join(families)}."
        )

    usable["stage"] = usable.apply(lambda row: _infer_stage(row["Model"], row["Location"]), axis=1)
    return usable


def plot_1x5_panels(
    rows: pd.DataFrame,
    *,
    series_prefix: str,
    family_suffixes: tuple[str, ...] = FAMILY_SUFFIXES,
    figure_title: str | None = None,
) -> tuple[object, object]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]

    families = _series_families(series_prefix, family_suffixes=family_suffixes)
    panel_count = len(families)
    fig_width = IEEE_TWO_COL_WIDTH_IN if panel_count >= 5 else max(2.2, 1.45 * panel_count)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(fig_width, FIG_HEIGHT_IN),
        dpi=300,
        sharey=True,
    )

    axes_list = list(axes) if hasattr(axes, "__len__") else [axes]
    present_stages: set[str] = set()

    for index, family in enumerate(families):
        ax = axes_list[index]
        family_rows = rows[rows["family"] == family].copy()

        if family_rows.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=7, transform=ax.transAxes)
        else:
            for stage in STAGE_ORDER:
                stage_rows = family_rows[family_rows["stage"] == stage]
                if stage_rows.empty:
                    continue
                present_stages.add(stage)
                ax.scatter(
                    stage_rows[LATENCY_COLUMN],
                    stage_rows[MAP_COLUMN],
                    s=16,
                    alpha=0.85,
                    color=STAGE_COLORS.get(stage, STAGE_COLORS["other"]),
                )

        ax.set_title(family.upper(), fontsize=7, pad=2)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes_list[0].set_ylabel(MAP_COLUMN, fontsize=7)
    axes_list[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.supxlabel(LATENCY_COLUMN, fontsize=7, y=0.14)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STAGE_COLORS[stage],
            markersize=5,
            label=STAGE_LABELS.get(stage, stage),
        )
        for stage in STAGE_ORDER
        if stage in present_stages
    ]

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            frameon=False,
            fontsize=6,
            ncol=max(1, len(legend_handles)),
            bbox_to_anchor=(0.5, 0.03),
            columnspacing=0.8,
            handletextpad=0.3,
        )

    if figure_title:
        fig.suptitle(figure_title, fontsize=9, y=0.98)
        fig.tight_layout(rect=[0.02, 0.12, 1, 0.93], pad=0.15, w_pad=0.25)
    else:
        fig.tight_layout(rect=[0.02, 0.12, 1, 1], pad=0.15, w_pad=0.25)
    return fig, axes


def run_series_panel(
    *,
    series_prefix: str,
    ds3_only: bool,
    family_suffixes: tuple[str, ...] = FAMILY_SUFFIXES,
    figure_title: str | None,
    default_name: str,
) -> int:
    try:
        df = load_csv(CSV_PATH)
        rows = prepare_rows(
            df,
            series_prefix=series_prefix,
            ds3_only=ds3_only,
            family_suffixes=family_suffixes,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    families = _series_families(series_prefix, family_suffixes=family_suffixes)
    print("Series:", series_prefix)
    print("Families:", ", ".join(families))
    print("Rows:", len(rows))
    print("\nRows by family:")
    print(rows["family"].value_counts().reindex(families, fill_value=0))
    print("\nRows by stage:")
    print(rows["stage"].value_counts().reindex(STAGE_ORDER, fill_value=0))

    try:
        fig, _ = plot_1x5_panels(
            rows,
            series_prefix=series_prefix,
            family_suffixes=family_suffixes,
            figure_title=figure_title,
        )
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error importing/plotting with matplotlib: {exc}", file=sys.stderr)
        return 1

    plt.show()
    prompt_save_figure(fig, default_name=default_name)
    return 0
