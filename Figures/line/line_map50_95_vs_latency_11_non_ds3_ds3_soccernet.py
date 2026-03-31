#!/usr/bin/env python
"""
Line plot comparing mAP50-95 vs latency for 11x families:
  1) Non-DS3 baseline models
  2) DS3 baseline models
  3) SoccerNet best-epoch models

SoccerNet values are auto-loaded for 11n/11s/11m from:
  research/soccernet/11n_SN_results.csv
  research/soccernet/11s_SN_results.csv
  research/soccernet/11m_SN_results.csv

For 11l/11x SoccerNet, fill the placeholders in SOCCERNET_MANUAL_MAP50_95.

Run:
  python Figures/line_map50_95_vs_latency_11_non_ds3_ds3_soccernet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from figure_save_dialog import prompt_save_figure

MAP_COLUMN = "mAP50-95"
LATENCY_COLUMN = "Latency ms"
MODEL_COLUMN = "Model"
LOCATION_COLUMN = "Location"

MODEL_SUMMARY_CSV = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
SOCCERNET_DIR = Path(__file__).resolve().parents[2] / "research" / "soccernet"
SOCCERNET_MAP_COLUMN = "metrics/mAP50-95(B)"
SOCCERNET_EPOCH_COLUMN = "epoch"

FAMILY_ORDER = ["11n", "11s", "11m", "11l", "11x"]
FAMILY_INDEX = {f: i for i, f in enumerate(FAMILY_ORDER)}

# Baseline row artifact preference when both .pt and .engine exist for a family.
ARTIFACT_PREFERENCE = "pt"
IEEE_SINGLE_COLUMN_WIDTH_IN = 3.5  # 88.9 mm (IEEE single-column width)
FIG_WIDTH_IN = IEEE_SINGLE_COLUMN_WIDTH_IN
FIG_HEIGHT_IN = 2.38
IEEE_SERIF_STACK = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
PLOT_LINE_WIDTH = 0.95
MARKER_SIZE = 4
MARKER_EDGE_WIDTH = 0.6
AXIS_BOX_LINE_WIDTH = 0.7
AXIS_LABEL_FONT_SIZE = 7
TICK_FONT_SIZE = 6
LEGEND_FONT_SIZE = 6
POINT_LABEL_FONT_SIZE = 6.0
POINT_LABEL_COLOR = "#111111"
POINT_LABEL_MAP_Y = 0.75
VERTICAL_GUIDE_LINE_WIDTH = 0.55
VERTICAL_GUIDE_ALPHA = 0.9
LEGEND_Y_ANCHOR = -0.24
LAYOUT_LEFT = 0.145
LAYOUT_RIGHT = 0.995
LAYOUT_TOP = 0.992
LAYOUT_BOTTOM = 0.19
TIGHT_LAYOUT_PAD = 0.06
LATENCY_X_MIN = 40
LATENCY_X_MAX = 450
LATENCY_X_TICKS = [50, 100, 150, 200, 250, 300, 350, 400, 450]
SERIES_COLORS = {
    "Roboflow": "#0072B2",
    "Football Analytics": "#D55E00",
    "Soccernet": "#009E73",
}

# Fill these once you have SoccerNet best-epoch mAP50-95 for 11l / 11x.
SOCCERNET_MANUAL_MAP50_95 = {
    "11l": .54,  # Example: 0.9123
    "11x": .54,  # Example: 0.9250
}


def _norm(text: object) -> str:
    return str(text).replace("\\", "/").strip().lower()


def _extract_family(model: object) -> str | None:
    m = _norm(model)
    if len(m) < 3:
        return None
    prefix = m[:3]
    return prefix if prefix in FAMILY_ORDER else None


def _infer_artifact(model: object) -> str:
    m = _norm(model)
    if m.endswith(".pt"):
        return "pt"
    if ".engine" in m:
        return "engine"
    return "other"


def _is_baseline_row(model: object, location: object) -> bool:
    m = _norm(model)
    loc = _norm(location)
    if "/baseline/" in loc:
        return True
    return "baseline" in m and "/pruning" not in loc and "/quantization" not in loc


def _is_ds3_row(model: object, location: object) -> bool:
    m = _norm(model)
    loc = _norm(location)
    return ("ds3" in m) or ("/ds3" in loc)


def _load_model_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = [MODEL_COLUMN, LOCATION_COLUMN, MAP_COLUMN, LATENCY_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df[MAP_COLUMN] = pd.to_numeric(df[MAP_COLUMN], errors="coerce")
    df[LATENCY_COLUMN] = pd.to_numeric(df[LATENCY_COLUMN], errors="coerce")
    df = df.dropna(subset=[MAP_COLUMN, LATENCY_COLUMN]).copy()
    if df.empty:
        raise ValueError("No rows with numeric mAP50-95 and Latency ms.")
    return df


def _select_baseline_family_rows(df: pd.DataFrame, *, is_ds3: bool) -> pd.DataFrame:
    working = df.copy()
    working["family"] = working[MODEL_COLUMN].apply(_extract_family)
    working = working[working["family"].isin(FAMILY_ORDER)].copy()
    working = working[working[LOCATION_COLUMN].apply(lambda x: "/object/" in _norm(x))].copy()
    working = working[working.apply(lambda r: _is_baseline_row(r[MODEL_COLUMN], r[LOCATION_COLUMN]), axis=1)].copy()
    working = working[working.apply(lambda r: _is_ds3_row(r[MODEL_COLUMN], r[LOCATION_COLUMN]) == is_ds3, axis=1)].copy()
    working["artifact"] = working[MODEL_COLUMN].apply(_infer_artifact)

    chosen_rows = []
    for family in FAMILY_ORDER:
        fam = working[working["family"] == family].copy()
        if fam.empty:
            continue

        preferred = fam[fam["artifact"] == ARTIFACT_PREFERENCE].copy()
        pool = preferred if not preferred.empty else fam
        chosen = pool.sort_values(by=[MAP_COLUMN, LATENCY_COLUMN], ascending=[False, True]).iloc[0]
        chosen_rows.append(chosen)

    if not chosen_rows:
        return pd.DataFrame(columns=list(df.columns) + ["family", "artifact", "family_order"])

    result = pd.DataFrame(chosen_rows).copy()
    result["family_order"] = result["family"].map(FAMILY_INDEX).astype(int)
    result = result.sort_values("family_order").copy()
    return result


def _load_best_soccernet_map(family: str) -> tuple[float | None, int | None]:
    csv_path = SOCCERNET_DIR / f"{family}_SN_results.csv"
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    if SOCCERNET_MAP_COLUMN not in df.columns:
        return None, None

    maps = pd.to_numeric(df[SOCCERNET_MAP_COLUMN], errors="coerce")
    valid = maps.dropna()
    if valid.empty:
        return None, None

    idx = valid.idxmax()
    best_map = float(valid.loc[idx])
    best_epoch = None
    if SOCCERNET_EPOCH_COLUMN in df.columns and pd.notna(df.loc[idx, SOCCERNET_EPOCH_COLUMN]):
        best_epoch = int(df.loc[idx, SOCCERNET_EPOCH_COLUMN])
    return best_map, best_epoch


def _build_soccernet_rows(non_ds3_rows: pd.DataFrame, ds3_rows: pd.DataFrame) -> pd.DataFrame:
    # Prefer non-DS3 family latency; fallback to DS3 latency for the same family.
    latency_lookup = {
        str(row["family"]): float(row[LATENCY_COLUMN])
        for _, row in ds3_rows.iterrows()
    }
    for _, row in non_ds3_rows.iterrows():
        latency_lookup[str(row["family"])] = float(row[LATENCY_COLUMN])

    rows = []
    for family in FAMILY_ORDER:
        latency = latency_lookup.get(family)
        if latency is None:
            continue

        if family in {"11n", "11s", "11m"}:
            best_map, best_epoch = _load_best_soccernet_map(family)
            source = "csv"
        else:
            best_map = SOCCERNET_MANUAL_MAP50_95.get(family)
            best_epoch = None
            source = "manual"

        rows.append(
            {
                "family": family,
                "family_order": FAMILY_INDEX[family],
                MAP_COLUMN: pd.to_numeric(best_map, errors="coerce"),
                LATENCY_COLUMN: latency,  # fallback rule: reuse same-family model latency
                "best_epoch": best_epoch,
                "source": source,
            }
        )

    result = pd.DataFrame(rows).sort_values("family_order").copy()
    return result


def _print_selection_summary(non_ds3: pd.DataFrame, ds3: pd.DataFrame, sn: pd.DataFrame) -> None:
    print("Selected baseline rows (non-DS3):")
    if non_ds3.empty:
        print("  (none)")
    else:
        print(non_ds3[["family", MODEL_COLUMN, "artifact", MAP_COLUMN, LATENCY_COLUMN]].to_string(index=False))

    print("\nSelected baseline rows (DS3):")
    if ds3.empty:
        print("  (none)")
    else:
        print(ds3[["family", MODEL_COLUMN, "artifact", MAP_COLUMN, LATENCY_COLUMN]].to_string(index=False))

    print("\nSoccerNet rows (best epoch mAP50-95):")
    if sn.empty:
        print("  (none)")
    else:
        print(sn[["family", MAP_COLUMN, LATENCY_COLUMN, "best_epoch", "source"]].to_string(index=False))

    missing_manual = sn[
        sn["family"].isin(["11l", "11x"]) & sn[MAP_COLUMN].isna()
    ]["family"].tolist()
    if missing_manual:
        print(
            "\nNote: fill SOCCERNET_MANUAL_MAP50_95 for: "
            + ", ".join(missing_manual)
        )


def _short_model_label(model_name: object) -> str:
    label = str(model_name).strip()
    if "/" in label or "\\" in label:
        label = Path(label).name
    for suffix in (".engine", ".pt"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label[:3]


def _add_model_labels(ax: object, labeled_df: pd.DataFrame) -> None:
    if labeled_df.empty:
        return

    line_df = labeled_df.sort_values("family_order")

    y_min, y_max = ax.get_ylim()
    if POINT_LABEL_MAP_Y < y_min or POINT_LABEL_MAP_Y > y_max:
        y_pad = (y_max - y_min) * 0.04 if y_max > y_min else 0.02
        ax.set_ylim(min(y_min, POINT_LABEL_MAP_Y - y_pad), max(y_max, POINT_LABEL_MAP_Y + y_pad))
        y_min, y_max = ax.get_ylim()

    guide_gap = max((y_max - y_min) * 0.03, 0.01)
    guide_bottom = max(y_min, POINT_LABEL_MAP_Y - guide_gap)
    guide_top = min(y_max, POINT_LABEL_MAP_Y + guide_gap)

    for _, row in line_df.iterrows():
        latency = float(row[LATENCY_COLUMN])
        if guide_bottom > y_min:
            ax.vlines(
                latency,
                y_min,
                guide_bottom,
                color="black",
                linestyles=":",
                linewidth=VERTICAL_GUIDE_LINE_WIDTH,
                alpha=VERTICAL_GUIDE_ALPHA,
                zorder=0,
            )
        if guide_top < y_max:
            ax.vlines(
                latency,
                guide_top,
                y_max,
                color="black",
                linestyles=":",
                linewidth=VERTICAL_GUIDE_LINE_WIDTH,
                alpha=VERTICAL_GUIDE_ALPHA,
                zorder=0,
            )
        ax.text(
            latency,
            POINT_LABEL_MAP_Y,
            _short_model_label(row[MODEL_COLUMN]),
            fontsize=POINT_LABEL_FONT_SIZE,
            color=POINT_LABEL_COLOR,
            va="center",
            ha="center",
        )


def _plot_lines(non_ds3: pd.DataFrame, ds3: pd.DataFrame, sn: pd.DataFrame) -> tuple[object, object]:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = IEEE_SERIF_STACK

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    series_specs = [
        ("Roboflow", non_ds3, SERIES_COLORS["Roboflow"]),
        ("Football Analytics", ds3, SERIES_COLORS["Football Analytics"]),
        ("Soccernet", sn.dropna(subset=[MAP_COLUMN]), SERIES_COLORS["Soccernet"]),
    ]

    for label, df_line, color in series_specs:
        if df_line.empty:
            continue
        df_line = df_line.sort_values("family_order")
        ax.plot(
            df_line[LATENCY_COLUMN],
            df_line[MAP_COLUMN],
            marker="o",
            linewidth=PLOT_LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            color=color,
            label=label,
        )

    ax.set_xlabel("Latency ms", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=2)
    ax.set_ylabel(MAP_COLUMN, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=2)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, width=AXIS_BOX_LINE_WIDTH)
    ax.set_xlim(LATENCY_X_MIN, LATENCY_X_MAX)
    ax.set_xticks(LATENCY_X_TICKS)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_BOX_LINE_WIDTH)
    ax.grid(True, linestyle="--", alpha=0.4)
    _add_model_labels(ax, ds3)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_order = [label for label in ("Roboflow", "Football Analytics", "Soccernet") if label in handle_by_label]
    if legend_order:
        ax.legend(
            [handle_by_label[label] for label in legend_order],
            legend_order,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            loc="lower right",
            # bbox_to_anchor=(0.5, LEGEND_Y_ANCHOR),
            ncol=len(legend_order),
            columnspacing=0.9,
            handlelength=1.5,
        )

    fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
    fig.subplots_adjust(
        left=LAYOUT_LEFT,
        right=LAYOUT_RIGHT,
        top=LAYOUT_TOP,
        bottom=LAYOUT_BOTTOM,
    )
    fig.set_size_inches(IEEE_SINGLE_COLUMN_WIDTH_IN, FIG_HEIGHT_IN, forward=True)
    return fig, ax


def main() -> int:
    try:
        df = _load_model_summary(MODEL_SUMMARY_CSV)
        non_ds3 = _select_baseline_family_rows(df, is_ds3=False)
        ds3 = _select_baseline_family_rows(df, is_ds3=True)
        sn = _build_soccernet_rows(non_ds3, ds3)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if non_ds3.empty and ds3.empty and sn.empty:
        print("Error: no usable rows found for plotting.", file=sys.stderr)
        return 1

    _print_selection_summary(non_ds3, ds3, sn)

    try:
        fig, _ = _plot_lines(non_ds3, ds3, sn)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Error plotting figure: {exc}", file=sys.stderr)
        return 1

    plt.show()
    prompt_save_figure(fig, default_name="map50_95_vs_latency_11_non_ds3_ds3_soccernet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
