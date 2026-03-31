from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

MAP_COLUMN = "mAP50-95"
FPS_COLUMN = "FPS (avg)"
REQUIRED_COLS = ["Model", MAP_COLUMN, FPS_COLUMN]

FAMILY_ORDER = ["11n", "11s", "11m", "11l", "11x"]
FAMILY_INDEX = {fam: idx for idx, fam in enumerate(FAMILY_ORDER)}

_RE_BASELINE = re.compile(r"^(11[a-z])_ds3_baseline\.engine$", re.IGNORECASE)
_RE_PRUNED = re.compile(r"^(11[a-z])_ds3_p(\d+)\.engine$", re.IGNORECASE)
_RE_QUANT = re.compile(r"^(11[a-z])_ds3_(fp16|int8)\.engine$", re.IGNORECASE)
_RE_PRUNED_QUANT = re.compile(r"^(11[a-z])_ds3_p(\d+)_(fp16|int8)\.engine$", re.IGNORECASE)


def parse_model_info(model: str) -> Optional[Tuple[str, str, Optional[int], Optional[str]]]:
    """
    Return (family, kind, prune_pct, quant_level) or None if not a DS3 11-series engine model.
    kinds: baseline, pruned, quant, pruned_quant
    """
    text = str(model).strip()
    if not text:
        return None

    m = _RE_BASELINE.match(text)
    if m:
        return (m.group(1).lower(), "baseline", None, None)

    m = _RE_PRUNED_QUANT.match(text)
    if m:
        return (m.group(1).lower(), "pruned_quant", int(m.group(2)), m.group(3).lower())

    m = _RE_PRUNED.match(text)
    if m:
        return (m.group(1).lower(), "pruned", int(m.group(2)), None)

    m = _RE_QUANT.match(text)
    if m:
        return (m.group(1).lower(), "quant", None, m.group(2).lower())

    return None


def load_ds3_11_engine(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    parsed = df["Model"].apply(parse_model_info)
    mask = parsed.notnull()
    df = df[mask].copy()
    if df.empty:
        return df

    parsed_vals = parsed[mask].tolist()
    parsed_df = pd.DataFrame(parsed_vals, columns=["family", "kind", "prune_pct", "quant_level"], index=df.index)
    df = pd.concat([df, parsed_df], axis=1)

    df[MAP_COLUMN] = pd.to_numeric(df[MAP_COLUMN], errors="coerce")
    df[FPS_COLUMN] = pd.to_numeric(df[FPS_COLUMN], errors="coerce")
    df = df.dropna(subset=[MAP_COLUMN, FPS_COLUMN]).copy()
    df["family_order"] = df["family"].map(FAMILY_INDEX).fillna(999).astype(int)
    return df
