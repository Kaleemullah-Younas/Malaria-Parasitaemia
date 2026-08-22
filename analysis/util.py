"""Small helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def df_to_md(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "_empty_"
    view = df.head(max_rows)
    cols = [str(c) for c in view.columns]

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v).replace("|", "/")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {len(df) - max_rows} more rows |")
    return "\n".join(lines)


def weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        return float(np.nanmean(y))
    return float(np.average(y, weights=w))


def nest_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    year = out.get("survey_year", pd.Series("", index=out.index)).astype(str)
    country = out["country"].astype(str)
    psu_raw = out["v001"] if "v001" in out.columns else out.get("hv001")
    strat_raw = out["v022"] if "v022" in out.columns else pd.Series(np.nan, index=out.index)
    if "v023" in out.columns:
        strat_raw = strat_raw.fillna(out["v023"])
    out["psu"] = country + "|" + year + "|" + psu_raw.astype("Int64").astype(str)
    out["stratum"] = country + "|" + year + "|" + strat_raw.astype("Int64").astype(str)
    if "weight" not in out.columns:
        out["weight"] = pd.to_numeric(out.get("v005"), errors="coerce") / 1_000_000
    if "hv005" in out.columns:
        alt = pd.to_numeric(out["hv005"], errors="coerce") / 1_000_000
        out["weight"] = out["weight"].fillna(alt)
    out["weight"] = out["weight"].fillna(0).clip(lower=0)
    return out
