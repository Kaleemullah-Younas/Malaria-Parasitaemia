"""Predictor recodes, missingness, survey design IDs (plan: 04_features.py / 05)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.config import (
    DATA_DIR,
    HB_FEATURE,
    IMPROVED_TOILET,
    IMPROVED_WATER,
    NET_USED_CODES,
    PRIMARY_FEATURES,
    TABLES,
    ensure_dirs,
)
from analysis.util import nest_ids


def _housing_grp(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series("missing", index=series.index, dtype="object")
    out = out.mask(x.between(10, 19), "natural")
    out = out.mask(x.between(20, 29), "rudimentary")
    out = out.mask(x.between(30, 39), "finished")
    out = out.mask(x.isna(), "missing")
    out = out.mask(~x.isna() & ~x.between(10, 39), "other")
    return out


def recode_features(df: pd.DataFrame) -> pd.DataFrame:
    out = nest_ids(df)
    out["wealth"] = pd.to_numeric(out.get("v190"), errors="coerce")
    out["urban"] = np.where(pd.to_numeric(out.get("v025"), errors="coerce") == 1, 1, np.where(pd.to_numeric(out.get("v025"), errors="coerce") == 2, 0, np.nan))
    out["educ"] = pd.to_numeric(out.get("v106"), errors="coerce")
    out["educ"] = out["educ"].where(out["educ"].between(0, 3), np.nan)
    hh = pd.to_numeric(out.get("v136"), errors="coerce")
    if "hv009" in out.columns:
        hh = hh.fillna(pd.to_numeric(out["hv009"], errors="coerce"))
    out["hhsize"] = hh
    water = pd.to_numeric(out.get("v113"), errors="coerce")
    toilet = pd.to_numeric(out.get("v116"), errors="coerce")
    out["water_improved"] = np.where(water.isna() | water.isin([96, 99]), np.nan, water.isin(IMPROVED_WATER).astype(float))
    out["toilet_improved"] = np.where(toilet.isna() | toilet.isin([96, 99]), np.nan, toilet.isin(IMPROVED_TOILET).astype(float))
    nets = pd.to_numeric(out.get("hml1"), errors="coerce")
    out["net_own"] = np.where(nets.isna(), np.nan, (nets > 0).astype(float))
    hml12 = pd.to_numeric(out.get("hml12"), errors="coerce")
    out["net_used"] = np.where(hml12.isna(), np.nan, hml12.isin(NET_USED_CODES).astype(float))
    out["floor_grp"] = _housing_grp(out["hv213"]) if "hv213" in out.columns else pd.Series("missing", index=out.index)
    out["wall_grp"] = _housing_grp(out["hv214"]) if "hv214" in out.columns else pd.Series("missing", index=out.index)
    out["roof_grp"] = _housing_grp(out["hv215"]) if "hv215" in out.columns else pd.Series("missing", index=out.index)
    out["female"] = np.where(pd.to_numeric(out.get("b4"), errors="coerce") == 2, 1, np.where(pd.to_numeric(out.get("b4"), errors="coerce") == 1, 0, np.nan))
    month = pd.to_numeric(out.get("v006"), errors="coerce")
    if "hv006" in out.columns:
        month = month.fillna(pd.to_numeric(out["hv006"], errors="coerce"))
    out["month"] = month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["lat_band"] = out["lat_band"].fillna("missing").astype(str)
    ng = pd.to_numeric(out.get("sahel_nigeria"), errors="coerce")
    out["sahel_nigeria"] = np.where(out["country"] != "Nigeria", 0.0, ng)
    if "hw53" in out.columns:
        hb = pd.to_numeric(out["hw53"], errors="coerce")
        hb = hb.where(~hb.isin([999, 9999, 99]), np.nan)
        # Stored with one implied decimal when values look like 30–200.
        hb = np.where(hb > 25, hb / 10.0, hb)
        out[HB_FEATURE] = hb
    else:
        out[HB_FEATURE] = np.nan
    if "hv253" in out.columns:
        irs = pd.to_numeric(out["hv253"], errors="coerce")
        out["irs"] = np.where(irs.isin([0, 1]), irs, np.nan)
    else:
        out["irs"] = np.nan
    return out


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["country", "survey_year"]
    cols = [c for c in PRIMARY_FEATURES + [HB_FEATURE, "irs"] if c in df.columns]
    rows = []
    for (country, year), g in df.groupby(keys, dropna=False):
        rec = {"country": country, "survey_year": year, "n": int(len(g))}
        for c in cols:
            rec[f"{c}_miss_pct"] = float(g[c].isna().mean() * 100) if c in g.columns else 100.0
        rec["primary_mean_miss_pct"] = float(np.mean([rec[f"{c}_miss_pct"] for c in PRIMARY_FEATURES if f"{c}_miss_pct" in rec]))
        rec["drop_gt20pct_predictors"] = int(rec["primary_mean_miss_pct"] > 20)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_features(pop: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if pop is None:
        pop = pd.read_parquet(DATA_DIR / "study_population.parquet")
    feat = recode_features(pop)
    miss = missingness_table(feat.loc[feat["train_sample"]])
    miss.to_csv(TABLES / "missingness_by_survey.csv", index=False)
    feat.to_parquet(DATA_DIR / "analytic.parquet", index=False)
    (TABLES / "missingness_summary.json").write_text(
        json.dumps(
            {
                "n_train": int(feat["train_sample"].sum()),
                "n_eval": int(feat["eval_sample"].sum()),
                "surveys_gt20pct": miss.loc[miss["drop_gt20pct_predictors"] == 1, ["country", "survey_year"]].to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(miss[["country", "survey_year", "n", "primary_mean_miss_pct", "drop_gt20pct_predictors"]].to_string(index=False))
    return feat


if __name__ == "__main__":
    run_features()
