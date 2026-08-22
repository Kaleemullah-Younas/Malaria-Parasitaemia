"""Sensitivity analyses (plan: 09). Not headline results."""

from __future__ import annotations

import pandas as pd

from analysis.config import DATA_DIR, MICROSCOPY_HEADLINE, TABLES, ensure_dirs
from analysis.models_loco import run_loco, weighted_auc


def microscopy_sensitivity(analytic: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=analytic.index)
    for country, year in MICROSCOPY_HEADLINE:
        mask |= (analytic["country"] == country) & (analytic["survey_year"].astype(str) == str(year))
    sub = analytic.loc[mask].copy()
    sub = sub[sub["microscopy"].isin([0, 1])]
    sub["rdt"] = sub["microscopy"]
    sub["train_sample"] = sub["train_sample"] & sub["microscopy"].isin([0, 1])
    sub["eval_sample"] = sub["eval_sample"] & sub["microscopy"].isin([0, 1])
    print(f"Microscopy sensitivity N train={int(sub.train_sample.sum())} countries={list(sub.country.unique())}")
    if sub.loc[sub.train_sample, "country"].nunique() < 2:
        out = pd.DataFrame({"note": ["Need two countries with microscopy for LOCO"]})
        out.to_csv(TABLES / "sensitivity_microscopy.csv", index=False)
        return out
    res = run_loco(sub, tag="microscopy", tune=False)
    res["metrics"].to_csv(TABLES / "sensitivity_microscopy.csv", index=False)
    return res["metrics"]


def hb_sensitivity(analytic: pd.DataFrame) -> pd.DataFrame:
    sub = analytic.copy()
    print("Hb-in-model sensitivity (not primary)")
    res = run_loco(sub, extra_num=["hb_gdl"], tag="hb", tune=False)
    res["metrics"].to_csv(TABLES / "sensitivity_hb.csv", index=False)
    return res["metrics"]


def drop_incomplete_surveys(analytic: pd.DataFrame) -> pd.DataFrame:
    miss = pd.read_csv(TABLES / "missingness_by_survey.csv")
    drop = miss.loc[miss["drop_gt20pct_predictors"] == 1, ["country", "survey_year"]]
    drop.to_csv(TABLES / "sensitivity_dropped_surveys.csv", index=False)
    if drop.empty:
        print("No surveys with >20% mean primary-predictor missingness")
        pd.DataFrame({"note": ["none dropped"]}).to_csv(TABLES / "sensitivity_incomplete.csv", index=False)
        return pd.DataFrame()
    key = set(zip(drop.country, drop.survey_year.astype(str)))
    keep = analytic[~analytic.apply(lambda r: (r.country, str(r.survey_year)) in key, axis=1)].copy()
    res = run_loco(keep, tag="incomplete_drop", tune=False)
    res["metrics"].to_csv(TABLES / "sensitivity_incomplete.csv", index=False)
    return res["metrics"]


def fever_dontknow_sensitivity(analytic: pd.DataFrame) -> pd.DataFrame:
    """Primary excludes h22==8. Alternate: treat don't-know as afebrile."""
    alt = analytic.copy()
    h22 = pd.to_numeric(alt.get("h22"), errors="coerce")
    extra = alt["in_age"] & alt["valid_rdt"] & h22.eq(8)
    n_extra = int(extra.sum())
    alt.loc[extra, "afebrile"] = True
    alt.loc[extra, "fever"] = 0
    alt.loc[extra, "train_sample"] = extra
    alt.loc[extra, "eval_sample"] = extra | alt["eval_sample"]
    print(f"Alternate fever coding: added {n_extra} children with h22==8 as afebrile")
    pd.DataFrame(
        {
            "definition": ["primary_h22eq0", "alternate_h22in_0_or_8"],
            "n_train": [int(analytic["train_sample"].sum()), int(alt["train_sample"].sum())],
            "rdt_pos_train": [
                int((analytic.loc[analytic.train_sample, "rdt"] == 1).sum()),
                int((alt.loc[alt.train_sample, "rdt"] == 1).sum()),
            ],
            "n_added_dontknow": [0, n_extra],
        }
    ).to_csv(TABLES / "sensitivity_fever_window.csv", index=False)
    # One-week recall does not exist in these files; this is the pre-specified substitute.
    if n_extra >= 50:
        res = run_loco(alt, tag="fever_dk", tune=False)
        res["metrics"].to_csv(TABLES / "sensitivity_fever_dk_metrics.csv", index=False)
        return res["metrics"]
    return pd.DataFrame()


def run_sensitivity() -> None:
    ensure_dirs()
    analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    microscopy_sensitivity(analytic)
    hb_sensitivity(analytic)
    drop_incomplete_surveys(analytic)
    fever_dontknow_sensitivity(analytic)


if __name__ == "__main__":
    run_sensitivity()
