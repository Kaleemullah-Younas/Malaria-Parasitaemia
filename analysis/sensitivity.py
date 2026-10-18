"""Sensitivity analyses (plan: 09). Not headline results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from analysis.config import DATA_DIR, MICROSCOPY_HEADLINE, SEED, TABLES, ensure_dirs
from analysis.models_loco import oe_ratio, run_loco, weighted_auc


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


def antimalarial_exclusion(analytic: pd.DataFrame | None = None) -> pd.DataFrame:
    """Count afebrile children dropped because recent antimalarial = 1.

    Locked counts: 93,334 afebrile with valid RDT → 138 excluded (0.15%) → 93,196.
    """
    ensure_dirs()
    flow_path = TABLES / "table2_flow.csv"
    if not flow_path.exists():
        raise FileNotFoundError(flow_path)
    flow = pd.read_csv(flow_path)
    s4 = flow.loc[flow["step"] == "4_afebrile_h22eq0"].groupby("country", as_index=False)[["n", "rdt_pos"]].sum()
    s5 = flow.loc[flow["step"] == "5_afebrile_no_recent_antimalarial"].groupby("country", as_index=False)[["n", "rdt_pos"]].sum()
    out = s4.merge(s5, on="country", suffixes=("_pre", "_post"))
    out["n_excluded"] = out["n_pre"] - out["n_post"]
    out["rdt_pos_excluded"] = out["rdt_pos_pre"] - out["rdt_pos_post"]
    total = pd.DataFrame(
        {
            "country": ["TOTAL"],
            "n_pre": [int(out["n_pre"].sum())],
            "rdt_pos_pre": [int(out["rdt_pos_pre"].sum())],
            "n_post": [int(out["n_post"].sum())],
            "rdt_pos_post": [int(out["rdt_pos_post"].sum())],
            "n_excluded": [int(out["n_excluded"].sum())],
            "rdt_pos_excluded": [int(out["rdt_pos_excluded"].sum())],
        }
    )
    tab = pd.concat([out, total], ignore_index=True)
    tab["pct_excluded"] = 100 * tab["n_excluded"] / tab["n_pre"]
    tab.to_csv(TABLES / "antimalarial_exclusion.csv", index=False)
    n_pre, n_ex, n_post = int(total.n_pre.iloc[0]), int(total.n_excluded.iloc[0]), int(total.n_post.iloc[0])
    print(f"Antimalarial exclusion: {n_pre:,} -> {n_ex} excluded ({100 * n_ex / n_pre:.2f}%) -> {n_post:,}")
    return tab


def equal_country_weight_loco(analytic: pd.DataFrame | None = None) -> pd.DataFrame:
    """LOCO with equal total weight per training country. Does not touch primary Table 2."""
    ensure_dirs()
    if analytic is None:
        analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    res = run_loco(
        analytic,
        tag="equal_country_weight",
        tune=False,
        equal_country_weight=True,
        reuse_primary_params=True,
        overwrite=True,
        n_boot=200,
        compute_aux=False,
    )
    res["metrics"].to_csv(TABLES / "metrics_loco_equal_country_weight.csv", index=False)
    return res["metrics"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _intercept_recal(y: np.ndarray, lp: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    lp = np.asarray(lp, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(y) & np.isfinite(lp) & (w > 0)
    y, lp, w = y[mask], lp[mask], w[mask]

    def nll(a: float) -> float:
        pn = np.clip(_expit(lp + a), 1e-12, 1 - 1e-12)
        return float(-np.sum(w * (y * np.log(pn) + (1 - y) * np.log(1 - pn))))

    res = minimize_scalar(nll, bounds=(-12, 12), method="bounded")
    return float(res.x)


def _platt_recal(y: np.ndarray, lp: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    lp = np.asarray(lp, dtype=float).reshape(-1, 1)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(y) & np.isfinite(lp.ravel()) & (w > 0)
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=500)
    lr.fit(lp[mask], y[mask], sample_weight=w[mask])
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def recalibration_sensitivity(preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Intercept-only and Platt recalibration on a 50/50 split inside each held-out country.

    This is a sensitivity, not a transport test: the held-out country is split after
    LOCO scores are already generated. AUC is nearly unchanged under a monotone map;
    O/E is the quantity that should move.
    """
    ensure_dirs()
    if preds is None:
        path = DATA_DIR / "predictions_primary.parquet"
        preds = pd.read_parquet(path)
    rng = np.random.default_rng(SEED)
    rows = []
    for country, g in preds.groupby("country"):
        g = g.copy()
        if "p_lgb" not in g.columns or (g["rdt"] == 1).sum() < 20:
            continue
        if "psu" in g.columns and g["psu"].nunique() >= 4:
            psus = np.asarray(g["psu"].astype(str).unique())
            rng.shuffle(psus)
            cut = max(1, len(psus) // 2)
            cal_psu = set(psus[:cut])
            cal_mask = g["psu"].astype(str).isin(cal_psu)
        else:
            idx = np.arange(len(g))
            cal_idx, _ = train_test_split(
                idx,
                test_size=0.5,
                random_state=SEED,
                stratify=g["rdt"].to_numpy() if g["rdt"].nunique() == 2 else None,
            )
            cal_mask = np.zeros(len(g), dtype=bool)
            cal_mask[cal_idx] = True
            cal_mask = pd.Series(cal_mask, index=g.index)
        cal = g.loc[cal_mask]
        ev = g.loc[~cal_mask]
        if len(ev) < 30 or (ev["rdt"] == 1).sum() < 5:
            continue
        y_cal = cal["rdt"].to_numpy()
        w_cal = pd.to_numeric(cal["weight"], errors="coerce").fillna(0).to_numpy()
        y_ev = ev["rdt"].to_numpy()
        w_ev = pd.to_numeric(ev["weight"], errors="coerce").fillna(0).to_numpy()
        p0 = ev["p_lgb"].to_numpy()
        lp_cal = _logit(cal["p_lgb"].to_numpy())
        lp_ev = _logit(p0)
        a = _intercept_recal(y_cal, lp_cal, w_cal)
        p_int = _expit(lp_ev + a)
        slope, intercept = _platt_recal(y_cal, lp_cal, w_cal)
        p_platt = _expit(slope * lp_ev + intercept)
        rec = {
            "country": country,
            "n_cal": int(len(cal)),
            "n_eval": int(len(ev)),
            "rdt_pos_eval": int((ev["rdt"] == 1).sum()),
            "auc_pre": weighted_auc(y_ev, p0, w_ev),
            "oe_pre": oe_ratio(y_ev, p0, w_ev),
            "auc_intercept": weighted_auc(y_ev, p_int, w_ev),
            "oe_intercept": oe_ratio(y_ev, p_int, w_ev),
            "intercept_shift": a,
            "auc_platt": weighted_auc(y_ev, p_platt, w_ev),
            "oe_platt": oe_ratio(y_ev, p_platt, w_ev),
            "platt_slope": slope,
            "platt_intercept": intercept,
        }
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "sensitivity_recalibration.csv", index=False)
    print(out.to_string(index=False))
    return out


def run_sensitivity() -> None:
    ensure_dirs()
    analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    microscopy_sensitivity(analytic)
    hb_sensitivity(analytic)
    drop_incomplete_surveys(analytic)
    fever_dontknow_sensitivity(analytic)


def run_revision_sensitivities(analytic: pd.DataFrame | None = None) -> None:
    """New JPMedAI revision analyses. Does not overwrite primary LOCO tables."""
    ensure_dirs()
    if analytic is None:
        analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    antimalarial_exclusion(analytic)
    equal_country_weight_loco(analytic)
    recalibration_sensitivity()


if __name__ == "__main__":
    run_sensitivity()
