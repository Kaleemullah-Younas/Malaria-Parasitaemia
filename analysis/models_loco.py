"""Leave-one-country-out models, survey-weighted metrics, incremental DCA (plan: 06–07)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.config import (
    CALIBRATION_BINS,
    DATA_DIR,
    DCA_THRESHOLDS,
    ELASTICNET_GRID,
    LGBM_GRID,
    MODELS_DIR,
    N_BOOTSTRAP,
    PRIMARY_FEATURES,
    SEED,
    SUBGROUP_RDT_POS_FLAG_N,
    TABLES,
    ensure_dirs,
)

NUM_COLS = [
    "wealth",
    "urban",
    "educ",
    "hhsize",
    "water_improved",
    "toilet_improved",
    "net_own",
    "net_used",
    "age_months",
    "female",
    "month_sin",
    "month_cos",
    "sahel_nigeria",
]
CAT_COLS = ["floor_grp", "wall_grp", "roof_grp", "lat_band"]


def equal_country_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale native DHS weights so each country has the same total weight mass.

    Evaluation metrics on a single held-out country are invariant to a constant
    scale, so this is applied only to training folds (see ``run_loco``).
    """
    out = df.copy()
    w = pd.to_numeric(out["weight"], errors="coerce").fillna(0).clip(lower=0.0)
    tot = w.groupby(out["country"]).transform("sum")
    n_c = max(int(out["country"].nunique()), 1)
    grand = float(w.sum())
    scale = np.where(tot > 0, (grand / n_c) / tot, 0.0)
    out["weight"] = w.to_numpy(dtype=float) * scale
    return out


def _xy(df: pd.DataFrame, extra_num: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = [c for c in NUM_COLS + CAT_COLS if c in df.columns]
    if extra_num:
        cols = cols + [c for c in extra_num if c in df.columns and c not in cols]
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("missing")
    y = pd.to_numeric(df["rdt"], errors="coerce").to_numpy()
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0).to_numpy()
    w = np.where(w > 0, w, 0.0)
    return X, y, w


def make_preprocessor(num_cols: list[str], cat_cols: list[str], iterative: bool = True) -> ColumnTransformer:
    if iterative:
        num_imp = IterativeImputer(random_state=SEED, max_iter=5, skip_complete=True)
    else:
        num_imp = SimpleImputer(strategy="median")
    num = Pipeline([("impute", num_imp), ("scale", StandardScaler())])
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num, num_cols), ("cat", cat, cat_cols)],
        remainder="drop",
    )


def weighted_auc(y, p, w) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    w = np.asarray(w)
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w >= 0)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y[mask], p[mask], sample_weight=w[mask]))
    except ValueError:
        return float("nan")


def oe_ratio(y, p, w) -> float:
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    exp = np.average(p[mask], weights=w[mask])
    obs = np.average(y[mask], weights=w[mask])
    if exp <= 0:
        return float("nan")
    return float(obs / exp)


def calib_slope_intercept(y, p, w) -> tuple[float, float]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & (w > 0)
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return float("nan"), float("nan")
    lp = np.log(p[mask] / (1 - p[mask])).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=500)
        lr.fit(lp, y[mask], sample_weight=w[mask])
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def calibration_bins(y, p, w, n_bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & (w > 0)
    y, p, w = y[mask], p[mask], w[mask]
    if len(y) < n_bins:
        return pd.DataFrame()
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = 0, 1
    rows = []
    for i in range(n_bins):
        m = (p >= qs[i]) & (p <= qs[i + 1] if i == n_bins - 1 else p < qs[i + 1])
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": i + 1,
                "n": int(m.sum()),
                "pred": float(np.average(p[m], weights=w[m])),
                "obs": float(np.average(y[m], weights=w[m])),
            }
        )
    return pd.DataFrame(rows)


def net_benefit(y, tested, w, t: float) -> float:
    y = np.asarray(y, dtype=float)
    tested = np.asarray(tested, dtype=bool)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    y, tested, w = y[mask], tested[mask], w[mask]
    if w.sum() <= 0:
        return float("nan")
    tp = np.average((y == 1) & tested, weights=w)
    fp = np.average((y == 0) & tested, weights=w)
    return float(tp - fp * (t / (1 - t)))


def dca_curves(y, fever, p_afebrile, w, afebrile_mask) -> pd.DataFrame:
    rows = []
    fever = np.asarray(fever) == 1
    p = np.asarray(p_afebrile, dtype=float)
    afe = np.asarray(afebrile_mask, dtype=bool)
    for t in DCA_THRESHOLDS:
        combo = fever | (afe & np.isfinite(p) & (p >= t))
        rows.append(
            {
                "threshold": t,
                "nb_none": 0.0,
                "nb_all": net_benefit(y, np.ones(len(y), dtype=bool), w, t),
                "nb_fever": net_benefit(y, fever, w, t),
                "nb_fever_plus_model": net_benefit(y, combo, w, t),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_ci(df: pd.DataFrame, pcol: str, n_boot: int = N_BOOTSTRAP) -> dict:
    rng = np.random.default_rng(SEED)
    y = pd.to_numeric(df["rdt"], errors="coerce").to_numpy()
    p = pd.to_numeric(df[pcol], errors="coerce").to_numpy()
    w = pd.to_numeric(df["weight"], errors="coerce").to_numpy()
    psu = pd.factorize(df["psu"].astype(str))[0]
    n_psu = int(psu.max()) + 1 if len(psu) else 0
    if n_psu < 5 or n_boot < 1:
        return {"auc_ci_low": float("nan"), "auc_ci_high": float("nan"), "oe_ci_low": float("nan"), "oe_ci_high": float("nan")}
    groups = [np.flatnonzero(psu == i) for i in range(n_psu)]
    aucs, oes = [], []
    for _ in range(n_boot):
        draw = rng.integers(0, n_psu, size=n_psu)
        idx = np.concatenate([groups[i] for i in draw], axis=0)
        aucs.append(weighted_auc(y[idx], p[idx], w[idx]))
        oes.append(oe_ratio(y[idx], p[idx], w[idx]))

    def ci(vals):
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals) < 20:
            return float("nan"), float("nan")
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    a_lo, a_hi = ci(aucs)
    o_lo, o_hi = ci(oes)
    return {"auc_ci_low": a_lo, "auc_ci_high": a_hi, "oe_ci_low": o_lo, "oe_ci_high": o_hi}


def _fit_enet(X, y, w, num_cols, cat_cols, params) -> Pipeline:
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(num_cols, cat_cols, iterative=False)),
            (
                "clf",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=params["l1_ratio"],
                        C=params["C"],
                        max_iter=800,
                        random_state=SEED,
                    ),
            ),
        ]
    )
    pipe.fit(X, y, clf__sample_weight=w)
    return pipe


def _fit_lgb(X, y, w, num_cols, cat_cols, params) -> Pipeline:
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(num_cols, cat_cols, iterative=False)),
            (
                "clf",
                lgb.LGBMClassifier(
                    objective="binary",
                    random_state=SEED,
                    verbosity=-1,
                    **params,
                ),
            ),
        ]
    )
    pipe.fit(X, y, clf__sample_weight=w)
    return pipe


@dataclass
class FoldResult:
    country: str
    model: str
    params: dict
    auc: float
    oe: float
    slope: float
    intercept: float


def inner_select(train: pd.DataFrame, extra_num: list[str] | None = None, equal_country_weight: bool = False):
    countries = sorted(train["country"].unique())
    num_cols = [c for c in NUM_COLS if c in train.columns]
    if extra_num:
        num_cols = num_cols + [c for c in extra_num if c in train.columns]
    cat_cols = [c for c in CAT_COLS if c in train.columns]
    best_enet, best_enet_s = {"l1_ratio": 0.5, "C": 1.0}, float("nan")
    best_lgb, best_lgb_s = LGBM_GRID[0], -1
    for params in LGBM_GRID:
        scores = []
        for val_c in countries:
            tr = train[train["country"] != val_c]
            if equal_country_weight:
                tr = equal_country_weights(tr)
            va = train[train["country"] == val_c]
            if va["rdt"].nunique() < 2 or (va["rdt"] == 1).sum() < 10:
                continue
            Xtr, ytr, wtr = _xy(tr, extra_num)
            Xva, yva, wva = _xy(va, extra_num)
            try:
                m = _fit_lgb(Xtr, ytr, wtr, num_cols, cat_cols, params)
                p = m.predict_proba(Xva)[:, 1]
                scores.append(weighted_auc(yva, p, wva))
            except Exception:
                continue
        s = float(np.nanmean(scores)) if scores else float("nan")
        if np.isfinite(s) and s > best_lgb_s:
            best_lgb, best_lgb_s = params, s
    if best_enet is None:
        best_enet = {"l1_ratio": 0.5, "C": 1.0}
    return best_enet, best_lgb, num_cols, cat_cols


def _predict_fold(train, test, extra_num=None, equal_country_weight: bool = False):
    best_enet, best_lgb, num_cols, cat_cols = inner_select(
        train, extra_num, equal_country_weight=equal_country_weight
    )
    Xtr, ytr, wtr = _xy(train, extra_num)
    Xte, yte, wte = _xy(test, extra_num)
    enet = _fit_enet(Xtr, ytr, wtr, num_cols, cat_cols, best_enet)
    gbm = _fit_lgb(Xtr, ytr, wtr, num_cols, cat_cols, best_lgb)
    p_enet = enet.predict_proba(Xte)[:, 1]
    p_lgb = gbm.predict_proba(Xte)[:, 1]
    return p_enet, p_lgb, enet, gbm, best_enet, best_lgb


def _metrics_block(df, pcol, model, country, extra=None, n_boot: int = N_BOOTSTRAP) -> dict:
    y, p, w = df["rdt"].to_numpy(), df[pcol].to_numpy(), df["weight"].to_numpy()
    slope, intercept = calib_slope_intercept(y, p, w)
    rec = {
        "country": country,
        "model": model,
        "n": int(len(df)),
        "rdt_pos": int((df["rdt"] == 1).sum()),
        "auc": weighted_auc(y, p, w),
        "oe": oe_ratio(y, p, w),
        "slope": slope,
        "intercept": intercept,
        "flag_pos_lt50": int((df["rdt"] == 1).sum() < SUBGROUP_RDT_POS_FLAG_N),
    }
    if extra:
        rec.update(extra)
    rec.update(cluster_bootstrap_ci(df, pcol, n_boot=n_boot))
    return rec


def _params_from_primary(country: str) -> tuple[dict, dict]:
    path = TABLES / "metrics_loco_primary.csv"
    if not path.exists():
        return {"l1_ratio": 0.5, "C": 1.0}, LGBM_GRID[1]
    m = pd.read_csv(path)
    en = m[(m["country"] == country) & (m["model"] == "elasticnet")]
    lg = m[(m["country"] == country) & (m["model"] == "lightgbm")]
    pe = json.loads(en.iloc[0]["params"]) if len(en) and pd.notna(en.iloc[0].get("params")) else {"l1_ratio": 0.5, "C": 1.0}
    pl = json.loads(lg.iloc[0]["params"]) if len(lg) and pd.notna(lg.iloc[0].get("params")) else LGBM_GRID[1]
    return pe, pl


def run_loco(
    analytic: pd.DataFrame | None = None,
    extra_num: list[str] | None = None,
    tag: str = "primary",
    tune: bool = True,
    equal_country_weight: bool = False,
    overwrite: bool = False,
    reuse_primary_params: bool = False,
    n_boot: int | None = None,
    compute_aux: bool = True,
) -> dict:
    ensure_dirs()
    dest = TABLES / f"metrics_loco_{tag}.csv"
    if tag == "primary" and dest.exists() and not overwrite:
        print(f"Primary LOCO metrics already exist at {dest}; skipping to protect Table 2. Pass overwrite=True to refit.")
        preds_path = DATA_DIR / "predictions_primary.parquet"
        return {
            "preds": pd.read_parquet(preds_path) if preds_path.exists() else pd.DataFrame(),
            "metrics": pd.read_csv(dest),
            "calib": pd.read_csv(TABLES / "calibration_bins_primary.csv") if (TABLES / "calibration_bins_primary.csv").exists() else pd.DataFrame(),
            "dca": pd.read_csv(TABLES / "dca_primary.csv") if (TABLES / "dca_primary.csv").exists() else pd.DataFrame(),
            "sub": pd.read_csv(TABLES / "subgroup_metrics_primary.csv") if (TABLES / "subgroup_metrics_primary.csv").exists() else pd.DataFrame(),
            "gbm": None,
        }
    if n_boot is None:
        n_boot = N_BOOTSTRAP
    if analytic is None:
        analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    train_all = analytic.loc[analytic["train_sample"]].copy()
    eval_all = analytic.loc[analytic["eval_sample"]].copy()
    countries = sorted(train_all["country"].unique())
    pred_frames = []
    metric_rows = []
    calib_rows = []
    dca_rows = []
    subgroup_rows = []
    last_gbm = None
    last_cols = None
    for held in countries:
        print(f"[{tag}] LOCO hold-out {held}")
        tr = train_all[train_all["country"] != held]
        if equal_country_weight:
            tr = equal_country_weights(tr)
        te_trainish = train_all[train_all["country"] == held]
        te_eval = eval_all[eval_all["country"] == held]
        if (te_trainish["rdt"] == 1).sum() < 10:
            print(f"  skip {held}: too few afebrile RDT+")
            continue
        if reuse_primary_params:
            pe, pl = _params_from_primary(held)
            num_cols = [c for c in NUM_COLS if c in tr.columns]
            if extra_num:
                num_cols = num_cols + [c for c in extra_num if c in tr.columns]
            cat_cols = [c for c in CAT_COLS if c in tr.columns]
            Xtr, ytr, wtr = _xy(tr, extra_num)
            Xte, yte, wte = _xy(te_trainish, extra_num)
            enet = _fit_enet(Xtr, ytr, wtr, num_cols, cat_cols, pe)
            gbm = _fit_lgb(Xtr, ytr, wtr, num_cols, cat_cols, pl)
            p_enet = enet.predict_proba(Xte)[:, 1]
            p_lgb = gbm.predict_proba(Xte)[:, 1]
        elif tune:
            p_enet, p_lgb, enet, gbm, pe, pl = _predict_fold(
                tr, te_trainish, extra_num, equal_country_weight=equal_country_weight
            )
        else:
            num_cols = [c for c in NUM_COLS if c in tr.columns]
            if extra_num:
                num_cols = num_cols + [c for c in extra_num if c in tr.columns]
            cat_cols = [c for c in CAT_COLS if c in tr.columns]
            pe = {"l1_ratio": 0.5, "C": 1.0}
            pl = LGBM_GRID[1]
            Xtr, ytr, wtr = _xy(tr, extra_num)
            Xte, yte, wte = _xy(te_trainish, extra_num)
            enet = _fit_enet(Xtr, ytr, wtr, num_cols, cat_cols, pe)
            gbm = _fit_lgb(Xtr, ytr, wtr, num_cols, cat_cols, pl)
            p_enet = enet.predict_proba(Xte)[:, 1]
            p_lgb = gbm.predict_proba(Xte)[:, 1]
        last_gbm, last_cols = gbm, extra_num
        te = te_trainish.copy()
        te["p_enet"] = p_enet
        te["p_lgb"] = p_lgb
        ev = te_eval.copy()
        if compute_aux:
            afe = ev["fever"].eq(0)
            if afe.any():
                Xafe, _, _ = _xy(ev.loc[afe], extra_num)
                ev.loc[afe, "p_enet"] = enet.predict_proba(Xafe)[:, 1]
                ev.loc[afe, "p_lgb"] = gbm.predict_proba(Xafe)[:, 1]
        pred_frames.append(te)
        for model, pcol in [("elasticnet", "p_enet"), ("lightgbm", "p_lgb")]:
            metric_rows.append(_metrics_block(te, pcol, model, held, extra={"params": json.dumps(pe if model == "elasticnet" else pl), "tag": tag}, n_boot=n_boot))
            if not compute_aux:
                continue
            bins = calibration_bins(te["rdt"], te[pcol], te["weight"])
            bins["country"] = held
            bins["model"] = model
            bins["tag"] = tag
            calib_rows.append(bins)
            dca = dca_curves(
                ev["rdt"].to_numpy(),
                ev["fever"].to_numpy(),
                ev[pcol].to_numpy() if pcol in ev.columns else np.full(len(ev), np.nan),
                ev["weight"].to_numpy(),
                ev["afebrile"].to_numpy() if "afebrile" in ev.columns else (ev["fever"] == 0).to_numpy(),
            )
            dca["country"] = held
            dca["model"] = model
            dca["tag"] = tag
            dca_rows.append(dca)
            for wealth, g in te.groupby("wealth"):
                if len(g) < 30:
                    continue
                rec = _metrics_block(g, pcol, model, held, extra={"subgroup": f"wealth_{int(wealth) if pd.notna(wealth) else 'na'}", "tag": tag}, n_boot=200)
                subgroup_rows.append(rec)
            for urb, g in te.groupby("urban"):
                if len(g) < 30:
                    continue
                rec = _metrics_block(g, pcol, model, held, extra={"subgroup": f"urban_{int(urb) if pd.notna(urb) else 'na'}", "tag": tag}, n_boot=200)
                subgroup_rows.append(rec)

    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    calib = pd.concat(calib_rows, ignore_index=True) if calib_rows else pd.DataFrame()
    dca = pd.concat(dca_rows, ignore_index=True) if dca_rows else pd.DataFrame()
    sub = pd.DataFrame(subgroup_rows)
    preds.to_parquet(DATA_DIR / f"predictions_{tag}.parquet", index=False)
    metrics.to_csv(TABLES / f"metrics_loco_{tag}.csv", index=False)
    calib.to_csv(TABLES / f"calibration_bins_{tag}.csv", index=False)
    dca.to_csv(TABLES / f"dca_{tag}.csv", index=False)
    sub.to_csv(TABLES / f"subgroup_metrics_{tag}.csv", index=False)
    if last_gbm is not None:
        import joblib

        joblib.dump(last_gbm, MODELS_DIR / f"lgb_lastfold_{tag}.joblib")
    print(metrics.to_string(index=False))
    return {"preds": preds, "metrics": metrics, "calib": calib, "dca": dca, "sub": sub, "gbm": last_gbm}


def run_senegal_temporal(analytic: pd.DataFrame | None = None) -> pd.DataFrame:
    if analytic is None:
        analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    sn = analytic[(analytic["country"] == "Senegal") & analytic["train_sample"]].copy()
    test_mask = sn["survey_year"].astype(str).str.contains("2020")
    tr = sn[~test_mask]
    te = sn[test_mask]
    if len(tr) < 100 or (te["rdt"] == 1).sum() < 5:
        print("Senegal temporal: insufficient sample; writing empty table")
        out = pd.DataFrame()
        out.to_csv(TABLES / "senegal_temporal.csv", index=False)
        return out
    print(f"Senegal temporal train N={len(tr)} test N={len(te)} RDT+={int((te.rdt==1).sum())}")
    p_enet, p_lgb, _, _, pe, pl = _predict_fold(tr, te)
    te = te.copy()
    te["p_enet"] = p_enet
    te["p_lgb"] = p_lgb
    rows = []
    for model, pcol in [("elasticnet", "p_enet"), ("lightgbm", "p_lgb")]:
        rows.append(_metrics_block(te, pcol, model, "Senegal-temporal"))
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "senegal_temporal.csv", index=False)
    te.to_parquet(DATA_DIR / "predictions_senegal_temporal.parquet", index=False)
    return out


if __name__ == "__main__":
    run_loco()
    run_senegal_temporal()
