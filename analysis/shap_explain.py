"""SHAP for the gradient boosting model (plan: 08). Model behavior, not causal biology."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from analysis.config import DATA_DIR, FIGURES, LGBM_GRID, SEED, TABLES, ensure_dirs
from analysis.models_loco import CAT_COLS, NUM_COLS, _fit_lgb, _xy

SHAP_LABELS = {
    "cat__lat_band_sahel": "Latitude band (Sahel)",
    "num__month_sin": "Survey month (sine)",
    "num__month_cos": "Survey month (cosine)",
    "num__urban": "Urban residence",
    "num__hhsize": "Household size",
    "num__age_months": "Child age (months)",
    "num__toilet_improved": "Improved sanitation",
    "num__educ": "Maternal education (years)",
    "num__wealth": "Wealth quintile",
    "cat__floor_grp_finished": "Floor: finished",
    "cat__floor_grp_natural": "Floor: natural",
    "cat__wall_grp_finished": "Wall: finished",
    "cat__roof_grp_finished": "Roof: finished",
    "cat__lat_band_guinean": "Latitude band (Guinean)",
    "cat__wall_grp_natural": "Wall: natural",
    "num__net_used": "Bed-net used last night",
    "num__net_own": "Bed-net ownership",
    "num__female": "Female child",
    "num__water_improved": "Improved water",
    "num__sahel_nigeria": "Nigeria Sahel band",
}


def _shap_label(name: str) -> str:
    if name in SHAP_LABELS:
        return SHAP_LABELS[name]
    cleaned = name.replace("cat__", "").replace("num__", "").replace("_", " ")
    return cleaned


def run_shap(analytic: pd.DataFrame | None = None, sample_n: int = 4000) -> None:
    ensure_dirs()
    if analytic is None:
        analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    train = analytic.loc[analytic["train_sample"]].copy()
    best_lgb = LGBM_GRID[1]
    num_cols = [c for c in NUM_COLS if c in train.columns]
    cat_cols = [c for c in CAT_COLS if c in train.columns]
    X, y, w = _xy(train)
    gbm: Pipeline = _fit_lgb(X, y, w, num_cols, cat_cols, best_lgb)
    Xt = gbm.named_steps["prep"].transform(X)
    feature_names = gbm.named_steps["prep"].get_feature_names_out()
    clf = gbm.named_steps["clf"]
    rng = np.random.default_rng(SEED)
    n = min(sample_n, Xt.shape[0])
    idx = rng.choice(Xt.shape[0], size=n, replace=False)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt[idx])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(TABLES / "shap_importance.csv", index=False)
    import matplotlib.pyplot as plt

    labels = [_shap_label(f) for f in feature_names]
    top = importance.head(15).copy()
    top["label"] = top["feature"].map(_shap_label)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(90 / 25.4 * 2 + 0.2, 110 / 25.4))
    ax.barh(
        top["label"][::-1],
        top["mean_abs_shap"][::-1],
        color="#08306B",
        edgecolor="#111111",
        linewidth=0.4,
        height=0.72,
    )
    ax.set_xlabel("Mean |SHAP|")
    ax.text(0.0, 1.02, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_shap_bar.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    plt.figure(figsize=(90 / 25.4 * 2 + 0.2, 110 / 25.4))
    shap.summary_plot(shap_values, Xt[idx], feature_names=labels, max_display=15, show=False)
    fig = plt.gcf()
    fig.text(0.01, 0.98, "B", fontsize=11, fontweight="bold", va="top")
    plt.tight_layout()
    plt.savefig(FIGURES / "figure5_shap_beeswarm.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(importance.head(15).to_string(index=False))
    print("SHAP describes how the fitted model uses features; it is not a causal statement about malaria biology.")


if __name__ == "__main__":
    run_shap()
