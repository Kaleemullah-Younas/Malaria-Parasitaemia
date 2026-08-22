"""SHAP for the gradient boosting model (plan: 08). Model behavior, not causal biology."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from analysis.config import DATA_DIR, FIGURES, LGBM_GRID, SEED, TABLES, ensure_dirs
from analysis.models_loco import CAT_COLS, NUM_COLS, _fit_lgb, _xy


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

    top = importance.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2c7fb8")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("LightGBM mean absolute SHAP (pooled training fit; model behavior)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_shap_bar.png", dpi=300)
    plt.close(fig)

    plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, Xt[idx], feature_names=feature_names, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "figure5_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(importance.head(15).to_string(index=False))
    print("SHAP describes how the fitted model uses features; it is not a causal statement about malaria biology.")


if __name__ == "__main__":
    run_shap()
