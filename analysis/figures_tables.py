"""Manuscript tables and 300 DPI figures (plan: 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.config import DATA_DIR, DPI, FIGURES, TABLES, ensure_dirs
from analysis.util import df_to_md, weighted_mean


def table1_characteristics(analytic: pd.DataFrame) -> pd.DataFrame:
    df = analytic.loc[analytic["train_sample"]].copy()
    rows = []
    for country, g in df.groupby("country"):
        w = g["weight"].to_numpy()
        rec = {
            "country": country,
            "n": int(len(g)),
            "rdt_pos": int((g["rdt"] == 1).sum()),
            "age_mean": weighted_mean(g["age_months"].to_numpy(), w),
            "pct_female": 100 * weighted_mean(g["female"].fillna(0).to_numpy(), w),
            "pct_urban": 100 * weighted_mean(g["urban"].fillna(0).to_numpy(), w),
            "pct_net_used": 100 * weighted_mean(g["net_used"].fillna(0).to_numpy(), w),
            "wealth_mean": weighted_mean(g["wealth"].to_numpy(), w),
            "weighted_rdt_prev": 100 * weighted_mean(g["rdt"].to_numpy(), w),
        }
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / "table1_characteristics.csv", index=False)
    return tab


def _panel(countries, plot_fn, outfile, title):
    n = len(countries)
    if n == 0:
        return
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False)
    for i, country in enumerate(countries):
        ax = axes[i // ncols][i % ncols]
        plot_fn(ax, country)
        ax.set_title(country)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure1_calibration(calib: pd.DataFrame, model: str = "lightgbm") -> None:
    sub = calib[calib["model"] == model]
    countries = sorted(sub["country"].unique())

    def plot(ax, country):
        g = sub[sub["country"] == country]
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.plot(g["pred"], g["obs"], "o-", color="#d95f02")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")

    _panel(countries, plot, FIGURES / "figure1_calibration.png", "Calibration (LightGBM, afebrile held-out country)")


def figure2_dca(dca: pd.DataFrame, model: str = "lightgbm") -> None:
    sub = dca[dca["model"] == model]
    countries = sorted(sub["country"].unique())

    def plot(ax, country):
        g = sub[sub["country"] == country]
        ax.plot(g["threshold"], g["nb_none"], ":", color="grey", label="Test none")
        ax.plot(g["threshold"], g["nb_all"], "--", color="black", label="Test all")
        ax.plot(g["threshold"], g["nb_fever"], "-", color="#1b9e77", label="Fever rule")
        ax.plot(g["threshold"], g["nb_fever_plus_model"], "-", color="#d95f02", label="Fever + model")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Net benefit")
        ax.axhline(0, color="lightgrey", lw=0.8)

    _panel(countries, plot, FIGURES / "figure2_dca.png", "DCA on all 6–59m with RDT: fever rule vs fever+model")
    # legend once
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([], [], ":", color="grey", label="Test none")
    ax.plot([], [], "--", color="black", label="Test all")
    ax.plot([], [], "-", color="#1b9e77", label="Fever rule")
    ax.plot([], [], "-", color="#d95f02", label="Fever + model (afebrile)")
    ax.legend(loc="center")
    ax.axis("off")
    fig.savefig(FIGURES / "figure2_dca_legend.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figure3_wealth_calibration(preds: pd.DataFrame) -> None:
    if preds.empty or "p_lgb" not in preds.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for w, g in preds.groupby("wealth"):
        if pd.isna(w) or len(g) < 50:
            continue
        from analysis.models_loco import calibration_bins

        bins = calibration_bins(g["rdt"], g["p_lgb"], g["weight"])
        if bins.empty:
            continue
        ax.plot(bins["pred"], bins["obs"], "o-", label=f"Q{int(w)}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Subgroup calibration by wealth quintile (pooled LOCO predictions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_wealth_calibration.png", dpi=DPI)
    plt.close(fig)


def figure4_subgroup_dca(analytic: pd.DataFrame, preds: pd.DataFrame) -> None:
    if preds.empty:
        return
    from analysis.models_loco import dca_curves

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    merged = preds.copy()
    for ax, col, title in ((axes[0], "wealth", "Wealth"), (axes[1], "urban", "Urban/rural")):
        for key, g in merged.groupby(col):
            if pd.isna(key) or (g["rdt"] == 1).sum() < 30:
                continue
            # Within afebrile, fever rule = test-none; still plot model vs none for equity
            y = g["rdt"].to_numpy()
            w = g["weight"].to_numpy()
            p = g["p_lgb"].to_numpy()
            fever = np.zeros(len(g))
            dca = dca_curves(y, fever, p, w, np.ones(len(g), dtype=bool))
            label = f"{col}={int(key)}"
            ax.plot(dca["threshold"], dca["nb_fever_plus_model"], label=label)
        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Net benefit (afebrile)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_subgroup_dca.png", dpi=DPI)
    plt.close(fig)


def write_results_markdown(tab1, metrics) -> None:
    from analysis.util import df_to_md
    from analysis.config import REPORTS

    lines = [
        "# Results extracts",
        "",
        "## Table 1 (survey-weighted, afebrile training sample)",
        "",
        df_to_md(tab1),
        "",
        "## LOCO discrimination and calibration (primary LightGBM/elastic net)",
        "",
        df_to_md(metrics),
        "",
        "Calibration plots: `outputs/figures/figure1_calibration.png`.",
        "DCA (fever rule vs fever+model on all 6–59 months with RDT): `outputs/figures/figure2_dca.png`.",
        "SHAP is a statement about model behavior, not malaria biology.",
    ]
    REPORTS.joinpath("results_extracts.md").write_text("\n".join(lines), encoding="utf-8")


def run_figures() -> None:
    ensure_dirs()
    analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    tab1 = table1_characteristics(analytic)
    pred_path = DATA_DIR / "predictions_primary.parquet"
    preds = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    calib_path = TABLES / "calibration_bins_primary.csv"
    dca_path = TABLES / "dca_primary.csv"
    metrics_path = TABLES / "metrics_loco_primary.csv"
    if calib_path.exists():
        figure1_calibration(pd.read_csv(calib_path))
    if dca_path.exists():
        figure2_dca(pd.read_csv(dca_path))
    if len(preds):
        figure3_wealth_calibration(preds)
        figure4_subgroup_dca(analytic, preds)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    write_results_markdown(tab1, metrics)
    print("Wrote tables/figures")


if __name__ == "__main__":
    run_figures()
