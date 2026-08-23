"""Manuscript tables and 300 DPI figures (plan: 10).

Print layout: two-column journal (~90 mm per panel). Figures 1 and 2 are split
into two 2x2 files so labels stay 8-10 pt instead of an 8-tile grid.
"""

from __future__ import annotations

import shutil
from string import ascii_uppercase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.config import DATA_DIR, DPI, FIGURES, TABLES, ensure_dirs
from analysis.util import df_to_md, weighted_mean

# Ascending weighted RDT prevalence (Table 1).
PREVALENCE_ORDER = [
    "Mauritania",
    "Gambia",
    "Senegal",
    "Niger",
    "Mali",
    "Nigeria",
    "Burkina Faso",
]
DISPLAY_NAME = {"Gambia": "The Gambia"}

# ~90 mm column at print size; 2 panels = two-column page width.
PANEL_W_IN = 90 / 25.4
PANEL_H_IN = 78 / 25.4
GUTTER_IN = 0.22

CALIB_COLOR = "#7F2704"
IDENTITY_COLOR = "#111111"
DCA_NONE = "#4D4D4D"
DCA_ALL = "#111111"
DCA_FEVER = "#014636"
DCA_MODEL = "#7F2704"
WEALTH_COLORS = {
    1: "#08306B",
    2: "#08519C",
    3: "#2171B5",
    4: "#6A51A3",
    5: "#3F007D",
}
URBAN_COLORS = {0: "#084594", 1: "#99000D"}


def _print_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#111111",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
        }
    )


def _letter(ax, i: int) -> None:
    ax.text(
        0.02,
        0.98,
        ascii_uppercase[i],
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color="#000000",
        zorder=5,
    )


def _country_title(country: str, n: int | None = None, n_pos: int | None = None) -> str:
    name = DISPLAY_NAME.get(country, country)
    if n is None:
        return name
    if n_pos is None:
        return f"{name}\nn = {n:,}"
    return f"{name}\nn = {n:,}; RDT+ = {n_pos:,}"


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


def _afebrile_n(tab1: pd.DataFrame | None) -> dict[str, tuple[int, int]]:
    if tab1 is None or tab1.empty:
        tab1 = pd.read_csv(TABLES / "table1_characteristics.csv")
    return {r.country: (int(r.n), int(r.rdt_pos)) for r in tab1.itertuples()}


def _eval_n(analytic: pd.DataFrame | None) -> dict[str, tuple[int, int]]:
    if analytic is not None and "eval_sample" in analytic.columns:
        ev = analytic.loc[analytic["eval_sample"]]
        out = {}
        for country, g in ev.groupby("country"):
            out[str(country)] = (int(len(g)), int((g["rdt"] == 1).sum()))
        return out
    flow = pd.read_csv(TABLES / "table2_flow.csv")
    sub = flow.loc[flow["step"] == "3_fever_item_nonmissing"]
    g = sub.groupby("country", as_index=False).agg(n=("n", "sum"), rdt_pos=("rdt_pos", "sum"))
    return {r.country: (int(r.n), int(r.rdt_pos)) for r in g.itertuples()}


def _ordered_countries(available) -> list[str]:
    have = set(available)
    ordered = [c for c in PREVALENCE_ORDER if c in have]
    extra = sorted(have - set(ordered))
    return ordered + extra


def _save_country_halves(
    countries: list[str],
    plot_fn,
    stem: str,
    n_lookup: dict[str, tuple[int, int]],
    empty_fn=None,
) -> None:
    """Two 2x2 figures (A-D then E-G) so each panel prints at ~90 mm."""
    _print_style()
    chunks = [countries[:4], countries[4:]]
    letter_offset = 0
    for part, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue
        ncols = 2
        nrows = 2
        fig_w = ncols * PANEL_W_IN + GUTTER_IN
        fig_h = nrows * PANEL_H_IN + 0.35
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        for i in range(nrows * ncols):
            ax = axes[i // ncols][i % ncols]
            if i < len(chunk):
                country = chunk[i]
                plot_fn(ax, country)
                n, n_pos = n_lookup.get(country, (None, None))
                ax.set_title(_country_title(country, n, n_pos), pad=4)
                _letter(ax, letter_offset + i)
            elif empty_fn is not None:
                empty_fn(ax)
            else:
                ax.axis("off")
        fig.tight_layout(w_pad=1.1, h_pad=1.15)
        suffix = "a" if part == 1 else "b"
        path = FIGURES / f"{stem}_{suffix}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if part == 1:
            shutil.copyfile(path, FIGURES / f"{stem}.png")
        letter_offset += len(chunk)


def figure1_calibration(calib: pd.DataFrame, model: str = "lightgbm", tab1: pd.DataFrame | None = None) -> None:
    sub = calib[calib["model"] == model]
    countries = _ordered_countries(sub["country"].unique())
    n_lookup = _afebrile_n(tab1)

    def plot(ax, country):
        g = sub[sub["country"] == country]
        ax.plot([0, 1], [0, 1], ls="--", color=IDENTITY_COLOR, lw=2.0, zorder=1)
        ax.plot(
            g["pred"],
            g["obs"],
            "o-",
            color=CALIB_COLOR,
            lw=3.0,
            ms=8.5,
            markeredgecolor="#111111",
            markeredgewidth=0.7,
            zorder=2,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed RDT-positive fraction")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    def empty(ax):
        ax.axis("off")
        ax.plot([], [], ls="--", color=IDENTITY_COLOR, lw=2.0, label="Perfect calibration")
        ax.plot(
            [],
            [],
            "o-",
            color=CALIB_COLOR,
            lw=3.0,
            ms=8.5,
            markeredgecolor="#111111",
            label="Decile mean (survey-weighted)",
        )
        ax.legend(loc="center", frameon=False, fontsize=8)

    _save_country_halves(countries, plot, "figure1_calibration", n_lookup, empty_fn=empty)


def figure2_dca(
    dca: pd.DataFrame,
    model: str = "lightgbm",
    analytic: pd.DataFrame | None = None,
) -> None:
    sub = dca[dca["model"] == model]
    countries = _ordered_countries(sub["country"].unique())
    n_lookup = _eval_n(analytic)

    def plot(ax, country):
        g = sub[sub["country"] == country].sort_values("threshold")
        ax.axvspan(0.05, 0.40, color="#D9D9D9", alpha=0.18, zorder=0)
        ax.axhline(0, color="#333333", lw=1.2, zorder=1)
        ax.plot(g["threshold"], g["nb_none"], ":", color=DCA_NONE, lw=2.6, label="Test none", zorder=2)
        ax.plot(g["threshold"], g["nb_all"], "--", color=DCA_ALL, lw=2.6, label="Test all", zorder=2)
        ax.plot(g["threshold"], g["nb_fever"], "-", color=DCA_FEVER, lw=3.1, label="Fever rule", zorder=3)
        ax.plot(
            g["threshold"],
            g["nb_fever_plus_model"],
            "-",
            color=DCA_MODEL,
            lw=3.1,
            label="Fever + model",
            zorder=3,
        )
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Net benefit")
        ax.set_xlim(0.05, 0.40)

    def empty(ax):
        ax.axis("off")
        handles = [
            Line2D([0], [0], ls=":", color=DCA_NONE, lw=2.6, label="Test none"),
            Line2D([0], [0], ls="--", color=DCA_ALL, lw=2.6, label="Test all"),
            Line2D([0], [0], ls="-", color=DCA_FEVER, lw=3.1, label="Fever rule"),
            Line2D([0], [0], ls="-", color=DCA_MODEL, lw=3.1, label="Fever + model"),
        ]
        ax.legend(handles=handles, loc="center", frameon=False, fontsize=9, title="Strategy")

    _save_country_halves(countries, plot, "figure2_dca", n_lookup, empty_fn=empty)

    _print_style()
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    ax.plot([], [], ":", color=DCA_NONE, lw=2.2, label="Test none")
    ax.plot([], [], "--", color=DCA_ALL, lw=2.2, label="Test all")
    ax.plot([], [], "-", color=DCA_FEVER, lw=2.5, label="Fever rule")
    ax.plot([], [], "-", color=DCA_MODEL, lw=2.5, label="Fever + model")
    ax.legend(loc="center", frameon=False, fontsize=9)
    ax.axis("off")
    fig.savefig(FIGURES / "figure2_dca_legend.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure3_wealth_calibration(preds: pd.DataFrame) -> None:
    if preds.empty or "p_lgb" not in preds.columns:
        return
    from analysis.models_loco import calibration_bins

    _print_style()
    fig_w = 2 * PANEL_W_IN + GUTTER_IN
    fig_h = PANEL_H_IN + 0.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot([0, 1], [0, 1], ls="--", color=IDENTITY_COLOR, lw=2.0, zorder=1)
    n_by_q = {}
    for w, g in preds.groupby("wealth"):
        if pd.isna(w) or len(g) < 50:
            continue
        q = int(w)
        n_by_q[q] = (int(len(g)), int((g["rdt"] == 1).sum()))
        bins = calibration_bins(g["rdt"], g["p_lgb"], g["weight"])
        if bins.empty:
            continue
        label = f"Q{q}"
        ax.plot(
            bins["pred"],
            bins["obs"],
            "o-",
            color=WEALTH_COLORS.get(q, "#333333"),
            lw=2.8,
            ms=8.0,
            markeredgecolor="#111111",
            markeredgewidth=0.5,
            label=label,
            zorder=2,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed RDT-positive fraction")
    ax.legend(title="Wealth quintile (1 = poorest)", frameon=False, loc="upper left")
    ax.set_title("Pooled leave-one-country-out LightGBM predictions")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_wealth_calibration.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(
        [{"wealth_quintile": q, "n": n, "rdt_pos": p} for q, (n, p) in sorted(n_by_q.items())]
    ).to_csv(TABLES / "figure3_wealth_n.csv", index=False)


def figure4_subgroup_dca(analytic: pd.DataFrame, preds: pd.DataFrame) -> None:
    if preds.empty:
        return
    from analysis.models_loco import dca_curves

    _print_style()
    fig_w = 2 * PANEL_W_IN + GUTTER_IN
    fig_h = PANEL_H_IN + 0.25
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    merged = preds.copy()
    specs = (
        (axes[0], "wealth", "Wealth quintile", True),
        (axes[1], "urban", "Urban / rural residence", False),
    )
    for ax, col, title, is_wealth in specs:
        ax.axvspan(0.05, 0.40, color="#D9D9D9", alpha=0.18, zorder=0)
        ax.axhline(0, color="#333333", lw=1.2, zorder=1)
        for key, g in merged.groupby(col):
            if pd.isna(key) or (g["rdt"] == 1).sum() < 30:
                continue
            y = g["rdt"].to_numpy()
            w = g["weight"].to_numpy()
            p = g["p_lgb"].to_numpy()
            fever = np.zeros(len(g))
            dca = dca_curves(y, fever, p, w, np.ones(len(g), dtype=bool))
            k = int(key)
            n = len(g)
            n_pos = int((g["rdt"] == 1).sum())
            if is_wealth:
                color = WEALTH_COLORS.get(k, "#333333")
                label = f"Q{k} (n={n:,})"
            else:
                color = URBAN_COLORS.get(k, "#333333")
                label = ("Rural" if k == 0 else "Urban") + f" (n={n:,}; RDT+={n_pos:,})"
            ax.plot(dca["threshold"], dca["nb_fever_plus_model"], color=color, lw=3.0, label=label, zorder=2)
        ax.set_title(title)
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Net benefit (afebrile)")
        ax.set_xlim(0.05, 0.40)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    _letter(axes[0], 0)
    _letter(axes[1], 1)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIGURES / "figure4_subgroup_dca.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_results_markdown(tab1, metrics) -> None:
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
        "Calibration: `outputs/figures/figure1_calibration_a.png` and `figure1_calibration_b.png`.",
        "DCA: `outputs/figures/figure2_dca_a.png` and `figure2_dca_b.png`.",
        "Standalone captions: `reports/figure_captions.md`.",
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
        figure1_calibration(pd.read_csv(calib_path), tab1=tab1)
    if dca_path.exists():
        figure2_dca(pd.read_csv(dca_path), analytic=analytic)
    if len(preds):
        figure3_wealth_calibration(preds)
        figure4_subgroup_dca(analytic, preds)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    write_results_markdown(tab1, metrics)
    print("Wrote tables/figures")


if __name__ == "__main__":
    run_figures()
