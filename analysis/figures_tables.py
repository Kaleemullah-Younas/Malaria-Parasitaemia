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

from matplotlib.patches import FancyBboxPatch

from analysis.config import DATA_DIR, DPI, FIGURES, PRIMARY_FEATURES, TABLES, ensure_dirs
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


def _weighted_pct(series: pd.Series, w: np.ndarray, value=1) -> float:
    s = pd.to_numeric(series, errors="coerce")
    mask = s.notna() & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return 100.0 * float(np.average((s[mask] == value).astype(float), weights=w[mask]))


def table1_expanded(analytic: pd.DataFrame) -> pd.DataFrame:
    """Every PRIMARY_FEATURES predictor plus percent missing, by country."""
    df = analytic.loc[analytic["train_sample"]].copy()
    rows = []
    binary = {
        "urban": "Urban residence (%)",
        "water_improved": "Improved water (%)",
        "toilet_improved": "Improved sanitation (%)",
        "net_own": "Household owns a bed-net (%)",
        "net_used": "Slept under a net last night (%)",
        "female": "Female child (%)",
        "sahel_nigeria": "Nigeria Sahel band (%)",
    }
    continuous = {
        "age_months": "Age (months)",
        "wealth": "Wealth quintile (mean, 1=poorest)",
        "educ": "Maternal education code (v106, 0–3)",
        "hhsize": "Household size",
        "month_sin": "Survey month sine",
        "month_cos": "Survey month cosine",
    }
    categorical = {
        "floor_grp": "Floor material",
        "wall_grp": "Wall material",
        "roof_grp": "Roof material",
        "lat_band": "Latitude band",
    }
    for country, g in df.groupby("country"):
        w = pd.to_numeric(g["weight"], errors="coerce").fillna(0).to_numpy()
        rec: dict = {"country": country, "n": int(len(g)), "rdt_pos": int((g["rdt"] == 1).sum())}
        rec["weighted_rdt_prev"] = 100 * weighted_mean(g["rdt"].to_numpy(), w)
        rec["rdt_miss_pct"] = 100 * float(g["rdt"].isna().mean())
        for col, label in continuous.items():
            if col not in g.columns:
                continue
            rec[col] = weighted_mean(pd.to_numeric(g[col], errors="coerce").to_numpy(), w)
            rec[f"{col}_miss_pct"] = 100 * float(pd.to_numeric(g[col], errors="coerce").isna().mean())
            rec[f"{col}_label"] = label
        for col, label in binary.items():
            if col not in g.columns:
                continue
            rec[col] = _weighted_pct(g[col], w, 1)
            rec[f"{col}_miss_pct"] = 100 * float(pd.to_numeric(g[col], errors="coerce").isna().mean())
            rec[f"{col}_label"] = label
        for col, label in categorical.items():
            if col not in g.columns:
                continue
            s = g[col].astype(str).fillna("missing")
            for level in sorted(s.unique()):
                m = s == level
                rec[f"{col}_{level}_pct"] = (
                    100 * float(w[m.to_numpy()].sum() / w.sum()) if w.sum() > 0 else 0.0
                )
            rec[f"{col}_miss_pct"] = rec.get(f"{col}_missing_pct", 100 * float(g[col].isna().mean()))
        for col in PRIMARY_FEATURES:
            if f"{col}_miss_pct" not in rec and col in g.columns:
                rec[f"{col}_miss_pct"] = 100 * float(pd.to_numeric(g[col], errors="coerce").isna().mean())
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / "table1_expanded.csv", index=False)
    long_rows = []
    for r in rows:
        country = r["country"]
        for col in PRIMARY_FEATURES:
            miss = r.get(f"{col}_miss_pct", float("nan"))
            val = r.get(col, float("nan"))
            long_rows.append(
                {
                    "country": country,
                    "predictor": col,
                    "estimate": val,
                    "missing_pct": miss,
                }
            )
    long = pd.DataFrame(long_rows)
    long.to_csv(TABLES / "table1_predictors_long.csv", index=False)
    return tab


def table_s3_subgroups() -> pd.DataFrame:
    """Supplementary Table 3: LightGBM wealth and urban/rural LOCO metrics."""
    path = TABLES / "subgroup_metrics_primary.csv"
    sub = pd.read_csv(path)
    keep = sub[sub["model"] == "lightgbm"].copy()
    keep.to_csv(TABLES / "table_s3_subgroup_metrics.csv", index=False)
    return keep


def figure_s1_workflow() -> None:
    """Study workflow (new supplementary figure; not a redo of Figure 1)."""
    ensure_dirs()
    _print_style()
    fig, ax = plt.subplots(figsize=(7.2, 9.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    boxes = [
        (7.2, "23 DHS/MIS waves\n7 Sahelian countries"),
        (6.0, "Merge KR child recode with HR\nhousehold RDT (hml35 / sh212)"),
        (4.8, "Eligibility: age 6–59 months;\nvalid RDT; h22 = 0; no recent\nantimalarial (138 excluded, 0.15%)"),
        (3.6, "Predictors obtainable without a\nblood test (PRIMARY_FEATURES)"),
        (2.4, "Inner leave-one-training-country-out\ntuning (LightGBM grid; elastic net fixed)"),
        (1.2, "Outer leave-one-country-out\nevaluation (held-out country unused)"),
        (0.15, "Calibration plots, decision curves,\nSHAP on pooled training LightGBM"),
    ]
    x0, width, height = 1.6, 6.8, 1.05
    for y, text in boxes:
        fancy = FancyBboxPatch(
            (x0, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.2,
            edgecolor="#111111",
            facecolor="#F7F7F7",
        )
        ax.add_patch(fancy)
        ax.text(x0 + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        y_top = boxes[i][0]
        y_bot = boxes[i + 1][0] + height
        ax.annotate(
            "",
            xy=(5, y_bot + 0.02),
            xytext=(5, y_top - 0.02),
            arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.4),
        )
    ax.set_title("Figure S1. Model-development and evaluation workflow", fontsize=11, pad=8)
    fig.tight_layout()
    path = FIGURES / "figure_s1_workflow.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def write_figure_captions() -> None:
    from analysis.config import REPORTS

    text = """# Figure captions (revision)

**Figure 1.** Calibration of the primary LightGBM model in leave-one-country-out validation among afebrile children aged 6–59 months with a valid RDT. Each panel is one held-out country, ordered by ascending survey-weighted RDT prevalence. The x-axis is the mean predicted probability in each predicted-risk decile; the y-axis is the survey-weighted observed RDT-positive fraction in that decile. The dashed black line is perfect calibration. Brown points and lines are decile means. O/E > 1 (points above the diagonal) is under-prediction; O/E < 1 is over-prediction. Sample sizes (n, RDT+) are shown in each panel title. Panels A–D: Mauritania (n = 7,063; RDT+ = 54), The Gambia (n = 5,001; RDT+ = 46), Senegal (n = 30,074; RDT+ = 573), Niger (n = 2,737; RDT+ = 410). Panels E–G: Mali (n = 17,033; RDT+ = 3,922), Nigeria (n = 15,979; RDT+ = 5,195), Burkina Faso (n = 15,309; RDT+ = 6,613). Files: `figure1_calibration_a.png`, `figure1_calibration_b.png`. Do not upload the unsplit `figure1_calibration.png`.

**Figure 2.** Decision-curve analysis for the primary LightGBM model on all children aged 6–59 months with a valid RDT and known fever status (not the afebrile subset alone). Each panel is one held-out country, same order as Figure 1. The x-axis is the decision threshold (probability of RDT positivity at which one would choose to test). The y-axis is net benefit. Strategies: dotted grey, test none; dashed black, test all; solid dark green, fever rule (test every child with a two-week fever history); solid brown, fever-plus-model (fever rule, plus testing an afebrile child if predicted probability exceeds the threshold). The light grey band is the pre-specified 5–40% threshold range. Sample sizes in each title are the full 6–59 month RDT sample used for the curves. Panels A–D: Mauritania (n = 8,632; RDT+ = 71), The Gambia (n = 5,949; RDT+ = 57), Senegal (n = 37,148; RDT+ = 876), Niger (n = 4,141; RDT+ = 942). Panels E–G: Mali (n = 22,274; RDT+ = 5,783), Nigeria (n = 24,533; RDT+ = 9,464), Burkina Faso (n = 21,478; RDT+ = 10,268). Files: `figure2_dca_a.png`, `figure2_dca_b.png`. Do not upload the unsplit `figure2_dca.png`.

**Figure 3.** Calibration by DHS wealth quintile using pooled leave-one-country-out LightGBM predictions among afebrile children. The x-axis is mean predicted probability; the y-axis is the survey-weighted observed RDT-positive fraction. The dashed black line is perfect calibration. Quintile 1 is poorest; quintile 5 is richest. Pooled sample: Q1 n = 22,941 (RDT+ = 4,581); Q2 n = 20,890 (4,381); Q3 n = 19,780 (3,968); Q4 n = 16,404 (2,900); Q5 n = 13,181 (983). File: `figure3_wealth_calibration.png`.

**Figure 4.** Decision curves by wealth quintile (A) and urban/rural residence (B) for the primary LightGBM model within the afebrile sample. Because all children in this sample are afebrile, the fever rule is equivalent to test-none; curves show fever-plus-model net benefit. The x-axis is the decision threshold; the y-axis is net benefit. The light grey band is the pre-specified 5–40% range. File: `figure4_subgroup_dca.png`.

**Figure 5.** SHAP attribution for a **pooled training** LightGBM model fitted on all training-sample children with the fixed hyperparameter setting `LGBM_GRID[1]` (`num_leaves=31`, `learning_rate=0.05`, `n_estimators=300`, `max_depth=5`). This is **not** one of the leave-one-country-out models reported in Table 2. TreeExplainer was applied to a random sample of 4,000 pooled training observations (seed 42). Rankings describe how this fitted model used the supplied features; they are not causal effects and they are not attributions from the validated LOCO models. (A) Mean absolute SHAP, highest at the top. (B) Beeswarm: each point is one child; colour is feature value (red = high, blue = low); horizontal position is the contribution to the log-odds of RDT positivity. Files: `figure5_shap_bar.png` (A), `figure5_shap_beeswarm.png` (B).

**Figure S1.** Workflow from survey selection through KR+HR merge, eligibility (including 138 antimalarial exclusions), predictors, inner leave-one-training-country-out tuning, outer leave-one-country-out evaluation, and calibration / decision-curve / SHAP outputs. File: `figure_s1_workflow.png`.
"""
    REPORTS.joinpath("figure_captions.md").write_text(text, encoding="utf-8")
    pack = FIGURES / "submission"
    pack.mkdir(parents=True, exist_ok=True)
    upload = [
        "figure1_calibration_a.png",
        "figure1_calibration_b.png",
        "figure2_dca_a.png",
        "figure2_dca_b.png",
        "figure3_wealth_calibration.png",
        "figure4_subgroup_dca.png",
        "figure5_shap_bar.png",
        "figure5_shap_beeswarm.png",
        "figure_s1_workflow.png",
    ]
    for name in upload:
        src = FIGURES / name
        if src.exists():
            shutil.copyfile(src, pack / name)
    FIGURES.joinpath("FIGURES_USED.md").write_text(
        "\n".join(
            [
                "# Figures to upload (no unsplit duplicates)",
                "",
                "Upload only these files:",
                "",
                "- `figure1_calibration_a.png`",
                "- `figure1_calibration_b.png`",
                "- `figure2_dca_a.png`",
                "- `figure2_dca_b.png`",
                "- `figure3_wealth_calibration.png`",
                "- `figure4_subgroup_dca.png`",
                "- `figure5_shap_bar.png`",
                "- `figure5_shap_beeswarm.png`",
                "- `figure_s1_workflow.png`",
                "",
                "Do **not** upload `figure1_calibration.png`, `figure2_dca.png`, or `figure2_dca_legend.png`.",
                "",
                "Captions: `reports/figure_captions.md`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


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


def run_figures(regenerate_primary: bool = False) -> None:
    """Write Table 1, Table S3, Figure S1, captions. Primary Figures 1–5 stay locked unless asked."""
    ensure_dirs()
    analytic = pd.read_parquet(DATA_DIR / "analytic.parquet")
    tab1 = table1_characteristics(analytic)
    table1_expanded(analytic)
    table_s3_subgroups()
    figure_s1_workflow()
    write_figure_captions()
    pred_path = DATA_DIR / "predictions_primary.parquet"
    preds = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    calib_path = TABLES / "calibration_bins_primary.csv"
    dca_path = TABLES / "dca_primary.csv"
    metrics_path = TABLES / "metrics_loco_primary.csv"
    if regenerate_primary:
        if calib_path.exists():
            figure1_calibration(pd.read_csv(calib_path), tab1=tab1)
        if dca_path.exists():
            figure2_dca(pd.read_csv(dca_path), analytic=analytic)
        if len(preds):
            figure3_wealth_calibration(preds)
            figure4_subgroup_dca(analytic, preds)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    write_results_markdown(tab1, metrics)
    print("Wrote tables/figures (primary Figures 1-5 locked)" if not regenerate_primary else "Wrote tables/figures")


if __name__ == "__main__":
    run_figures()
