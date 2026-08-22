"""Step 2 study population filters and Table 2 (plan: 03_study_population.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.config import DATA_DIR, RDT_POSITIVE_FLAG_N, REPORTS, TABLES, ensure_dirs
from analysis.load_merge import run_merge
from analysis.util import df_to_md


def _age_months(df: pd.DataFrame) -> pd.Series:
    if "b19" in df.columns:
        age = pd.to_numeric(df["b19"], errors="coerce")
    else:
        age = pd.Series(np.nan, index=df.index)
    if "hw1" in df.columns:
        hw1 = pd.to_numeric(df["hw1"], errors="coerce")
        age = age.fillna(hw1)
    if "hml16" in df.columns:
        age = age.fillna(pd.to_numeric(df["hml16"], errors="coerce"))
    return age


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["age_months"] = _age_months(work)
    work["weight"] = pd.to_numeric(work["v005"], errors="coerce") / 1_000_000 if "v005" in work.columns else np.nan
    if "hv005" in work.columns:
        alt = pd.to_numeric(work["hv005"], errors="coerce") / 1_000_000
        work["weight"] = work["weight"].fillna(alt) if hasattr(work["weight"], "fillna") else alt
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work["h22"] = pd.to_numeric(work.get("h22"), errors="coerce") if "h22" in work.columns else np.nan
    work["rdt"] = pd.to_numeric(work.get("rdt"), errors="coerce")
    work["fever"] = np.where(work["h22"] == 1, 1, np.where(work["h22"] == 0, 0, np.nan))
    work["afebrile"] = work["fever"] == 0
    work["has_antimalarial_item"] = work["antimalarial_any"].notna() if "antimalarial_any" in work.columns else False

    flow_rows = []
    keys = ["country", "survey_year", "recode"]

    def add_step(name: str, mask: pd.Series) -> None:
        sub = work.loc[mask]
        g = (
            sub.groupby(keys, dropna=False)
            .agg(n=("age_months", "size"), rdt_pos=("rdt", lambda s: int((s == 1).sum())))
            .reset_index()
        )
        g["step"] = name
        flow_rows.append(g)

    m0 = pd.Series(True, index=work.index)
    add_step("0_kr_rows", m0)
    m1 = m0 & work["age_months"].between(6, 59)
    add_step("1_age_6_59", m1)
    m2 = m1 & work["rdt"].notna()
    add_step("2_valid_rdt", m2)
    m3 = m2 & work["h22"].notna()
    add_step("3_fever_item_nonmissing", m3)
    m4 = m3 & (work["fever"] == 0)
    add_step("4_afebrile_h22eq0", m4)
    # Antimalarial exclusion only where the item was asked (typically febrile children).
    # Among afebrile this is structurally missing; do not drop them for missing treatment.
    # Antimalarial items are asked after fever. Missing must not drop afebrile children
    # (pandas NA is not False: `~(NA==1)` is NA and would exclude the row).
    no_recent_am = ~work["antimalarial_any"].eq(1).fillna(False) if "antimalarial_any" in work.columns else True
    m5 = m4 & no_recent_am
    add_step("5_afebrile_no_recent_antimalarial", m5)

    flow = pd.concat(flow_rows, ignore_index=True)

    work["in_age"] = work["age_months"].between(6, 59)
    work["valid_rdt"] = work["rdt"].isin([0, 1])
    work["eval_sample"] = work["in_age"] & work["valid_rdt"] & work["fever"].isin([0, 1])
    work["train_sample"] = work["eval_sample"] & work["afebrile"] & no_recent_am

    t2_rows = []
    train_df = work.loc[work["train_sample"]]
    for (country, year, recode), g in train_df.groupby(keys, dropna=False):
        w = g["weight"].fillna(0) + 1e-12
        pos = int((g["rdt"] == 1).sum())
        t2_rows.append(
            {
                "country": country,
                "survey_year": year,
                "recode": recode,
                "n_afebrile": int(len(g)),
                "rdt_positive": pos,
                "rdt_negative": int((g["rdt"] == 0).sum()),
                "weighted_prev": float(np.average(g["rdt"], weights=w)),
                "unweighted_prev": float(g["rdt"].mean()),
                "flag_rdtpos_lt30": int(pos < RDT_POSITIVE_FLAG_N),
            }
        )
    table2 = pd.DataFrame(t2_rows)
    country_rows = []
    for country, g in train_df.groupby("country"):
        pos = int((g["rdt"] == 1).sum())
        country_rows.append(
            {
                "country": country,
                "n_afebrile": int(len(g)),
                "rdt_positive": pos,
                "flag_rdtpos_lt30": int(pos < RDT_POSITIVE_FLAG_N),
            }
        )
    country = pd.DataFrame(country_rows)
    return work, flow, table2.assign(_country_flag=table2["country"].map(dict(zip(country.country, country.flag_rdtpos_lt30))))


def write_phase1_report(work: pd.DataFrame, flow: pd.DataFrame, table2: pd.DataFrame, merge_fail: pd.DataFrame) -> None:
    lines = []
    lines.append("# Phase 1 dataset lock — Step 2 study population")
    lines.append("")
    lines.append("Date: 2026-08-22")
    lines.append("")
    lines.append("This report is the hard gate before modeling. The endpoint is malaria RDT positivity among children aged 6–59 months with no reported fever in the preceding two weeks (`h22==0`). Care-seeking items (`h32y`) are asked only after fever and are not independently informative in the afebrile subset.")
    lines.append("")
    lines.append("## Merge failures")
    lines.append("")
    if len(merge_fail):
        cols = [c for c in merge_fail.columns if c in {"country", "survey_year", "recode", "n_kr", "n_hr_person", "kr_no_rdt_merge", "rdt_no_kr", "kr_no_gps", "rdt_source", "rdt_present", "skip_reason"}]
        lines.append(df_to_md(merge_fail[cols]))
    lines.append("")
    lines.append("## Filter flow (N at each step)")
    lines.append("")
    pivot = flow.pivot_table(index=["country", "survey_year", "recode"], columns="step", values="n", aggfunc="sum")
    lines.append(df_to_md(pivot.reset_index()))
    lines.append("")
    lines.append("## Table 2 — afebrile subgroup with valid RDT")
    lines.append("")
    show = table2.drop(columns=["_country_flag"], errors="ignore")
    lines.append(df_to_md(show))
    lines.append("")
    flagged = table2[table2["flag_rdtpos_lt30"] == 1]
    lines.append("## Flags: country-year afebrile RDT+ < 30")
    lines.append("")
    if len(flagged):
        lines.append(df_to_md(flagged[["country", "survey_year", "recode", "n_afebrile", "rdt_positive"]]))
        lines.append("")
        lines.append("These survey-years are too small for stable calibration plots if analysed alone. They remain in the pooled training set unless a later decision excludes them from LOCO test folds.")
    else:
        lines.append("None.")
    country = (
        work.loc[work["train_sample"]]
        .groupby("country")
        .agg(n=("rdt", "size"), rdt_pos=("rdt", lambda s: int((s == 1).sum())))
        .reset_index()
    )
    lines.append("")
    lines.append("## Pooled by country (LOCO test-fold size)")
    lines.append("")
    country["flag_lt30"] = country["rdt_pos"] < 30
    lines.append(df_to_md(country))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- RDT lives in HR person slots (`hml35_nn`); KR has no `hml35`. Merge key: `v001+v002+b16` = `hv001+hv002+hvidx`.")
    lines.append("- Senegal 2008–09 uses `sh212` (1=positive, 2=negative) if `hml35` is absent.")
    lines.append("- Mali 2010 has no KR file and is excluded from the afebrile endpoint.")
    lines.append("- One-week fever recall is not collected; `h22` is a two-week window.")
    lines.append("- Recent antimalarial items are typically asked only of febrile children, so HRP2 persistence after treatment cannot be fully excluded in the afebrile group.")
    lines.append("- No rows were dropped silently; unmatched KR/RDT/GPS counts are in `outputs/tables/merge_failures.csv`.")
    REPORTS.joinpath("phase1_dataset_lock.md").write_text("\n".join(lines), encoding="utf-8")


def run_study_population(merged: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if merged is None:
        path = DATA_DIR / "merged_raw.parquet"
        merged = pd.read_parquet(path) if path.exists() else run_merge()
    work, flow, table2 = apply_filters(merged)
    flow.to_csv(TABLES / "table2_flow.csv", index=False)
    table2.drop(columns=["_country_flag"], errors="ignore").to_csv(TABLES / "table2_afebrile.csv", index=False)
    fail_path = TABLES / "merge_failures.csv"
    merge_fail = pd.read_csv(fail_path) if fail_path.exists() else pd.DataFrame()
    write_phase1_report(work, flow, table2, merge_fail)
    out = DATA_DIR / "study_population.parquet"
    keep_cols = [c for c in work.columns if not c.endswith("_p") or c in work.columns[:5]]
    # store a slimmer frame for modeling
    slim_keep = [
        "country",
        "survey_year",
        "recode",
        "caseid",
        "v001",
        "v002",
        "b16",
        "v005",
        "weight",
        "v006",
        "v021",
        "v022",
        "v023",
        "v024",
        "v025",
        "v106",
        "v113",
        "v116",
        "v136",
        "v190",
        "b4",
        "age_months",
        "h22",
        "fever",
        "afebrile",
        "h32y",
        "h32z",
        "rdt",
        "microscopy",
        "hw53",
        "hml1",
        "hml12",
        "hv213",
        "hv214",
        "hv215",
        "hv253",
        "hv009",
        "gps_lat",
        "gps_lon",
        "lat_band",
        "sahel_nigeria",
        "antimalarial_any",
        "in_age",
        "valid_rdt",
        "eval_sample",
        "train_sample",
    ]
    slim = work[[c for c in slim_keep if c in work.columns]].copy()
    slim.to_parquet(out, index=False)
    print(f"Train (afebrile) N={int(work['train_sample'].sum()):,}")
    print(f"Eval (6-59 valid RDT, known fever) N={int(work['eval_sample'].sum()):,}")
    print(table2.drop(columns=["_country_flag"], errors="ignore").to_string(index=False))
    print(f"Wrote {REPORTS / 'phase1_dataset_lock.md'}")
    return slim


if __name__ == "__main__":
    run_study_population()
