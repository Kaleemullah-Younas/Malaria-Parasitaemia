"""Load KR+HR+GPS, reshape malaria person slots, flag merge failures (plan: 02_load_merge.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import DATA_DIR, TABLES, ensure_dirs
from analysis.inventory import run_inventory
from analysis.io_dhs import (
    antimalarial_any,
    list_dta_columns,
    recode_rdt,
    read_dta,
    read_gps_dbf,
    reshape_person,
    wanted_hr_columns,
    wanted_kr_columns,
)


def _lat_band(lat: pd.Series) -> pd.Series:
    x = pd.to_numeric(lat, errors="coerce")
    band = pd.Series(pd.NA, index=x.index, dtype="object")
    band = band.mask(x < 10, "guinean")
    band = band.mask((x >= 10) & (x < 14), "sudan")
    band = band.mask(x >= 14, "sahel")
    return band


def load_one(row: pd.Series) -> tuple[pd.DataFrame, dict]:
    stats = {
        "country": row.country,
        "survey_year": row.survey_year,
        "recode": row.recode,
        "n_kr": 0,
        "n_hr_hh": 0,
        "n_hr_person": 0,
        "n_gps": 0,
        "kr_no_line": 0,
        "kr_no_rdt_merge": 0,
        "rdt_no_kr": 0,
        "kr_no_gps": 0,
        "rdt_source": "none",
        "rdt_present": False,
    }
    if not row.has_kr or not row.has_hr or row.exclude_no_kr:
        stats["skip_reason"] = "no_kr_or_no_hr"
        return pd.DataFrame(), stats

    kr_path = Path(row.kr_path)
    hr_path = Path(row.hr_path)
    kr_avail = [c.lower() for c in list_dta_columns(kr_path)]
    hr_avail = [c.lower() for c in list_dta_columns(hr_path)]
    kr = read_dta(kr_path, wanted=wanted_kr_columns(kr_avail))
    hr = read_dta(hr_path, wanted=wanted_hr_columns(hr_avail))
    stats["n_kr"] = int(len(kr))
    stats["n_hr_hh"] = int(len(hr))

    hh_cols = [c for c in hr.columns if not any(c.startswith(p + "_") for p in ("hml35", "hml32", "hml12", "hml16", "sh212", "sh214r"))]
    hr_hh = hr[hh_cols].copy()
    person = reshape_person(hr)
    stats["n_hr_person"] = int(len(person))

    scheme = "standard"
    rdt_col = None
    if "hml35" in person.columns and person["hml35"].notna().any():
        rdt_col = "hml35"
        scheme = "standard"
    elif "sh212" in person.columns and person["sh212"].notna().any():
        rdt_col = "sh212"
        scheme = "sh212"
    stats["rdt_source"] = rdt_col or "none"
    stats["rdt_present"] = rdt_col is not None
    if rdt_col:
        person["rdt"] = recode_rdt(person[rdt_col], scheme)
        person["rdt_raw"] = pd.to_numeric(person[rdt_col], errors="coerce")
    else:
        person["rdt"] = pd.NA
        person["rdt_raw"] = pd.NA
    if "hml32" in person.columns:
        person["microscopy"] = recode_rdt(person["hml32"], "standard")
    else:
        person["microscopy"] = pd.NA
    if "hml12" in person.columns:
        person["hml12"] = pd.to_numeric(person["hml12"], errors="coerce")

    for col in ("v001", "v002", "b16", "hv001", "hv002"):
        if col in kr.columns:
            kr[col] = pd.to_numeric(kr[col], errors="coerce")
        if col in hr_hh.columns:
            hr_hh[col] = pd.to_numeric(hr_hh[col], errors="coerce")
        if col in person.columns:
            person[col] = pd.to_numeric(person[col], errors="coerce")

    stats["kr_no_line"] = int(kr["b16"].isna().sum()) if "b16" in kr.columns else int(len(kr))

    merged = kr.merge(
        hr_hh,
        left_on=["v001", "v002"],
        right_on=["hv001", "hv002"],
        how="left",
        suffixes=("", "_hr"),
    )
    if "hvidx" not in person.columns:
        person["hvidx"] = pd.NA
    merged = merged.merge(
        person,
        left_on=["v001", "v002", "b16"],
        right_on=["hv001", "hv002", "hvidx"],
        how="left",
        suffixes=("", "_p"),
    )
    stats["kr_no_rdt_merge"] = int(merged["rdt"].isna().sum()) if "rdt" in merged.columns else int(len(merged))

    tested = person.loc[person["rdt"].notna(), ["hv001", "hv002", "hvidx"]].copy()
    if len(tested) and "b16" in kr.columns:
        kr_keys = kr[["v001", "v002", "b16"]].dropna()
        t2 = tested.merge(
            kr_keys,
            left_on=["hv001", "hv002", "hvidx"],
            right_on=["v001", "v002", "b16"],
            how="left",
        )
        stats["rdt_no_kr"] = int(t2["b16"].isna().sum())
    else:
        stats["rdt_no_kr"] = int(len(tested))

    if row.gps_path:
        gps = read_gps_dbf(Path(row.gps_path))
        stats["n_gps"] = int(len(gps))
        if len(gps):
            merged = merged.merge(gps, on="hv001", how="left")
            stats["kr_no_gps"] = int(merged["gps_lat"].isna().sum()) if "gps_lat" in merged.columns else int(len(merged))
        else:
            merged["gps_lat"] = pd.NA
            merged["gps_lon"] = pd.NA
            stats["kr_no_gps"] = int(len(merged))
    else:
        merged["gps_lat"] = pd.NA
        merged["gps_lon"] = pd.NA
        stats["kr_no_gps"] = int(len(merged))

    merged["country"] = row.country
    merged["survey_year"] = row.survey_year
    merged["recode"] = row.recode
    merged["modal_year"] = row.modal_year if "modal_year" in row.index else pd.NA
    merged["antimalarial_any"] = antimalarial_any(merged)
    if "gps_lat" in merged.columns:
        merged["lat_band"] = _lat_band(merged["gps_lat"])
        merged["sahel_nigeria"] = np.where(
            (merged["country"] == "Nigeria") & (pd.to_numeric(merged["gps_lat"], errors="coerce") >= 11),
            1,
            np.where(merged["country"] == "Nigeria", 0, pd.NA),
        )
    else:
        merged["lat_band"] = pd.NA
        merged["sahel_nigeria"] = pd.NA
    return merged, stats


def run_merge() -> pd.DataFrame:
    ensure_dirs()
    _, _, years = run_inventory()
    keep = years[years["keep"]].copy()
    frames = []
    fail_rows = []
    for _, row in keep.iterrows():
        print(f"Loading {row.country} {row.survey_year} {row.recode} ...")
        df, stats = load_one(row)
        fail_rows.append(stats)
        if len(df):
            frames.append(df)
            print(f"  KR={stats['n_kr']:,} person={stats['n_hr_person']:,} rdt_source={stats['rdt_source']}")
        else:
            print(f"  skipped ({stats.get('skip_reason', '')})")
    fail = pd.DataFrame(fail_rows)
    fail.to_csv(TABLES / "merge_failures.csv", index=False)
    if not frames:
        raise RuntimeError("No surveys merged")
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    out = DATA_DIR / "merged_raw.parquet"
    pooled.to_parquet(out, index=False)
    print(f"Pooled N={len(pooled):,} -> {out}")
    return pooled


if __name__ == "__main__":
    run_merge()
