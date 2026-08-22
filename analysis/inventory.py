"""Survey manifest and per-country-year variable dictionary (plan: 01_inventory.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.config import COUNTRY_DIR_TO_NAME, DHS_ROOT, EXCLUDE_NO_KR, TABLES, ensure_dirs
from analysis.io_dhs import (
    dta_labels_na,
    list_dta_columns,
    map_lower,
    parse_recode_stem,
    read_dta,
)


INTEREST_VARS = {
    "rdt": ["hml35", "sh212"],
    "microscopy": ["hml32"],
    "fever": ["h22"],
    "care_seeking": ["h32y", "h32z"],
    "age": ["b19", "hw1"],
    "child_line": ["b16"],
    "weight": ["v005", "hv005"],
    "psu": ["v001", "hv001"],
    "stratum": ["v022", "v023"],
    "wealth": ["v190"],
    "residence": ["v025"],
    "region": ["v024"],
    "education": ["v106"],
    "hhsize": ["v136", "hv009"],
    "water": ["v113"],
    "toilet": ["v116"],
    "nets_hh": ["hml1"],
    "net_use": ["hml12"],
    "irs": ["hv253"],
    "floor": ["hv213"],
    "wall": ["hv214"],
    "roof": ["hv215"],
    "sex": ["b4"],
    "month": ["v006", "hv006"],
    "haemoglobin": ["hw53"],
}


def _find_files(folder: Path, kind: str) -> list[Path]:
    if not folder.is_dir():
        return []
    hits = []
    for p in folder.iterdir():
        if p.suffix.upper() != ".DTA" and not (kind == "GPS" and p.suffix.lower() == ".dbf"):
            continue
        if kind == "GPS":
            if p.suffix.lower() == ".dbf" and "GE" in p.stem.upper():
                hits.append(p)
        else:
            parsed = parse_recode_stem(p)
            if parsed and parsed[1] == ("KR" if kind == "KR" else "HR"):
                hits.append(p)
    return sorted(hits)


def discover_surveys() -> pd.DataFrame:
    rows = []
    for country_dir, country in COUNTRY_DIR_TO_NAME.items():
        cpath = DHS_ROOT / country_dir
        if not cpath.is_dir():
            continue
        for year_dir in sorted(p for p in cpath.iterdir() if p.is_dir()):
            year = year_dir.name
            kr_files = _find_files(year_dir / "KR", "KR")
            hr_files = _find_files(year_dir / "HR", "HR")
            gps_files = _find_files(year_dir / "GPS", "GPS")
            hr_by_recode = {}
            for hr in hr_files:
                parsed = parse_recode_stem(hr)
                if parsed:
                    hr_by_recode[parsed[2]] = hr
            gps_by_recode = {}
            for gps in gps_files:
                parsed = parse_recode_stem(gps)
                if parsed:
                    gps_by_recode[parsed[2]] = gps
            recodes = sorted({parse_recode_stem(p)[2] for p in kr_files if parse_recode_stem(p)})
            if not recodes and hr_files:
                recodes = sorted({parse_recode_stem(p)[2] for p in hr_files if parse_recode_stem(p)})
            if not recodes:
                recodes = ["MISSING"]
            for recode in recodes:
                kr = next((p for p in kr_files if parse_recode_stem(p) and parse_recode_stem(p)[2] == recode), None)
                hr = hr_by_recode.get(recode)
                gps = gps_by_recode.get(recode)
                if gps is None and len(gps_files) == 1:
                    gps = gps_files[0]
                rows.append(
                    {
                        "country": country,
                        "survey_year": year,
                        "recode": recode,
                        "kr_path": str(kr) if kr else "",
                        "hr_path": str(hr) if hr else "",
                        "gps_path": str(gps) if gps else "",
                        "has_kr": kr is not None,
                        "has_hr": hr is not None,
                        "has_gps": gps is not None,
                        "exclude_no_kr": (country, year) in EXCLUDE_NO_KR,
                    }
                )
    return pd.DataFrame(rows)


def _var_status(cols_lower: set[str], na_flags: dict[str, bool], names: list[str]) -> str:
    present = [n for n in names if n in cols_lower]
    if not present:
        return "absent"
    collected = [n for n in present if not na_flags.get(n, False)]
    if not collected:
        return "na_dictionary:" + ",".join(present)
    return "present:" + ",".join(collected)


def build_variable_dictionary(manifest: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for row in manifest.itertuples(index=False):
        kr_cols: set[str] = set()
        hr_cols: set[str] = set()
        na_flags: dict[str, bool] = {}
        if row.kr_path:
            knames = list_dta_columns(Path(row.kr_path))
            kr_cols = set(map_lower(knames))
            na_flags.update(dta_labels_na(Path(row.kr_path)))
        if row.hr_path:
            hnames = list_dta_columns(Path(row.hr_path))
            hr_cols = set(map_lower(hnames))
            na_flags.update(dta_labels_na(Path(row.hr_path)))
        all_cols = kr_cols | hr_cols
        rec = {
            "country": row.country,
            "survey_year": row.survey_year,
            "recode": row.recode,
            "n_kr_cols": len(kr_cols),
            "n_hr_cols": len(hr_cols),
        }
        for key, names in INTEREST_VARS.items():
            rec[key] = _var_status(all_cols, na_flags, names)
        rdt_hr_slots = sum(1 for c in hr_cols if c.startswith("hml35_"))
        sh212_slots = sum(1 for c in hr_cols if c.startswith("sh212_"))
        rec["hml35_slots"] = rdt_hr_slots
        rec["sh212_slots"] = sh212_slots
        rec["rdt_source"] = (
            "hml35_hr"
            if rdt_hr_slots
            else ("sh212_hr" if sh212_slots else ("hml35_kr" if "hml35" in kr_cols else "none"))
        )
        recs.append(rec)
    return pd.DataFrame(recs)


def sample_interview_years(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        path = row.kr_path or row.hr_path
        if not path:
            continue
        p = Path(path)
        names = list_dta_columns(p)
        lower = map_lower(names)
        year_col = "v007" if "v007" in lower else ("hv007" if "hv007" in lower else None)
        if year_col is None:
            rows.append({**row._asdict(), "modal_year": pd.NA, "year_min": pd.NA, "year_max": pd.NA})
            continue
        df = read_dta(p, wanted=[year_col])
        years = pd.to_numeric(df[year_col], errors="coerce").dropna()
        # DHS sometimes stores 2-digit years
        years = years.where(years > 100, years + 1900)
        years = years.where(years > 1980, years + 100)
        if years.empty:
            modal = pd.NA
            ymin = ymax = pd.NA
        else:
            modal = int(years.mode().iloc[0])
            ymin, ymax = int(years.min()), int(years.max())
        d = row._asdict() if hasattr(row, "_asdict") else dict(row._asdict())
        d.update({"modal_year": modal, "year_min": ymin, "year_max": ymax})
        rows.append(d)
    return pd.DataFrame(rows)


def flag_duplicate_years(years: pd.DataFrame) -> pd.DataFrame:
    out = years.copy()
    out["keep"] = True
    out["dup_note"] = ""
    grouped = out.groupby(["country", "modal_year"], dropna=False)
    for (country, myear), g in grouped:
        if pd.isna(myear) or len(g) <= 1:
            continue
        # Prefer folder whose survey_year contains the modal year.
        scores = []
        for i, r in g.iterrows():
            folder = str(r["survey_year"])
            score = 0
            if str(int(myear)) in folder:
                score += 10
            if r["has_kr"] and r["has_hr"]:
                score += 3
            if r["has_gps"]:
                score += 1
            # Prefer later recode letters only as tie-break via recode string length.
            scores.append((score, i))
        scores.sort(reverse=True)
        winner = scores[0][1]
        for i, r in g.iterrows():
            if i == winner:
                continue
            out.loc[i, "keep"] = False
            out.loc[i, "dup_note"] = f"duplicate modal_year={int(myear)}; kept recode {out.loc[winner, 'recode']}"
    return out


def run_inventory() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    manifest = discover_surveys()
    vardict = build_variable_dictionary(manifest)
    years = sample_interview_years(manifest)
    years = flag_duplicate_years(years)
    manifest.to_csv(TABLES / "survey_manifest.csv", index=False)
    vardict.to_csv(TABLES / "variable_dictionary.csv", index=False)
    years.to_csv(TABLES / "survey_years_dedup.csv", index=False)
    (TABLES / "inventory_summary.json").write_text(
        json.dumps(
            {
                "n_survey_rows": int(len(manifest)),
                "n_keep": int(years["keep"].sum()) if "keep" in years else int(len(years)),
                "n_drop_dup": int((~years["keep"]).sum()) if "keep" in years else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifest: {len(manifest)} survey files")
    print(years[["country", "survey_year", "recode", "modal_year", "keep", "dup_note"]].to_string(index=False))
    return manifest, vardict, years


if __name__ == "__main__":
    run_inventory()
