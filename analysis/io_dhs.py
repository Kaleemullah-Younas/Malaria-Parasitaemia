"""DHS I/O helpers: DTA column census, HR person-reshape, GPS DBF read."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyreadstat

from analysis.config import HR_HH, HR_PERSON_PREFIXES, KR_CORE, KR_OPTIONAL_PREFIXES

_PERSON_RE = re.compile(
    r"^(hml35|hml32|hml12|hml16|sh212|sh214r)_(\d+)$",
    re.IGNORECASE,
)
_STEM_RE = re.compile(r"^([A-Z]{2})(KR|HR|GE)(.+)FL$", re.IGNORECASE)


def parse_recode_stem(path: Path) -> tuple[str, str, str] | None:
    m = _STEM_RE.match(path.stem)
    if not m:
        return None
    return m.group(1).upper(), m.group(2).upper(), m.group(3).upper()


def list_dta_columns(path: Path) -> list[str]:
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    return list(meta.column_names)


def map_lower(names: list[str]) -> dict[str, str]:
    return {n.lower(): n for n in names}


def dta_labels_na(path: Path) -> dict[str, bool]:
    """True if the dictionary label starts with NA (variable not collected)."""
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    labels = getattr(meta, "column_labels", None) or []
    names = list(meta.column_names)
    out: dict[str, bool] = {}
    for name, lab in zip(names, labels):
        text = (lab or "").strip()
        out[name.lower()] = text.upper().startswith("NA") or text.upper().startswith("NA-")
    return out


def read_dta(path: Path, wanted: list[str] | None = None) -> pd.DataFrame:
    names = list_dta_columns(path)
    lower_map = map_lower(names)
    usecols = None
    if wanted is not None:
        usecols = [lower_map[w] for w in wanted if w in lower_map]
        if not usecols:
            return pd.DataFrame()
    try:
        df, _ = pyreadstat.read_dta(str(path), usecols=usecols)
    except UnicodeDecodeError:
        df, _ = pyreadstat.read_dta(str(path), usecols=usecols, encoding="latin1")
    df.columns = [c.lower() for c in df.columns]
    return df


def wanted_kr_columns(available_lower: list[str]) -> list[str]:
    keep = [c for c in KR_CORE if c in available_lower]
    for col in available_lower:
        if col.startswith(KR_OPTIONAL_PREFIXES) and col not in keep:
            keep.append(col)
    return keep


def wanted_hr_columns(available_lower: list[str]) -> list[str]:
    keep = [c for c in HR_HH if c in available_lower]
    for col in available_lower:
        if _PERSON_RE.match(col) and col not in keep:
            keep.append(col)
    return keep


def reshape_person(hr: pd.DataFrame) -> pd.DataFrame:
    """Wide HR person slots (hml35_01, ...) to long rows keyed by hvidx."""
    slot_map: dict[str, dict[int, str]] = {}
    for col in hr.columns:
        m = _PERSON_RE.match(col)
        if not m:
            continue
        prefix = m.group(1).lower()
        idx = int(m.group(2))
        slot_map.setdefault(prefix, {})[idx] = col
    if not slot_map:
        return pd.DataFrame(columns=["hv001", "hv002", "hvidx"])

    id_cols = [c for c in ("hv001", "hv002") if c in hr.columns]
    frames = []
    indices = sorted({i for d in slot_map.values() for i in d})
    for idx in indices:
        piece = hr[id_cols].copy()
        piece["hvidx"] = idx
        empty = True
        for prefix, mapping in slot_map.items():
            col = mapping.get(idx)
            if col is None:
                piece[prefix] = pd.NA
                continue
            piece[prefix] = hr[col].values
            if piece[prefix].notna().any():
                empty = False
        if not empty:
            frames.append(piece)
    if not frames:
        return pd.DataFrame(columns=id_cols + ["hvidx"])
    long = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in long.columns if c not in id_cols + ["hvidx"]]
    if value_cols:
        long = long.loc[long[value_cols].notna().any(axis=1)].copy()
    return long


def recode_rdt(series: pd.Series, scheme: str) -> pd.Series:
    """Return 1/0/NA. scheme='standard' (0/1) or 'sh212' (1 pos, 2 neg)."""
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Float64")
    if scheme == "sh212":
        out = out.mask(s == 1, 1.0)
        out = out.mask(s == 2, 0.0)
    else:
        out = out.mask(s == 1, 1.0)
        out = out.mask(s == 0, 0.0)
    return out


def read_gps_dbf(dbf_path: Path) -> pd.DataFrame:
    from dbfread import DBF

    try:
        recs = list(DBF(str(dbf_path), encoding="latin-1", load=True))
    except UnicodeDecodeError:
        recs = list(DBF(str(dbf_path), encoding="cp1252", load=True))
    if not recs:
        return pd.DataFrame()
    gps = pd.DataFrame(recs)
    gps.columns = [str(c).lower() for c in gps.columns]
    cluster_col = next(
        (c for c in ("dhsclust", "cluster", "hv001", "dhsid") if c in gps.columns),
        None,
    )
    if cluster_col is None:
        return pd.DataFrame()
    lat_col = next((c for c in ("latnum", "latitude", "lat") if c in gps.columns), None)
    lon_col = next((c for c in ("longnum", "longitude", "long", "lon") if c in gps.columns), None)
    out = pd.DataFrame({"hv001": pd.to_numeric(gps[cluster_col], errors="coerce")})
    if lat_col:
        out["gps_lat"] = pd.to_numeric(gps[lat_col], errors="coerce")
    if lon_col:
        out["gps_lon"] = pd.to_numeric(gps[lon_col], errors="coerce")
    out = out.dropna(subset=["hv001"])
    out["hv001"] = out["hv001"].astype(int)
    if "gps_lat" in out.columns:
        out.loc[out["gps_lat"].isin([0, 9999, -9999]), "gps_lat"] = pd.NA
    if "gps_lon" in out.columns:
        out.loc[out["gps_lon"].isin([0, 9999, -9999]), "gps_lon"] = pd.NA
    return out.drop_duplicates("hv001")


def antimalarial_any(kr: pd.DataFrame) -> pd.Series:
    """Any recent antimalarial among ml13* / h37* malaria-drug items."""
    # ml13a–h and h37a–h typically include antimalarials; skip symptom-relief letters when labelled NA.
    candidates = [
        c
        for c in kr.columns
        if re.match(r"^(ml13[a-h]|h37[a-h]|h37e|h37da|h37aa|h37ab)$", c)
    ]
    if not candidates:
        return pd.Series(pd.NA, index=kr.index, dtype="Float64")
    block = kr[candidates].apply(pd.to_numeric, errors="coerce")
    taken = (block == 1).any(axis=1)
    asked = block.notna().any(axis=1)
    out = pd.Series(pd.NA, index=kr.index, dtype="Float64")
    out = out.mask(asked, taken.astype(float))
    return out
