"""Pipeline configuration (plan file: 00_config.py)."""

from __future__ import annotations

from pathlib import Path

SEED = 42
N_BOOTSTRAP = 500
DPI = 300
RDT_POSITIVE_FLAG_N = 30
SUBGROUP_RDT_POS_FLAG_N = 50

ROOT = Path(__file__).resolve().parents[1]
DHS_ROOT = ROOT / "DHS_clean"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
DATA_DIR = OUTPUTS / "data"
MODELS_DIR = OUTPUTS / "models"
REPORTS = ROOT / "reports"

COUNTRY_DIR_TO_NAME = {
    "Burkina_Faso": "Burkina Faso",
    "Gambia": "Gambia",
    "Mali": "Mali",
    "Mauritania": "Mauritania",
    "Niger": "Niger",
    "Nigeria": "Nigeria",
    "Senegal": "Senegal",
}

# Surveys that cannot support the afebrile endpoint (no KR / no fever item).
EXCLUDE_NO_KR = {("Mali", "2010")}

# Recode collisions: same country-year restated in two folders. Filled after inventory
# if two stems share a modal interview year; prefer the dedicated year folder.
PREFERRED_STEMS: dict[tuple[str, str], str] = {}

KR_CORE = [
    "caseid",
    "v000",
    "v001",
    "v002",
    "v003",
    "v005",
    "v006",
    "v007",
    "v008",
    "v016",
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
    "b16",
    "b19",
    "hw1",
    "h22",
    "h32y",
    "h32z",
    "hw53",
    "hml32",
    "hml35",
]

KR_OPTIONAL_PREFIXES = ("ml13", "h37")

HR_HH = [
    "hhid",
    "hv001",
    "hv002",
    "hv005",
    "hv006",
    "hv007",
    "hv021",
    "hv022",
    "hv023",
    "hv024",
    "hv025",
    "hv009",
    "hv213",
    "hv214",
    "hv215",
    "hv253",
    "hml1",
]

HR_PERSON_PREFIXES = ("hml35", "hml32", "hml12", "hml16", "sh212", "sh214r")

# JMP-style improved water / sanitation (DHS recode groups).
IMPROVED_WATER = {11, 12, 13, 14, 21, 31, 41, 51, 61, 62, 71}
IMPROVED_TOILET = {11, 12, 13, 14, 15, 21, 22, 41}

# hml12: 1 treated net, 2 untreated, 3 both.
NET_USED_CODES = {1, 2, 3}

# Standard hml35: 0 neg, 1 pos. Senegal 2008 sh212: 1 pos, 2 neg.
VALID_RDT_STANDARD = {0, 1}

DCA_THRESHOLDS = [i / 100 for i in range(5, 41)]
CALIBRATION_BINS = 10

LGBM_GRID = [
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200, "max_depth": 3},
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "max_depth": 5},
    {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 150, "max_depth": 4},
]

ELASTICNET_GRID = {"l1_ratio": [0.2, 0.5, 0.8], "C": [0.1, 1.0, 10.0]}

MICROSCOPY_HEADLINE = [
    ("Burkina Faso", "2021"),
    ("Nigeria", "2021"),
]

PRIMARY_FEATURES = [
    "wealth",
    "urban",
    "educ",
    "hhsize",
    "water_improved",
    "toilet_improved",
    "net_own",
    "net_used",
    "floor_grp",
    "wall_grp",
    "roof_grp",
    "age_months",
    "female",
    "month_sin",
    "month_cos",
    "lat_band",
    "sahel_nigeria",
]

HB_FEATURE = "hb_gdl"


def ensure_dirs() -> None:
    for path in (OUTPUTS, TABLES, FIGURES, DATA_DIR, MODELS_DIR, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
