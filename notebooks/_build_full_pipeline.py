"""Build notebooks/full_pipeline.ipynb with analysis module source inlined in order."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUT = ROOT / "notebooks" / "full_pipeline.ipynb"

MODULES = [
    "config.py",
    "util.py",
    "io_dhs.py",
    "inventory.py",
    "load_merge.py",
    "study_population.py",
    "features.py",
    "models_loco.py",
    "shap_explain.py",
    "sensitivity.py",
    "figures_tables.py",
]

IMPORT_RE = re.compile(
    r"^(from analysis\.|import analysis\.)",
)

MAIN_RE = re.compile(r"\nif __name__ == [\"']__main__[\"']:[\s\S]*\Z")


def strip_module(text: str, name: str) -> str:
    lines = []
    skipping_import = False
    for line in text.splitlines(keepends=True):
        if skipping_import:
            if ")" in line:
                skipping_import = False
            continue
        if IMPORT_RE.match(line.lstrip()):
            if "(" in line and ")" not in line:
                skipping_import = True
            continue
        lines.append(line)
    body = "".join(lines)
    body = MAIN_RE.sub("\n", body)
    if name == "config.py":
        body = body.replace(
            "ROOT = Path(__file__).resolve().parents[1]",
            "ROOT = Path.cwd().resolve()  # notebook cwd is the repository root",
        )
    return body.strip() + "\n"


def cell(cell_type: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if cell_type == "code":
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def main() -> None:
    cells = [
        cell(
            "markdown",
            "# Full pipeline (module source inlined)\n\n"
            "First cell: seed 42, paths, `uv` kernel **sahel-malaria**.\n\n"
            "Runtime independence: each module below is pasted in order "
            "(`config`, `io_dhs`, `inventory`, `load_merge`, `study_population`, "
            "`features`, `models_loco`, `shap_explain`, `sensitivity`, `figures_tables`). "
            "`from analysis.*` imports are stripped so names come from earlier cells. "
            "The `analysis/` package is **not deleted**; reviewers already received it via OSF.\n\n"
            "This notebook **does not** overwrite `metrics_loco_primary.csv` or regenerate Figures 1–5. "
            "Primary tables must already exist from a previous `uv run python -m analysis.run_all`.",
        ),
        cell(
            "code",
            "from pathlib import Path\n"
            "import os\n"
            "import sys\n"
            "import json\n"
            "from dataclasses import dataclass\n"
            "\n"
            "SEED = 42\n"
            "HERE = Path.cwd().resolve()\n"
            "ROOT = HERE if (HERE / \"analysis\").is_dir() else HERE.parent\n"
            "os.chdir(ROOT)\n"
            "print(\"ROOT\", ROOT)\n"
            "print(\"SEED\", SEED)\n"
            "print(\"Use kernel sahel-malaria (uv .venv)\")\n"
            "print(sys.executable)\n",
        ),
    ]
    for name in MODULES:
        src = strip_module((ANALYSIS / name).read_text(encoding="utf-8"), name)
        cells.append(cell("markdown", f"## Inlined `{name}`"))
        cells.append(cell("code", src))
    cells.append(
        cell(
            "markdown",
            "## Run revision outputs (primary Table 2 locked)\n\n"
            "Loads saved primary metrics, then equal-country-weight LOCO, "
            "recalibration, expanded Table 1, Table S3, Figure S1.",
        )
    )
    cells.append(
        cell(
            "code",
            "ensure_dirs()\n"
            "primary = pd.read_csv(TABLES / \"metrics_loco_primary.csv\")\n"
            "print(primary[primary.model == \"lightgbm\"][[\"country\", \"auc\", \"oe\"]].to_string(index=False))\n"
            "_ = run_loco(tag=\"primary\", overwrite=False)\n"
            "analytic = pd.read_parquet(DATA_DIR / \"analytic.parquet\") if (DATA_DIR / \"analytic.parquet\").exists() else None\n"
            "print(antimalarial_exclusion(analytic).tail(1).to_string(index=False))\n"
            "eq = equal_country_weight_loco(analytic)\n"
            "print(eq[eq.model == \"lightgbm\"][[\"country\", \"auc\", \"oe\"]].to_string(index=False))\n"
            "print(recalibration_sensitivity().to_string(index=False))\n"
            "if analytic is not None:\n"
            "    table1_expanded(analytic)\n"
            "table_s3_subgroups()\n"
            "figure_s1_workflow()\n"
            "run_figures(regenerate_primary=False)\n"
            "print(\"done\")\n",
        )
    )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "sahel-malaria", "language": "python", "name": "sahel-malaria"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", OUT, "cells", len(cells))


if __name__ == "__main__":
    main()
