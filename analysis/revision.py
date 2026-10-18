"""JPMedAI revision analyses. Does not overwrite primary Table 2 or Figures 1–5."""

from __future__ import annotations

from analysis.config import DATA_DIR, ensure_dirs
from analysis.figures_tables import run_figures
from analysis.sensitivity import run_revision_sensitivities


def main() -> None:
    ensure_dirs()
    analytic = None
    path = DATA_DIR / "analytic.parquet"
    if path.exists():
        import pandas as pd

        analytic = pd.read_parquet(path)
    print("=== Revision sensitivities (equal-country weight, recalibration, antimalarial n) ===")
    run_revision_sensitivities(analytic)
    print("=== Table 1 expanded, Table S3, Figure S1, captions ===")
    run_figures(regenerate_primary=False)
    print("Revision outputs written. Primary metrics_loco_primary.csv was not overwritten.")


if __name__ == "__main__":
    main()
