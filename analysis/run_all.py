"""Full pipeline: Phase 1 dataset lock through figures and sensitivity."""

from __future__ import annotations

from analysis.config import ensure_dirs
from analysis.features import run_features
from analysis.figures_tables import run_figures
from analysis.models_loco import run_loco, run_senegal_temporal
from analysis.sensitivity import run_sensitivity
from analysis.shap_explain import run_shap
from analysis.study_population import run_study_population


def main() -> None:
    ensure_dirs()
    print("=== Phase 1: dataset lock / study population ===")
    pop = run_study_population()
    print("=== Phase 2: features and missingness ===")
    analytic = run_features(pop)
    print("=== Phase 3: LOCO models ===")
    run_loco(analytic, tag="primary", tune=True, overwrite=False)
    run_senegal_temporal(analytic)
    print("=== Phase 3: SHAP ===")
    run_shap(analytic)
    print("=== Phase 4: figures (primary 1–5 locked if already written) ===")
    run_figures(regenerate_primary=False)
    print("=== Phase 4: sensitivity analyses ===")
    run_sensitivity()
    print("=== Phase 4: figures after sensitivity (primary 1–5 still locked) ===")
    run_figures(regenerate_primary=False)
    print("Done. See reports/ and outputs/.")


if __name__ == "__main__":
    main()
