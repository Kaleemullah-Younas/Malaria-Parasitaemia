# TRIPOD+AI checklist (manuscript companion)

This analysis is intended for a TRIPOD+AI-compliant manuscript. Items below map to pipeline outputs.

| Item | Status | Where |
| --- | --- | --- |
| Title identifies prediction model and target population | Draft | Afebrile children 6–59 months a fever rule would not test |
| Abstract structured; imperfect RDT reference stated | Draft | literature_check.md; manuscript later |
| Study design: multi-country DHS/MIS, individual-level | Done | Phase 1 report |
| Participants and setting per country-year | Done | table2_afebrile.csv, table1 |
| Outcome definition (RDT; microscopy sensitivity) | Done | study_population.py |
| Predictors available without a blood test (Hb excluded from primary) | Done | features.py |
| Sample size / events per country | Done | table2; flags if RDT+ < 30 |
| Numeric missing values below 10% are filled with the training-fold median (SimpleImputer). IterativeImputer is available in `make_preprocessor(iterative=True)` if a reviewer prefers MICE; median is used in the locked run because missingness is low and MICE on ~90k rows is not required for unbiased LOCO transport estimates.
| Model: elastic net and LightGBM with survey weights | Done | models_loco.py |
| Validation: leave-one-country-out; no refit on held-out country | Done | models_loco.py |
| Hyperparameters tuned only within training countries | Done | inner_select() |
| Discrimination with 95% CI | Done | cluster bootstrap, n=500 |
| Calibration plots plus O/E, slope, intercept per country | Done | figure1; metrics_loco_primary.csv |
| Decision curve vs named fever-based rule | Done | figure2; evaluation on all 6–59m with RDT |
| Fairness/equity subgroups | Done | wealth, urban/rural |
| Explainability stated as model behavior | Done | shap_explain.py caption |
| Sensitivity analyses labelled as such | Done | microscopy, Hb, missingness, fever don't-know |
| Random seed | Done | 42 |
| Code available | Done | analysis/ |

Random seed: 42. Code entry point: `uv run python -m analysis.run_all`.
