# Sahel afebrile malaria case-finding

Survey-weighted, leave-one-country-out models to identify malaria RDT-positive children among those a fever-based testing rule would not test (DHS/MIS, seven Sahelian countries).

## Environment

```bash
uv venv .venv --python 3.11
uv sync
uv run python -m ipykernel install --user --name sahel-malaria --display-name "Python (sahel-malaria)"
```

Jupyter kernel: **`sahel-malaria`** (the project `.venv`). Select it in both notebooks.

## Run

Phase 1 only (dataset lock / Step 2 N table):

```bash
uv run python -m analysis.run_phase1
```

Full pipeline (seed=42). Primary LOCO metrics are not overwritten if `outputs/tables/metrics_loco_primary.csv` already exists:

```bash
uv run python -m analysis.run_all
```

JPMedAI revision analyses (equal-country-weight LOCO, recalibration, expanded Table 1, Table S3, Figure S1). Does not overwrite primary Table 2 or regenerate Figures 1–5:

```bash
uv run python -m analysis.revision
```

Notebooks (kernel `sahel-malaria`):

- `notebooks/reproduce.ipynb` — imports `analysis.*`
- `notebooks/full_pipeline.ipynb` — module source inlined; rebuild with `uv run python notebooks/_build_full_pipeline.py`

Outputs: `outputs/tables`, `outputs/figures` (300 DPI), `reports/phase1_dataset_lock.md`. Manuscript Markdown for the JPMedAI revision is in `docs/revision/`.
