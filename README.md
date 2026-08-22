# Sahel afebrile malaria case-finding

Survey-weighted, leave-one-country-out models to identify malaria RDT-positive children among those a fever-based testing rule would not test (DHS/MIS, seven Sahelian countries).

## Environment

```bash
uv venv .venv --python 3.11
uv sync
```

## Run

Phase 1 only (dataset lock / Step 2 N table):

```bash
uv run python -m analysis.run_phase1
```

Full pipeline (seed=42):

```bash
uv run python -m analysis.run_all
```

Outputs: `outputs/tables`, `outputs/figures` (300 DPI), `reports/phase1_dataset_lock.md`.
