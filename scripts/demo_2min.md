# ReproFlow 2-minute demo script

> Day 1 currently covers the reproducible experiment loop. Later workflow steps are roadmap items.

1. Show the three important files: `experiment.py`, `day1_run_all.py`, and `pyproject.toml`.
2. Run `uv run python scripts/day1_run_all.py --tag recorded-demo`.
3. Point out the nine independent run directories while the matrix executes.
4. Open one `environment.json` and explain command, Python version, Git commit, and script hash.
5. Open the matching `metrics.json`, `stdout.log`, and `manifest.json`.
6. Open `results.tsv` and show the three models across seeds 42, 43, and 44.
7. Open `baseline.json` and explain that logistic regression is the fixed comparison baseline.
8. End on the README roadmap: plan approval, checkpointing, RAG, reporting, and evidence registry.

