# Contributing

ReproFlow accepts focused changes that preserve reproducibility, explicit approval, and artifact
traceability.

## Local checks

```bash
uv sync --extra dev
uv run ruff check src tests scripts
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests --cov=reproflow --cov-fail-under=80 -q
uv run reproflow eval --project . --minimum-passes 18
```

Tests must not require an API key or network access. Add a deterministic fixture for new Agent
behavior and keep generated commands visible for review. Never commit `.env`, raw private logs,
virtual environments, `.reproflow/`, or `runs/`.

## Pull requests

Describe the user-visible behavior, approval boundary, failure mode, and verification performed.
Security-sensitive changes should include a regression test for the rejected path.
