# ReproFlow Agent

ReproFlow is a reproducible workflow system for machine-learning experiments and paper evidence.

The Day 1 milestone provides a complete CPU experiment loop: three sklearn classifiers, three fixed
seeds, independent run directories, immutable configuration snapshots, structured metrics, logs,
manifests, a TSV result ledger, and a fixed baseline summary.

## Quick start

```bash
uv python install 3.12
uv sync --extra dev
uv run python scripts/day1_run_all.py --tag first-demo
```

The command must finish with `9/9 succeeded`. See [README.md](README.md) for the full Chinese guide,
artifact layout, acceptance commands, and roadmap.

The research-loop design is inspired by Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch). ReproFlow generalizes the idea toward safe,
auditable ML workflows; it does not copy the nanochat training implementation.

