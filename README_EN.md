# ReproFlow Agent

ReproFlow is a reproducible agent workflow for machine-learning experiments and paper evidence.

The Day 4 milestone completes this runnable path:

```text
goal → structured plan → human approval → safe execution → recovery
     → metric parsing → summary.csv → aggregate.csv → metric plot
```

## Implemented

- A CPU sklearn demo comparing Logistic Regression, Random Forest, and SVM across three seeds.
- Pydantic/YAML plans, a deterministic Mock Planner, and an OpenAI-compatible Planner.
- Preflight checks and an audited approval gate.
- An async, shell-free subprocess Runner with timeouts, cancellation, exit codes, an environment
  allowlist, and bounded logs.
- LangGraph orchestration with a persistent SQLite checkpoint.
- Idempotent resume: successful runs are skipped; failed, timed-out, or unfinished runs are retried.
- JSON, CSV, and Regex metric parsers with strict numeric validation.
- Per-run `summary.csv`, per-variant `aggregate.csv`, `failures.csv`, and a mean ± std PNG plot.
- Traceable run snapshots, environments, logs, metrics, manifests, and retry history.

## Setup and run

```bash
uv python install 3.12
uv sync --extra dev
uv run reproflow init .

REPROFLOW_RAG_BACKEND=lexical uv run reproflow plan \
  --goal "compare three models across three random seeds" \
  --planner mock

uv run reproflow preflight <plan_id>
uv run reproflow approve <plan_id> --actor Ethan --reason "matrix and paths reviewed"
uv run reproflow run <plan_id>
uv run reproflow workflow-show <plan_id>
```

The current implementation uses `plan_id` as `workflow_id`, preventing accidental overwrite or
duplicate execution of the same approved plan.

## Failure and resume demo

```bash
uv run reproflow run <plan_id> \
  --simulate-failure svm:43 \
  --simulate-timeout random_forest:44 \
  --timeout-seconds 1

uv run reproflow resume <plan_id>
```

Resume clears injected demo faults by default and retries only incomplete runs. Use
`--keep-simulations` to repeat them. A one-shot process interruption can be demonstrated with
`--crash-after 2`. Pressing Control+C cancels the active subprocess and records the cancellation.

## Outputs

```text
runs/<workflow_id>/
├── <variant>-seed-<seed>/
│   ├── artifacts/
│   ├── plan_snapshot.yaml
│   ├── environment.json
│   ├── stdout.log
│   ├── stderr.log
│   ├── metrics.json
│   ├── manifest.json
│   └── attempts.jsonl
├── summary.csv
├── aggregate.csv
├── failures.csv
└── plots/metrics.png
```

## Verification

```bash
uv run ruff check src tests
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests -q
```

All 26 tests pass. The Day 3/4 core module set has 86% combined coverage, including crash recovery,
timeouts, cancellation, missing metrics, partial success, parser behavior, checkpoints, CSV outputs,
and plots. The real sklearn acceptance run completed 9/9; a fault-injected workflow recovered to
9/9 without rerunning successful tasks.

The experiment-loop design is inspired by Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch). ReproFlow generalizes it toward safe,
auditable research workflows; it does not copy the nanochat training implementation.

## License

MIT. See [LICENSE](LICENSE).
