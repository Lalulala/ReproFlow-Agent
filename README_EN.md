# ReproFlow Agent

ReproFlow is a reproducible agent workflow for machine-learning experiments and paper evidence.

The Day 2 milestone adds structured experiment planning, security preflight checks, and an audited
human approval gate on top of the Day 1 reproducible sklearn experiment loop.

## What works now

- A CPU demo comparing Logistic Regression, Random Forest, and SVM across seeds 42, 43, and 44.
- Independent logs, metrics, environment snapshots, manifests, and immutable run directories.
- Pydantic experiment plans persisted as YAML and indexed in SQLite.
- A deterministic Mock Planner that works without an API key.
- An OpenAI-compatible Planner whose output cannot alter executable fields.
- Checks for command allowlists, shell syntax, command shape, paths, dependencies, arguments, and
  artifact collisions.
- Audited approve/reject actions; an unsafe plan cannot become approved.

## Setup

```bash
uv python install 3.12
uv sync --extra dev
```

## Day 2 walkthrough

```bash
uv run reproflow init .

REPROFLOW_RAG_BACKEND=lexical uv run reproflow plan \
  --goal "compare three models across three random seeds" \
  --planner mock

uv run reproflow plans
uv run reproflow plan-show <plan_id>
uv run reproflow preflight <plan_id>
uv run reproflow approve <plan_id> --actor Ethan --reason "matrix and paths reviewed"
```

To use an OpenAI-compatible endpoint, set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`REPROFLOW_MODEL`, then replace `--planner mock` with `--planner api`. Only descriptive fields from
the model are accepted; commands, variants, seeds, paths, timeout, and metrics remain locked to a
local safe template.

Run the Day 1 experiment matrix with:

```bash
uv run python scripts/day1_run_all.py --tag first-demo
```

## Verification

```bash
uv run ruff check src tests
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests -q
```

The Day 2 suite covers valid planning, malicious API output, path escapes, shell operators,
unapproved artifact overwrites, argument injection, audit events, and CLI behavior. Core Day 2
module coverage is currently 84%. Live API execution requires a user-provided key; its contract and
safety boundary are tested with a simulated compatible response.

The research-loop design is inspired by Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch). ReproFlow generalizes the idea toward safe,
auditable ML workflows; it does not copy the nanochat training implementation.

## License

MIT. See [LICENSE](LICENSE).
