# ReproFlow Agent

[![CI](https://github.com/Lalulala/ReproFlow-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Lalulala/ReproFlow-Agent/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![Agent evals](https://img.shields.io/badge/Agent%20evals-20%2F20-1f883d.svg)](evals/latest_results.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ReproFlow is a reproducible agent workflow for machine-learning experiments and paper evidence.

![ReproFlow conversational experiment UI](docs/assets/ui-overview.svg)

The Day 6 and repository-agent milestones complete:

```text
goal → RAG and experiment memory → structured plan → human approval
     → safe execution and recovery → verified metrics → Markdown report
     → proposed evidence → human review → paper evidence sync
```

## Implemented agent capabilities

- Mock and OpenAI-compatible planners with schema-constrained output.
- Command, path, argument, dependency, and artifact guardrails.
- Human approval before execution and before evidence synchronization.
- LangGraph state orchestration with persistent SQLite checkpoints.
- Async shell-free execution, timeout, cancellation, bounded logs, and idempotent resume.
- Experiment, failure, and lesson memories generated from verified workflows.
- Local Markdown, TXT, PDF, paper evidence, and historical report RAG.
- Offline lexical retrieval and Chroma with local `all-MiniLM-L6-v2` embeddings.
- Stage-specific ContextPacks: Planner gets relevant knowledge and memory, Runner gets only the
  approved plan, and Reporter gets only verified results.
- JSON, CSV, and Regex metric parsing; per-run and aggregate CSV files; plots.
- Jinja2 Markdown reports whose numerical values come only from verified artifacts.
- Proposed Evidence Claims linked to runs, metrics, commits, configuration hashes, and artifacts.
- Reviewed evidence synchronization limited to `paper/evidence_registry.jsonl` and
  `paper/generated_results.md`.
- A repository-level Agent that inspects a local Git repository, chooses which files to read,
  proposes auditable code diffs, defines the run matrix and metrics, and executes only after
  human approval.
- Dependency Preflight compares requested repository requirements with the current environment and
  proposes a project-owned uv virtual environment when packages or Python versions conflict.
- Repair Agent turns bounded failure logs into a separate draft plan with a new diff, dependency
  decision, and approval gate; it never overwrites or silently approves the failed plan.
- Four UI entries: Chat Experiments, Results, Evidence Registry, and Knowledge & Memory. Structured
  workflow and repository execution remain underlying Agent and CLI capabilities.

## Quick start

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
```

The completed workflow automatically writes metrics, summaries, a plot, memories, `report.md`,
and proposed Evidence Claims.

## Run against an existing repository

Inspect a repository without writing files or running commands:

```bash
uv run reproflow repo inspect /path/to/repository \
  --goal "reproduce the main experiment and compare three seeds"
```

After loading an OpenAI-compatible API configuration, ask the Agent to inspect the repository and
propose its own code and execution plan:

```bash
set -a
source .env
set +a

uv run reproflow repo plan /path/to/repository \
  --goal "reproduce the main experiment and compare three seeds" \
  --agent api
```

Review the generated `repo_plans/<repo_plan_id>.md`, then approve and execute it:

```bash
uv run reproflow repo show <repo_plan_id>
uv run reproflow repo dependencies <repo_plan_id>
uv run reproflow repo approve <repo_plan_id> --actor Ethan
uv run reproflow repo run <repo_plan_id>
uv run reproflow repo resume <repo_plan_id>  # retry only failed runs
```

Create a separately reviewed repair after a runtime or environment failure:

```bash
uv run reproflow repo repair <failed_repo_plan_id> \
  --feedback "preserve the experiment matrix and repair the root cause"
uv run reproflow repo approve <new_repair_plan_id> --actor Ethan
uv run reproflow repo run <new_repair_plan_id>
```

API mode sends filtered and redacted source snippets to the configured provider. `.env`,
credentials, private keys, binaries, datasets, caches, and virtual environments are excluded.
Before approval, the Agent performs no code writes and executes no repository commands.

## Knowledge and memory

```bash
uv run reproflow memories
uv run reproflow context-show --stage planner --task "compare the models"

uv run reproflow knowledge index --backend lexical
uv run reproflow knowledge search "ROC-AUC protocol" --backend lexical

uv run reproflow knowledge index --backend chroma
uv run reproflow knowledge search "Which model performed best?" --backend chroma
```

The first Chroma run downloads an approximately 79 MB local MiniLM model. Documents are embedded
locally. Retrieval results include path, section/page, score, tags, and a content hash.

## Reports and evidence

```bash
uv run reproflow report <workflow_id> --narrator mock
uv run reproflow evidence list
uv run reproflow evidence show <claim_id>
uv run reproflow evidence approve <claim_id> --actor Ethan
uv run reproflow evidence sync
```

API Narration is available with `--narrator api` after loading `.env`. It sends verified summary
and aggregate data to the configured provider. The model may not emit numbers; a numeric response
falls back to the deterministic Mock Narrator.

Proposed evidence cannot be synced. Reviewed claims become supported, contradicted, or inconclusive.
Commit or configuration changes can be audited with:

```bash
uv run reproflow evidence audit-stale <claim_id> --plan-id <plan_id>
```

## UI

```bash
uv run reproflow ui
```

The default Chat Experiments page accepts natural-language experiment requests, presents repository
inspection findings and code diffs inside the conversation, and keeps approval and execution as
explicit buttons. Starting a chat creates only a transient draft: it is added to history and named
from the first request only after the first assistant response completes. Conversations and linked
plan IDs are then persisted in the project SQLite database, so a previous session can be resumed
from the sidebar after restarting the UI. The sidebar uses dark full-width active buttons without
radio dots and does not expose the local project path. Results and Evidence first select a readable
experiment label containing title, status, timestamp, and short ID. Knowledge & Memory can use either
project-wide scope or one experiment, filtering reports and memories accordingly. Structured workflow
and repository execution remain behind the Agent and available through the CLI.

## Verification

```bash
uv run ruff check src tests scripts
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests \
  --cov=reproflow --cov-fail-under=80 -q
uv run reproflow eval --project . --minimum-passes 18
```

All 50 automated tests pass with 85.84% core coverage. The deterministic Agent acceptance suite passes
20/20 cases across planning, command guardrails, RAG retrieval, and metric consistency; see
[`evals/latest_results.json`](evals/latest_results.json). The acceptance audit
confirmed historical-memory reuse in a second plan, local Chroma/MiniLM retrieval over 16 chunks,
traceable report numbers, an evidence approval gate, stale detection, and experiment-scoped Results,
Evidence, and Memory pages. Repository-agent tests additionally cover source discovery, code creation
only after approval, source-drift blocking, secret exclusion/redaction, the CLI approval chain, and
a complete nine-run sklearn experiment.

Real API planning and controlled execution were also evaluated against micrograd,
homemade-machine-learning, and ML-From-Scratch. See
[`evals/repository_compatibility.md`](evals/repository_compatibility.md) for the success matrix,
failure analysis, and fixes applied.

Release materials include the [architecture and trust boundaries](docs/architecture.md),
[two-minute demo script](scripts/demo_2min.md), [security policy](SECURITY.md), and
[version history](CHANGELOG.md).

The experiment-loop design is inspired by Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch). ReproFlow generalizes it into a safe,
auditable research agent; it does not copy the nanochat training implementation.

## License

MIT. See [LICENSE](LICENSE).
