# ReproFlow Agent

ReproFlow is a reproducible agent workflow for machine-learning experiments and paper evidence.

The Day 6 milestone completes:

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
- Four Streamlit pages: Workflow, Runs, Evidence, and Knowledge.

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

The UI creates plans from goals, exposes the separate approval and run controls, and shows workflow
traces, run metrics and plots, evidence review/sync, and local knowledge/memory search.

## Verification

```bash
uv run ruff check src tests
REPROFLOW_RAG_BACKEND=lexical uv run pytest tests -q
```

All 34 tests pass. The Day 5/6 core module set has 81% combined coverage. The real acceptance audit
confirmed historical-memory reuse in a second plan, local Chroma/MiniLM retrieval over 16 chunks,
traceable report numbers, an evidence approval gate, stale detection, and zero exceptions across
all four Streamlit pages.

The experiment-loop design is inspired by Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch). ReproFlow generalizes it into a safe,
auditable research agent; it does not copy the nanochat training implementation.

## License

MIT. See [LICENSE](LICENSE).
