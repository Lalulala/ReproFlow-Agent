# ReproFlow Architecture

```mermaid
flowchart LR
    U[Researcher] --> UI[Conversational UI / CLI]
    UI --> RA[Repository Agent]
    RA --> CP[Stage-specific ContextPack]
    KB[(Local RAG)] --> CP
    MM[(Experiment memory)] --> CP
    CP --> PP[Plan + code diff + run matrix]
    PP --> H1{Human approval}
    H1 -->|approved| DP[Dependency Preflight]
    DP --> VE[Project-owned uv environment]
    VE --> LG[LangGraph workflow]
    LG --> RUN[Shell-free Runner]
    RUN --> MET[Verified metrics]
    MET --> CSV[Summary / aggregate CSV]
    MET --> REP[Chinese report]
    MET --> ER[Evidence proposal]
    RUN -->|failure| FIX[Repair Agent]
    FIX --> H1
    ER --> H2{Evidence approval}
    H2 --> PAPER[Paper evidence registry]
    LG <--> CK[(SQLite checkpoints)]
    REP --> MM
```

## Trust boundaries

| Boundary | Input | Control | Output |
| --- | --- | --- | --- |
| Repository context | Local source and documentation | Ignore rules, size limits, secret filtering | Bounded manifest and excerpts |
| Planner | Goal and ContextPack | Pydantic schema, locked executable fields | Reviewable plan and diff |
| Dependency setup | Requirement files | Normalization, explicit argv, project uv environment | Isolated interpreter |
| Runner | Approved argv | `shell=False`, allowlist, path containment, timeout | Logs, metrics, manifests |
| Reporter | Verified CSV and metrics | No unverified numerical generation | Traceable report |
| Evidence sync | Proposed claims | Human approval and staleness audit | Two paper evidence files |

## Persistent data

- `.reproflow/reproflow.db`: plans, workflows, runs, traces, memories, evidence, and chat history.
- `.reproflow/checkpoints.sqlite`: LangGraph recovery checkpoints.
- `runs/<workflow_id>/`: immutable run logs, metrics, manifests, summaries, plots, and reports.
- `paper/`: approved evidence registry and generated results summary only.
