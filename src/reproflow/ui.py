from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from reproflow.approval import UnsafePlanError, approve_plan
from reproflow.context import build_context_pack
from reproflow.evidence import approve_claim, sync_evidence
from reproflow.planner import get_planner
from reproflow.rag import ChromaKnowledgeBase, LexicalKnowledgeBase
from reproflow.storage import Store
from reproflow.workflow import resume_workflow, start_workflow

PAGES = ("Workflow", "Runs", "Evidence", "Knowledge")


def project_root() -> Path:
    candidate = Path(sys.argv[-1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd().resolve()


def workflow_page(root: Path, store: Store) -> None:
    st.header("Workflow")
    st.subheader("Create an experiment plan")
    goal = st.text_input(
        "Experiment goal", value="比较三类模型在三个随机种子下的效果"
    )
    planner_mode = st.selectbox("Planner", ["mock", "api"])
    if st.button("Create draft plan"):
        try:
            context = build_context_pack(root, "planner", goal)
            plan = get_planner(planner_mode).create_plan(goal, context, root)
            store.save_plan(plan)
            store.add_plan_event(
                plan.plan_id,
                "created",
                planner_mode,
                "Generated from Streamlit experiment goal",
            )
            st.success(f"Created {plan.plan_id}")
        except (RuntimeError, ValueError) as error:
            st.error(str(error))

    plans = store.list_plans()
    if plans:
        st.subheader("Review and launch")
        selected_plan_id = st.selectbox("Plan", [plan.plan_id for plan in plans])
        selected_plan = store.get_plan(selected_plan_id)
        st.caption(f"Status: {selected_plan.status.value} · {selected_plan.title}")
        with st.expander("Structured plan"):
            st.json(selected_plan.model_dump(mode="json"))
        actor = st.text_input("Plan reviewer", value="Ethan")
        if selected_plan.status.value == "draft" and st.button("Approve safe plan"):
            try:
                approve_plan(root, selected_plan_id, actor, "Reviewed in Streamlit")
                st.rerun()
            except (UnsafePlanError, ValueError) as error:
                st.error(f"Approval blocked: {error}")
        if selected_plan.status.value == "approved" and st.button("Run approved workflow"):
            try:
                with st.spinner("Running approved experiment matrix..."):
                    start_workflow(root, selected_plan_id)
                st.rerun()
            except ValueError as error:
                st.error(f"Run blocked: {error}")

    st.divider()
    st.subheader("Workflow status")
    workflows = store.list_workflows()
    if not workflows:
        st.info("No workflows yet. Create, approve, and run a plan from the CLI.")
        return
    workflow_id = st.selectbox("Workflow", [item["workflow_id"] for item in workflows])
    state = store.get_workflow(workflow_id)
    records = state.get("run_records", [])
    succeeded = sum(record["status"] == "succeeded" for record in records)
    first, second, third = st.columns(3)
    first.metric("Stage", state["stage"])
    second.metric("Runs", len(records))
    third.metric("Succeeded", succeeded)
    if state.get("report_path"):
        st.markdown(f"[Open report]({state['report_path']})")
    if state["stage"] != "complete" and st.button("Resume incomplete workflow"):
        with st.spinner("Resuming safe tasks..."):
            resume_workflow(root, workflow_id)
        st.rerun()
    st.subheader("Agent Trace")
    st.dataframe(pd.DataFrame(store.list_traces(workflow_id)), width="stretch")


def runs_page(root: Path, store: Store) -> None:
    st.header("Runs")
    workflows = store.list_workflows()
    if not workflows:
        st.info("No run records are available.")
        return
    workflow_id = st.selectbox(
        "Workflow", [item["workflow_id"] for item in workflows], key="runs-workflow"
    )
    records = store.list_runs(workflow_id)
    rows = []
    for record in records:
        rows.append(
            {
                "run_id": record.run_id,
                "variant": record.variant,
                "seed": record.seed,
                "status": record.status.value,
                "duration_seconds": record.duration_seconds,
                **record.metrics,
                "error": record.error,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")
    state = store.get_workflow(workflow_id)
    if state.get("aggregate_path") and (root / state["aggregate_path"]).is_file():
        st.subheader("Aggregate")
        st.dataframe(pd.read_csv(root / state["aggregate_path"]), width="stretch")
    if state.get("plot_path") and (root / state["plot_path"]).is_file():
        st.image(str(root / state["plot_path"]), caption="Verified mean ± std")


def evidence_page(root: Path, store: Store) -> None:
    st.header("Evidence Registry")
    claims = store.list_claims()
    if not claims:
        st.info("No evidence proposals yet. Complete a workflow first.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "claim_id": claim.claim_id,
                    "status": claim.status.value,
                    "proposed": claim.proposed_status,
                    "metric": claim.metric,
                    "variant": claim.observed_variant,
                    "delta": claim.delta,
                    "workflow": claim.workflow_id,
                }
                for claim in claims
            ]
        ),
        width="stretch",
    )
    claim_id = st.selectbox("Claim", [claim.claim_id for claim in claims])
    claim = store.get_claim(claim_id)
    st.json(claim.model_dump(mode="json"))
    actor = st.text_input("Reviewer", value="Ethan")
    if claim.status.value == "proposed" and st.button("Approve selected evidence"):
        approve_claim(root, claim_id, actor)
        st.rerun()
    if st.button("Sync reviewed evidence to paper/"):
        try:
            registry, results = sync_evidence(root)
            st.success(f"Synced {registry.name} and {results.name}")
        except ValueError as error:
            st.error(str(error))


def knowledge_page(root: Path, store: Store) -> None:
    st.header("Knowledge")
    backend = st.selectbox("RAG backend", ["lexical", "chroma"])
    knowledge = (
        LexicalKnowledgeBase(root) if backend == "lexical" else ChromaKnowledgeBase(root)
    )
    if st.button("Rebuild local index"):
        with st.spinner("Indexing local protocols, papers, and reports..."):
            count = knowledge.index()
        st.success(f"Indexed {count} chunks")
    query = st.text_input("Search local evidence")
    if query:
        items = knowledge.search(query, limit=5)
        if not items:
            st.warning("No evidence found in the local knowledge index.")
        for item in items:
            location = f"page {item.page}" if item.page else (item.section or "document")
            with st.expander(f"{item.title} · {location} · score {item.score:.3f}"):
                st.caption(f"{item.path} · SHA256 {item.content_hash}")
                st.write(item.content)
    st.subheader("Experiment memories")
    memories = store.list_memories()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "kind": memory.kind,
                    "workflow": memory.workflow_id,
                    "text": memory.text,
                    "tags": ", ".join(memory.tags),
                }
                for memory in memories
            ]
        ),
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="ReproFlow Agent", page_icon="🧪", layout="wide")
    root = project_root()
    store = Store(root)
    st.sidebar.title("ReproFlow Agent")
    st.sidebar.caption(str(root))
    page = st.sidebar.radio("Page", PAGES)
    if page == "Workflow":
        workflow_page(root, store)
    elif page == "Runs":
        runs_page(root, store)
    elif page == "Evidence":
        evidence_page(root, store)
    else:
        knowledge_page(root, store)


if __name__ == "__main__":
    main()
