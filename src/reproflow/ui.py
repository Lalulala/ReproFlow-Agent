from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAIError

from reproflow.approval import UnsafePlanError, approve_plan
from reproflow.context import build_context_pack
from reproflow.evidence import approve_claim, sync_evidence
from reproflow.human_views import render_evidence_markdown, render_plan_markdown
from reproflow.planner import get_planner
from reproflow.rag import ChromaKnowledgeBase, LexicalKnowledgeBase
from reproflow.repo_agent import (
    RepoPlanStore,
    approve_repo_plan,
    create_repo_plan,
    create_repo_repair_plan,
    inspect_repository,
    render_manifest_markdown,
    render_repo_plan_markdown,
    run_repo_plan,
)
from reproflow.storage import Store
from reproflow.workflow import resume_workflow, start_workflow

PAGES = ("对话实验", "实验结果", "证据库", "知识与记忆")
PAGE_DESCRIPTIONS = {
    "对话实验": "用自然语言完成仓库理解、计划、审批与修复",
    "实验结果": "按实验查看运行、指标、图表和报告",
    "证据库": "按实验审核指标主张与论文证据",
    "知识与记忆": "按实验检索报告、经验和失败模式",
}
WELCOME_MESSAGE = (
    "你好，我是 ReproFlow。告诉我想在当前仓库完成什么实验。\n\n"
    "例如：找到主实验入口，复现 baseline，并用 42、43、44 三个随机种子比较新方法。\n\n"
    "我会先阅读仓库，再给出可读计划、代码 Diff、隔离环境和运行命令。"
)


def project_root() -> Path:
    candidate = Path(sys.argv[-1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd().resolve()


def apply_product_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rf-ink: #172033;
            --rf-muted: #64748b;
            --rf-brand: #5b5bd6;
            --rf-brand-soft: #eef0ff;
            --rf-border: #e6eaf2;
            --rf-surface: rgba(255,255,255,.88);
        }
        .stApp {
            background:
                radial-gradient(circle at 82% 3%, rgba(99,102,241,.09), transparent 26rem),
                linear-gradient(180deg, #fbfcff 0%, #f7f9fd 100%);
            color: var(--rf-ink);
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0;
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1220px;
            padding-top: 2.2rem;
            padding-bottom: 5rem;
        }
        section[data-testid="stSidebar"] {
            background: rgba(247,249,253,.96);
            border-right: 1px solid var(--rf-border);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.6rem;
        }
        [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
            visibility: hidden;
            height: 0;
        }
        button[data-testid="stExpandSidebarButton"] {
            visibility: visible !important;
            position: fixed !important;
            top: .75rem;
            left: .75rem;
            z-index: 1000000;
            width: 2.35rem;
            height: 2.35rem;
            border: 1px solid #dfe3ec;
            border-radius: .72rem;
            background: rgba(255,255,255,.96);
            color: var(--rf-ink);
            box-shadow: 0 7px 22px rgba(23,32,51,.12);
            pointer-events: auto;
        }
        button[data-testid="stExpandSidebarButton"]:hover {
            background: var(--rf-brand-soft);
            border-color: #cfd3ff;
        }
        .stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"] {
            color: var(--rf-ink);
        }
        .rf-brand {
            display: flex;
            gap: .75rem;
            align-items: center;
            padding: .35rem 0 1.15rem;
        }
        .rf-brand-mark {
            width: 2.35rem;
            height: 2.35rem;
            display: grid;
            place-items: center;
            border-radius: .8rem;
            background: linear-gradient(135deg, #6266e8, #7c3aed);
            color: white;
            box-shadow: 0 8px 22px rgba(91,91,214,.25);
            font-size: 1.15rem;
        }
        .rf-brand-name {font-size: 1.03rem; font-weight: 750; color: var(--rf-ink);}
        .rf-brand-sub {font-size: .72rem; color: var(--rf-muted); margin-top: .1rem;}
        .rf-nav-caption, .rf-history-caption {
            font-size: .72rem;
            font-weight: 700;
            color: #94a3b8;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin: .7rem 0 .2rem;
        }
        .st-key-rf_navigation .stButton {
            margin-bottom: .3rem;
        }
        .st-key-rf_navigation .stButton > button {
            width: 100%;
            min-height: 2.7rem;
            justify-content: flex-start;
            padding: .68rem .82rem;
            border: 0;
            border-radius: .72rem;
            background: transparent;
            box-shadow: none;
            transition: background .16s ease, box-shadow .16s ease;
        }
        .st-key-rf_navigation .stButton > button:hover {
            background: #eceff6;
        }
        .st-key-rf_navigation .stButton > button[kind="primary"] {
            background: #25283a !important;
            box-shadow: 0 8px 22px rgba(37,40,58,.16);
        }
        .st-key-rf_navigation .stButton > button p {
            color: #344056 !important;
            font-weight: 650;
        }
        .st-key-rf_navigation .stButton > button[kind="primary"] p {
            color: #fff !important;
        }
        .rf-hero {
            padding: 1.2rem 0 1.35rem;
        }
        .rf-kicker {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .28rem .62rem;
            border: 1px solid #dfe2ff;
            border-radius: 999px;
            background: var(--rf-brand-soft);
            color: #4f46b5;
            font-size: .74rem;
            font-weight: 700;
            letter-spacing: .04em;
        }
        .rf-hero h1 {
            margin: .7rem 0 .35rem;
            font-size: clamp(2rem, 4vw, 3.15rem);
            line-height: 1.08;
            letter-spacing: -.035em;
            color: var(--rf-ink);
        }
        .rf-hero p {color: var(--rf-muted); font-size: 1rem; max-width: 48rem; margin: 0;}
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--rf-surface);
            border-color: var(--rf-border) !important;
            border-radius: 1rem;
            box-shadow: 0 8px 30px rgba(23,32,51,.035);
        }
        [data-testid="stChatMessage"] {
            background: rgba(255,255,255,.78);
            border: 1px solid var(--rf-border);
            border-radius: 1rem;
            padding: .55rem .75rem;
            margin-bottom: .8rem;
            box-shadow: 0 5px 18px rgba(23,32,51,.025);
        }
        [data-testid="stChatMessage"] p {color: #273247 !important;}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #f0f1ff;
            border-color: #dfe2ff;
        }
        .stButton > button {
            border-radius: .65rem;
            border-color: #dfe3ec;
            font-weight: 600;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #5b5bd6, #6d4de3);
            border: none;
            box-shadow: 0 6px 18px rgba(91,91,214,.22);
        }
        [data-testid="stChatInput"] {border-color: #d9deea; border-radius: .85rem;}
        [data-testid="stWidgetLabel"] p {color: #475569 !important; font-weight: 600;}
        [data-baseweb="input"] > div, [data-baseweb="select"] > div {
            background: #fff !important;
            border-color: #dfe3ec !important;
        }
        [data-baseweb="input"] input {color: #172033 !important;}
        .rf-session-meta {font-size: .68rem; color: #94a3b8; margin: -.45rem 0 .35rem .35rem;}
        .rf-safety {
            margin-top: 1rem;
            padding: .7rem .8rem;
            border-radius: .7rem;
            background: #eefbf4;
            color: #26734d;
            font-size: .72rem;
            border: 1px solid #d8f2e4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="rf-brand">
          <div class="rf-brand-mark">R</div>
          <div><div class="rf-brand-name">ReproFlow</div>
          <div class="rf-brand-sub">科研实验 Agent</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_hero(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="rf-hero">
          <span class="rf-kicker">{kicker}</span>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_page(root: Path, store: Store) -> None:
    page_hero(
        "STRUCTURED WORKFLOW",
        "标准实验工作流",
        "使用固定 Schema 完成计划、审批、运行和恢复。",
    )
    st.subheader("创建实验计划")
    goal = st.text_input("实验目标", value="比较三类模型在三个随机种子下的效果")
    planner_mode = st.selectbox("规划模式", ["mock", "api"])
    if st.button("生成草稿计划"):
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
            st.success(f"已生成 {plan.plan_id}")
        except (RuntimeError, ValueError) as error:
            st.error(str(error))

    plans = store.list_plans()
    if plans:
        st.subheader("审核与启动")
        selected_plan_id = st.selectbox("实验计划", [plan.plan_id for plan in plans])
        selected_plan = store.get_plan(selected_plan_id)
        st.caption(f"Status: {selected_plan.status.value} · {selected_plan.title}")
        st.markdown(render_plan_markdown(selected_plan))
        with st.expander("查看机器可读原始数据（高级）"):
            st.json(selected_plan.model_dump(mode="json"))
        actor = st.text_input("计划审核人", value="Ethan")
        if selected_plan.status.value == "draft" and st.button("批准安全计划"):
            try:
                approve_plan(root, selected_plan_id, actor, "Reviewed in Streamlit")
                st.rerun()
            except (UnsafePlanError, ValueError) as error:
                st.error(f"审批被阻止：{error}")
        if selected_plan.status.value == "approved" and st.button("运行已批准工作流"):
            try:
                with st.spinner("正在运行已批准的实验矩阵……"):
                    start_workflow(root, selected_plan_id)
                st.rerun()
            except ValueError as error:
                st.error(f"运行被阻止：{error}")

    st.divider()
    st.subheader("工作流状态")
    workflows = store.list_workflows()
    if not workflows:
        st.info("暂无工作流。请先创建、审批并运行计划。")
        return
    workflow_id = st.selectbox("工作流", [item["workflow_id"] for item in workflows])
    state = store.get_workflow(workflow_id)
    records = state.get("run_records", [])
    succeeded = sum(record["status"] == "succeeded" for record in records)
    first, second, third = st.columns(3)
    first.metric("阶段", state["stage"])
    second.metric("运行数", len(records))
    third.metric("成功", succeeded)
    if state.get("report_path"):
        st.markdown(f"[打开报告]({state['report_path']})")
    if state["stage"] != "complete" and st.button("恢复未完成工作流"):
        with st.spinner("正在恢复安全任务……"):
            resume_workflow(root, workflow_id)
        st.rerun()
    st.subheader("Agent 轨迹")
    st.dataframe(pd.DataFrame(store.list_traces(workflow_id)), width="stretch")


STATUS_LABELS = {
    "complete": "已完成",
    "completed": "已完成",
    "partial_failure": "部分失败",
    "failed": "失败",
    "running": "运行中",
    "prepared": "待运行",
    "approved": "已批准",
}


def _experiment_time(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value[:16]


def _experiment_catalog(root: Path, store: Store) -> list[dict]:
    claim_counts: dict[str, int] = {}
    for claim in store.list_claims():
        claim_counts[claim.workflow_id] = claim_counts.get(claim.workflow_id, 0) + 1
    memory_counts: dict[str, int] = {}
    for memory in store.list_memories():
        if memory.workflow_id:
            memory_counts[memory.workflow_id] = memory_counts.get(memory.workflow_id, 0) + 1

    catalog = []
    repo_store = RepoPlanStore(root)
    for item in store.list_workflows():
        workflow_id = item["workflow_id"]
        title = item.get("goal") or ""
        try:
            if item["plan_id"].startswith("repo-plan-"):
                plan = repo_store.get(item["plan_id"])
            else:
                plan = store.get_plan(item["plan_id"])
            title = getattr(plan, "title", None) or getattr(plan, "goal", None) or title
        except KeyError:
            pass
        title = " ".join(str(title or "未命名实验").split())
        if len(title) > 38:
            title = f"{title[:38]}…"
        records = store.list_runs(workflow_id)
        status = STATUS_LABELS.get(item["stage"], item["stage"])
        updated_at = _experiment_time(item["updated_at"])
        catalog.append(
            {
                **item,
                "title": title,
                "status_label": status,
                "updated_label": updated_at,
                "run_count": len(records),
                "success_count": sum(record.status.value == "succeeded" for record in records),
                "claim_count": claim_counts.get(workflow_id, 0),
                "memory_count": memory_counts.get(workflow_id, 0),
                "label": f"{title} · {status} · {updated_at} · {workflow_id[-8:]}",
            }
        )
    return catalog


def _select_experiment(
    root: Path, store: Store, *, key: str, empty_message: str = "暂无实验记录。"
) -> dict | None:
    catalog = _experiment_catalog(root, store)
    if not catalog:
        st.info(empty_message)
        return None
    by_id = {item["workflow_id"]: item for item in catalog}
    workflow_id = st.selectbox(
        "选择实验",
        list(by_id),
        format_func=lambda value: by_id[value]["label"],
        key=key,
    )
    return by_id[workflow_id]


def _render_experiment_summary(experiment: dict) -> None:
    with st.container(border=True):
        st.markdown(f"#### {experiment['title']}")
        st.caption(
            f"实验 ID：`{experiment['workflow_id']}` · 最后更新：{experiment['updated_label']}"
        )
        first, second, third, fourth = st.columns(4)
        first.metric("状态", experiment["status_label"])
        second.metric("成功运行", f"{experiment['success_count']}/{experiment['run_count']}")
        third.metric("证据", experiment["claim_count"])
        fourth.metric("记忆", experiment["memory_count"])


def runs_page(root: Path, store: Store) -> None:
    page_hero("VERIFIED RESULTS", "实验结果", "选择一次实验，查看运行、聚合指标、失败原因与报告。")
    experiment = _select_experiment(root, store, key="results-experiment")
    if experiment is None:
        return
    _render_experiment_summary(experiment)
    workflow_id = experiment["workflow_id"]
    records = store.list_runs(workflow_id)
    rows = []
    for record in records:
        rows.append(
            {
                "运行 ID": record.run_id,
                "方案": record.variant,
                "随机种子": record.seed,
                "状态": record.status.value,
                "耗时（秒）": record.duration_seconds,
                **record.metrics,
                "错误": record.error,
            }
        )
    st.subheader("逐次运行")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.warning("这次实验没有产生可用的运行记录，请查看报告或失败信息。")
    state = store.get_workflow(workflow_id)
    if state.get("aggregate_path") and (root / state["aggregate_path"]).is_file():
        st.subheader("聚合结果")
        st.dataframe(pd.read_csv(root / state["aggregate_path"]), width="stretch")
    if state.get("plot_path") and (root / state["plot_path"]).is_file():
        st.image(str(root / state["plot_path"]), caption="已验证指标：mean ± std")
    if state.get("report_path") and (root / state["report_path"]).is_file():
        with st.expander("查看本次实验的中文报告", expanded=False):
            st.markdown((root / state["report_path"]).read_text(encoding="utf-8"))
    if state.get("errors"):
        with st.expander("查看失败信息"):
            for error in state["errors"]:
                st.error(str(error))


def _append_chat(
    store: Store,
    session_id: str,
    role: str,
    content: str,
    *,
    plan_id: str | None = None,
) -> None:
    store.add_chat_message(session_id, role, content, plan_id=plan_id)


def _draft_chat_session(root: Path, agent_mode: str) -> dict:
    return {
        "session_id": None,
        "title": None,
        "repository_path": str(root),
        "agent_mode": agent_mode,
        "current_plan_id": None,
    }


def _chat_title_from_prompt(prompt: str, max_length: int = 22) -> str:
    normalized = " ".join(prompt.strip().split())
    for prefix in ("请帮我", "帮我", "我想要", "我想", "能不能", "可以帮我"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].lstrip("，,：: ")
            break
    candidate = normalized.split("。", 1)[0].split("！", 1)[0].split("？", 1)[0]
    candidate = candidate.strip("，,：:；;。.！!？? ") or "实验对话"
    return candidate if len(candidate) <= max_length else f"{candidate[:max_length]}…"


def _ensure_chat_session(store: Store, root: Path, default_agent: str) -> dict:
    if st.session_state.get("reproflow_chat_draft"):
        return _draft_chat_session(root, default_agent)
    selected = st.session_state.get("reproflow_chat_session_id")
    if selected:
        try:
            session = store.get_chat_session(selected)
        except KeyError:
            st.session_state.pop("reproflow_chat_session_id", None)
        else:
            st.session_state.setdefault("chat-repository", session["repository_path"])
            st.session_state.setdefault("chat-agent-mode", session["agent_mode"])
            return session

    legacy_messages = st.session_state.pop("reproflow_chat_messages", None)
    legacy_plan = st.session_state.pop("reproflow_chat_plan_id", None)
    if legacy_messages:
        session = store.create_chat_session(str(root), default_agent, "迁移的对话")
        for message in legacy_messages:
            store.add_chat_message(
                session["session_id"], message["role"], message["content"]
            )
        if legacy_plan:
            session = store.update_chat_session(
                session["session_id"], current_plan_id=legacy_plan, update_plan=True
            )
    else:
        sessions = store.list_chat_sessions()
        if not sessions:
            st.session_state.reproflow_chat_draft = True
            st.session_state["chat-repository"] = str(root)
            st.session_state["chat-agent-mode"] = default_agent
            return _draft_chat_session(root, default_agent)
        session = sessions[0]
    st.session_state.reproflow_chat_session_id = session["session_id"]
    st.session_state.reproflow_chat_draft = False
    st.session_state["chat-repository"] = session["repository_path"]
    st.session_state["chat-agent-mode"] = session["agent_mode"]
    return session


def _history_time(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%m-%d %H:%M")
    except ValueError:
        return value[:16]


def _render_chat_history(
    store: Store, root: Path, default_agent: str, current_id: str | None
) -> None:
    st.sidebar.markdown('<div class="rf-history-caption">对话历史</div>', unsafe_allow_html=True)
    if st.sidebar.button("＋ 新建对话", use_container_width=True, type="primary"):
        st.session_state.reproflow_chat_session_id = None
        st.session_state.reproflow_chat_draft = True
        st.session_state["chat-repository"] = str(root)
        st.session_state["chat-agent-mode"] = default_agent
        st.rerun()
    for session in store.list_chat_sessions(limit=16):
        prefix = "● " if session["session_id"] == current_id else ""
        if st.sidebar.button(
            f"{prefix}{session['title']}",
            key=f"history-{session['session_id']}",
            use_container_width=True,
        ):
            st.session_state.reproflow_chat_session_id = session["session_id"]
            st.session_state.reproflow_chat_draft = False
            st.session_state["chat-repository"] = session["repository_path"]
            st.session_state["chat-agent-mode"] = session["agent_mode"]
            st.rerun()
        st.sidebar.markdown(
            f'<div class="rf-session-meta">{_history_time(session["updated_at"])}</div>',
            unsafe_allow_html=True,
        )


def _chat_result_message(state: dict) -> str:
    records = state.get("run_records", [])
    succeeded = sum(record.get("status") == "succeeded" for record in records)
    completed = state.get("stage") == "completed"
    heading = "实验工作流执行完成。\n\n" if completed else "实验工作流出现部分失败。\n\n"
    next_step = (
        "你可以继续发送新要求，开始下一轮实验计划。"
        if completed
        else "你可以点击 Repair Agent，让它根据失败日志生成新的待审批修复方案。"
    )
    return heading + (
        f"- 成功运行：{succeeded}/{len(records)}\n"
        f"- 当前阶段：`{state.get('stage', 'unknown')}`\n"
        f"- 汇总 CSV：`{state.get('aggregate_path', '-')}`\n"
        f"- 中文报告：`{state.get('report_path', '-')}`\n\n"
        f"报告和 Evidence 都只使用已验证的实验指标。{next_step}"
    )


def chat_agent_page(root: Path, store: Store) -> None:
    default_agent = "api" if os.getenv("OPENAI_API_KEY") else "mock"
    store.prune_empty_chat_sessions()
    session = _ensure_chat_session(store, root, default_agent)
    session_id = session["session_id"]
    _render_chat_history(store, root, default_agent, session_id)

    st.markdown(
        """
        <div class="rf-hero">
          <span class="rf-kicker">✦ RESEARCH WORKFLOW</span>
          <h1>和实验 Agent 一起工作</h1>
          <p>描述目标，审阅计划与代码，在可追溯的审批边界内完成实验、报告和修复。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("#### 实验设置")
        first, second, third = st.columns([3, 1, 1])
        repository = first.text_input("目标 Git 仓库", key="chat-repository")
        agent = second.selectbox("Agent 模式", ["api", "mock"], key="chat-agent-mode")
        actor = third.text_input("审核人", value="Ethan", key="chat-actor")
        mode_hint = "DeepSeek API · 会发送筛选脱敏后的源码" if agent == "api" else "离线 Mock"
        st.caption(f"当前模式：{mode_hint}　·　计划、依赖、Diff 和命令均需审批")
    if session_id and (
        repository != session["repository_path"] or agent != session["agent_mode"]
    ):
        session = store.update_chat_session(
            session_id,
            repository_path=repository,
            agent_mode=agent,
        )

    messages = (
        store.list_chat_messages(session_id)
        if session_id
        else [{"role": "assistant", "content": WELCOME_MESSAGE, "plan_id": None}]
    )
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("plan_id"):
                try:
                    message_plan = RepoPlanStore(root).get(message["plan_id"])
                except KeyError:
                    st.caption("关联计划已不存在。")
                else:
                    with st.expander(
                        f"查看计划详情 · {message_plan.status.value} · {message_plan.repo_plan_id}"
                    ):
                        st.markdown(render_repo_plan_markdown(message_plan))

    prompt = st.chat_input("描述实验目标，或补充你对当前计划的修改要求……")
    if prompt:
        is_first_exchange = session_id is None
        if not is_first_exchange:
            _append_chat(store, session_id, "user", prompt)
        current_id = session.get("current_plan_id")
        effective_goal = prompt
        if current_id:
            try:
                current = RepoPlanStore(root).get(current_id)
                if current.status.value == "draft":
                    effective_goal = f"{current.goal}\n\n用户补充要求：{prompt}"
            except KeyError:
                current_id = None
        try:
            with st.spinner("Agent 正在探索仓库、阅读代码并制定实验方案……"):
                manifest = inspect_repository(repository, effective_goal)
                plan = create_repo_plan(root, repository, effective_goal, agent=agent)
            response = (
                f"我已检查 **{manifest.file_count}** 个可安全读取的文件，并生成了新的"
                f"待审核计划 `{plan.repo_plan_id}`。\n\n"
                "代码 Diff、隔离环境和运行命令已收进下方计划详情。"
            )
            response_plan_id = plan.repo_plan_id
        except (OpenAIError, OSError, RuntimeError, ValueError) as error:
            response = (
                "这次规划没有成功，仓库未被修改，也没有命令被执行。"
                f"\n\n错误：`{error}`"
            )
            response_plan_id = None
        if is_first_exchange:
            session = store.create_chat_session(
                repository,
                agent,
                title=_chat_title_from_prompt(prompt),
            )
            session_id = session["session_id"]
            st.session_state.reproflow_chat_session_id = session_id
            st.session_state.reproflow_chat_draft = False
            _append_chat(store, session_id, "assistant", WELCOME_MESSAGE)
            _append_chat(store, session_id, "user", prompt)
        if response_plan_id:
            store.update_chat_session(
                session_id,
                repository_path=repository,
                agent_mode=agent,
                current_plan_id=response_plan_id,
                update_plan=True,
            )
        _append_chat(
            store,
            session_id,
            "assistant",
            response,
            plan_id=response_plan_id,
        )
        st.rerun()

    repo_plan_id = session.get("current_plan_id")
    if not repo_plan_id:
        return
    try:
        plan = RepoPlanStore(root).get(repo_plan_id)
    except KeyError:
        st.warning("当前对话关联的计划已不存在，请开始新对话。")
        return

    with st.container(border=True):
        st.markdown("#### 当前计划")
        status_column, environment_column, repair_column = st.columns(3)
        status_column.metric("状态", plan.status.value)
        environment_column.metric(
            "运行环境", plan.environment.mode if plan.environment else "current"
        )
        repair_column.metric("修复轮次", f"{plan.repair_attempt}/3")
        st.caption(f"{plan.repo_plan_id} · {plan.title}")
        if plan.status.value == "draft":
            st.warning("批准表示你已接受计划中的代码 Diff、隔离环境依赖和全部运行命令。")
        if plan.status.value == "draft" and st.button(
            "批准当前 Diff、依赖与命令", type="primary", key="chat-approve"
        ):
            try:
                approve_repo_plan(root, repo_plan_id, actor)
                _append_chat(
                    store,
                    session_id,
                    "assistant",
                    f"计划 `{repo_plan_id}` 已由 **{actor}** 批准，但尚未执行。"
                    "请点击“执行已批准计划”开始实验。",
                )
                st.rerun()
            except ValueError as error:
                st.error(f"审批被阻止：{error}")
        elif plan.status.value == "approved" and st.button(
            "执行已批准计划", type="primary", key="chat-run"
        ):
            try:
                with st.spinner("正在写入已批准的代码并运行实验……"):
                    state = run_repo_plan(root, repo_plan_id)
                _append_chat(store, session_id, "assistant", _chat_result_message(state))
                st.rerun()
            except (OSError, ValueError) as error:
                _append_chat(store, session_id, "assistant", f"执行被阻止或失败：`{error}`")
                st.rerun()
        elif plan.status.value == "partial_failure":
            repair_action, retry_action = st.columns(2)
            if repair_action.button("让 Repair Agent 生成新修复方案", type="primary"):
                try:
                    with st.spinner("Repair Agent 正在读取失败日志并生成新的 Diff……"):
                        repaired = create_repo_repair_plan(root, repo_plan_id)
                    session = store.update_chat_session(
                        session_id,
                        current_plan_id=repaired.repo_plan_id,
                        update_plan=True,
                    )
                    _append_chat(
                        store,
                        session_id,
                        "assistant",
                        "我根据失败日志生成了新的修复计划。原计划和失败产物保持不变；"
                        "新的 Diff、依赖与命令需要重新审批。",
                        plan_id=repaired.repo_plan_id,
                    )
                    st.rerun()
                except (OpenAIError, OSError, RuntimeError, ValueError) as error:
                    st.error(f"修复计划生成失败：{error}")
            if retry_action.button("原计划直接重试", key="chat-resume"):
                try:
                    with st.spinner("正在恢复失败实验；已成功的运行不会重复……"):
                        state = run_repo_plan(root, repo_plan_id, resume=True)
                    _append_chat(store, session_id, "assistant", _chat_result_message(state))
                    st.rerun()
                except (OSError, ValueError) as error:
                    _append_chat(store, session_id, "assistant", f"恢复失败：`{error}`")
                    st.rerun()
        elif plan.status.value == "completed":
            report_path = root / "runs" / repo_plan_id / "report.md"
            st.success("当前实验已完成。你可以继续描述下一轮实验。")
            if report_path.is_file():
                with st.expander("查看中文实验报告"):
                    st.markdown(report_path.read_text(encoding="utf-8"))


def repo_agent_page(root: Path) -> None:
    page_hero(
        "REPOSITORY AGENT",
        "仓库实验 Agent",
        "以表单方式探索仓库、生成 Diff、检查依赖并管理修复计划。",
    )
    repository = st.text_input("本地 Git 仓库", value=str(root))
    goal = st.text_area("实验要求", value="识别仓库中的实验入口并完成可复现对比")
    agent = st.selectbox("Agent 模式", ["mock", "api"])
    first, second = st.columns(2)
    if first.button("仅探索仓库"):
        try:
            st.markdown(render_manifest_markdown(inspect_repository(repository, goal)))
        except ValueError as error:
            st.error(str(error))
    if second.button("生成代码与执行计划"):
        try:
            with st.spinner("Agent 正在阅读仓库并决定执行方案..."):
                create_repo_plan(root, repository, goal, agent=agent)
            st.rerun()
        except (RuntimeError, ValueError) as error:
            st.error(str(error))

    plan_store = RepoPlanStore(root)
    plans = plan_store.list()
    if not plans:
        st.info("尚无仓库级执行计划。")
        return
    st.divider()
    repo_plan_id = st.selectbox("仓库计划", [item.repo_plan_id for item in plans])
    plan = plan_store.get(repo_plan_id)
    st.markdown(render_repo_plan_markdown(plan))
    with st.expander("查看机器可读原始数据（高级）"):
        st.json(plan.model_dump(mode="json"))
    actor = st.text_input("代码与命令审核人", value="Ethan")
    if plan.status.value == "draft" and st.button("批准 Diff、隔离依赖与全部执行命令"):
        try:
            approve_repo_plan(root, repo_plan_id, actor)
            st.rerun()
        except ValueError as error:
            st.error(str(error))
    if plan.status.value == "approved" and st.button("写入已批准代码并运行"):
        try:
            with st.spinner("正在执行已批准的仓库工作流..."):
                run_repo_plan(root, repo_plan_id)
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))
    if plan.status.value == "partial_failure" and st.button("恢复失败运行"):
        try:
            run_repo_plan(root, repo_plan_id, resume=True)
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))
    if plan.status.value == "partial_failure" and st.button("生成新的 Repair Plan"):
        try:
            with st.spinner("正在分析失败日志、依赖和相关源码……"):
                repaired = create_repo_repair_plan(root, repo_plan_id)
            st.success(f"已生成待审批修复计划：{repaired.repo_plan_id}")
            st.rerun()
        except (OpenAIError, OSError, RuntimeError, ValueError) as error:
            st.error(str(error))


def evidence_page(root: Path, store: Store) -> None:
    page_hero(
        "EVIDENCE REGISTRY",
        "科研证据库",
        "选择一次实验，把它的指标转成可审核、可追溯的论文主张。",
    )
    experiment = _select_experiment(
        root,
        store,
        key="evidence-experiment",
        empty_message="暂无实验记录，请先在对话中运行一次实验。",
    )
    if experiment is None:
        return
    _render_experiment_summary(experiment)
    claims = store.list_claims(workflow_id=experiment["workflow_id"])
    if not claims:
        st.info("所选实验还没有生成证据提议。只有经过验证的指标才能进入证据库。")
        return
    st.subheader("本次实验的证据")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "证据 ID": claim.claim_id,
                    "审核状态": claim.status.value,
                    "建议结论": claim.proposed_status,
                    "指标": claim.metric,
                    "实验方案": claim.observed_variant,
                    "相对基线变化": claim.delta,
                }
                for claim in claims
            ]
        ),
        width="stretch",
    )
    by_claim_id = {claim.claim_id: claim for claim in claims}
    claim_id = st.selectbox(
        "查看证据主张",
        list(by_claim_id),
        format_func=lambda value: (
            f"{by_claim_id[value].metric} · {by_claim_id[value].observed_variant} "
            f"vs {by_claim_id[value].baseline_variant} · {value}"
        ),
    )
    claim = store.get_claim(claim_id)
    st.markdown(render_evidence_markdown(claim))
    with st.expander("查看机器可读原始数据（高级）"):
        st.json(claim.model_dump(mode="json"))
    actor = st.text_input("证据审核人", value="Ethan")
    if claim.status.value == "proposed" and st.button("批准所选证据"):
        approve_claim(root, claim_id, actor)
        st.rerun()
    if st.button("同步所有已审核证据到 paper/"):
        try:
            registry, results = sync_evidence(root)
            st.success(f"已同步 {registry.name} 和 {results.name}")
        except ValueError as error:
            st.error(str(error))


def knowledge_page(root: Path, store: Store) -> None:
    page_hero(
        "MEMORY & RAG",
        "知识与记忆",
        "按项目或某次实验检索本地论文、实验报告、失败模式与历史经验。",
    )
    catalog = _experiment_catalog(root, store)
    by_id = {item["workflow_id"]: item for item in catalog}
    scope_options = ["__all__", *by_id]
    selected_scope = st.selectbox(
        "知识范围",
        scope_options,
        format_func=lambda value: (
            "全部项目知识与实验"
            if value == "__all__"
            else f"仅本次实验：{by_id[value]['label']}"
        ),
        key="knowledge-experiment",
    )
    selected_experiment = None if selected_scope == "__all__" else by_id[selected_scope]
    if selected_experiment:
        _render_experiment_summary(selected_experiment)
        st.caption("当前范围只展示本次实验的报告检索结果、经验和失败记忆。")
    else:
        st.caption("当前范围包含 knowledge/、paper/ 以及全部历史实验报告。")

    backend = st.selectbox("RAG 后端", ["lexical", "chroma"])
    knowledge = LexicalKnowledgeBase(root) if backend == "lexical" else ChromaKnowledgeBase(root)
    if st.button("重建本地索引"):
        with st.spinner("正在索引本地协议、论文和实验报告……"):
            count = knowledge.index()
        st.success(f"已索引 {count} 个知识块")
    query = st.text_input("搜索本地证据")
    if query:
        items = knowledge.search(query, limit=20 if selected_experiment else 5)
        if selected_experiment:
            workflow_path = f"runs/{selected_experiment['workflow_id']}/"
            items = [
                item
                for item in items
                if workflow_path in item.path.replace("\\", "/")
            ][:5]
        if not items:
            scope_label = "所选实验" if selected_experiment else "本地知识库"
            st.warning(f"{scope_label}中没有找到依据。")
        for item in items:
            location = f"page {item.page}" if item.page else (item.section or "document")
            with st.expander(f"{item.title} · {location} · score {item.score:.3f}"):
                st.caption(f"{item.path} · SHA256 {item.content_hash}")
                st.write(item.content)
    st.subheader("实验记忆")
    memories = store.list_memories(
        workflow_id=selected_experiment["workflow_id"] if selected_experiment else None
    )
    if memories:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "类型": memory.kind,
                        "实验": memory.workflow_id or "项目级",
                        "内容": memory.text,
                        "标签": ", ".join(memory.tags),
                        "记录时间": memory.created_at,
                    }
                    for memory in memories
                ]
            ),
            width="stretch",
        )
    else:
        st.info("当前范围还没有实验记忆。")


def main() -> None:
    st.set_page_config(
        page_title="ReproFlow · 科研实验 Agent",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "ReproFlow · 可复现科研实验与证据工作流"},
    )
    apply_product_theme()
    root = project_root()
    store = Store(root)
    render_sidebar_brand()
    st.sidebar.markdown('<div class="rf-nav-caption">功能导航</div>', unsafe_allow_html=True)
    page = st.session_state.get("reproflow_page", "对话实验")
    if page not in PAGES:
        page = "对话实验"
    with st.sidebar.container(key="rf_navigation"):
        for option in PAGES:
            if st.button(
                option,
                key=f"nav-{option}",
                type="primary" if page == option else "secondary",
                use_container_width=True,
            ):
                st.session_state.reproflow_page = option
                st.rerun()
    st.sidebar.caption(PAGE_DESCRIPTIONS[page])
    if page == "对话实验":
        chat_agent_page(root, store)
    elif page == "实验结果":
        runs_page(root, store)
    elif page == "证据库":
        evidence_page(root, store)
    else:
        knowledge_page(root, store)
    st.sidebar.markdown(
        '<div class="rf-safety">● 安全审批已启用<br>代码、依赖与 Evidence 均需确认</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
