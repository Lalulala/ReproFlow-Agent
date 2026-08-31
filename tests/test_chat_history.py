from __future__ import annotations

from pathlib import Path

import pytest

from reproflow.storage import Store
from reproflow.ui import _chat_title_from_prompt


def test_chat_history_persists_across_store_instances(tmp_path: Path) -> None:
    project = tmp_path / "chat-project"
    project.mkdir()
    store = Store(project)
    session = store.create_chat_session("/tmp/repository", "api")
    store.add_chat_message(session["session_id"], "assistant", "欢迎")
    store.add_chat_message(session["session_id"], "user", "比较三个随机种子")
    store.update_chat_session(
        session["session_id"],
        title="三个种子实验",
        current_plan_id="repo-plan-demo",
        update_plan=True,
    )

    reopened = Store(project)
    restored = reopened.get_chat_session(session["session_id"])
    messages = reopened.list_chat_messages(session["session_id"])
    assert restored["title"] == "三个种子实验"
    assert restored["current_plan_id"] == "repo-plan-demo"
    assert [message["content"] for message in messages] == ["欢迎", "比较三个随机种子"]
    assert reopened.list_chat_sessions()[0]["session_id"] == session["session_id"]


def test_chat_history_rejects_unknown_roles(tmp_path: Path) -> None:
    store = Store(tmp_path)
    session = store.create_chat_session("/tmp/repository", "mock")
    with pytest.raises(ValueError, match="Unsupported chat role"):
        store.add_chat_message(session["session_id"], "system", "hidden")


def test_empty_chat_sessions_are_hidden_and_pruned(tmp_path: Path) -> None:
    store = Store(tmp_path)
    empty = store.create_chat_session(str(tmp_path), "mock")
    store.add_chat_message(empty["session_id"], "assistant", "welcome")
    completed = store.create_chat_session(str(tmp_path), "mock", "真实实验")
    store.add_chat_message(completed["session_id"], "user", "运行实验")
    store.add_chat_message(completed["session_id"], "assistant", "计划完成")

    assert [item["session_id"] for item in store.list_chat_sessions()] == [
        completed["session_id"]
    ]
    assert store.prune_empty_chat_sessions() == 1
    with pytest.raises(KeyError):
        store.get_chat_session(empty["session_id"])


def test_chat_title_is_created_from_first_prompt() -> None:
    assert _chat_title_from_prompt("请帮我 比较三个模型的实验效果") == "比较三个模型的实验效果"
    assert _chat_title_from_prompt("我想要复现 baseline。然后测试新方法") == "复现 baseline"
    assert _chat_title_from_prompt("运行一个非常非常非常非常非常非常长的实验目标描述") == (
        "运行一个非常非常非常非常非常非常长的实验目标…"
    )
