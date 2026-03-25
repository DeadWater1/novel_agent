from __future__ import annotations

from pathlib import Path

from novel_agent.config import AgentConfig
from novel_agent.workspace import WorkspaceManager


def test_workspace_doc_order_contains_agent_first(tmp_path: Path):
    config = AgentConfig(workspace_root=tmp_path / "workspace", session_root=tmp_path / "sessions")
    workspace = WorkspaceManager(config)
    workspace.bootstrap()
    docs = workspace.load_workspace_docs("摘要")
    agent_index = docs.find("# Agent")
    tools_index = docs.find("# Tools")
    memory_access_index = docs.find("# Memory Access")
    session_index = docs.find("# Session Summary")
    assert agent_index != -1
    assert tools_index != -1
    assert memory_access_index != -1
    assert session_index != -1
    assert agent_index < tools_index < memory_access_index < session_index


def test_memory_search_and_memory_get_are_available(tmp_path: Path):
    config = AgentConfig(workspace_root=tmp_path / "workspace", session_root=tmp_path / "sessions")
    workspace = WorkspaceManager(config)
    workspace.bootstrap()
    workspace.append_long_term_entries(["用户偏好压缩时保留人物关系"])
    workspace.append_daily_entries(["今天讨论了剧情推进"])

    results = workspace.memory_search("人物关系", max_results=3)
    assert results
    assert results[0]["source_id"] == "long_term"

    latest_daily = workspace.memory_get("daily_latest")
    assert latest_daily["target"] == "daily_latest"
    assert "今天讨论了剧情推进" in latest_daily["content"]
