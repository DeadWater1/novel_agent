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
    memory_index = docs.find("# Long-Term Memory")
    session_index = docs.find("# Session Summary")
    assert agent_index != -1
    assert tools_index != -1
    assert memory_index != -1
    assert session_index != -1
    assert agent_index < tools_index < memory_index < session_index
