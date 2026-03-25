from __future__ import annotations

from pathlib import Path

from novel_agent.config import AgentConfig
from novel_agent.controller import ControllerDependencies, NovelAgentController
from novel_agent.memory import SessionState
from novel_agent.registry import build_default_registry
from novel_agent.schemas import CompressionResult, DecisionOutput, MemoryWrite
from novel_agent.workspace import WorkspaceManager


class StubDecisionBackend:
    def __init__(self, decision: DecisionOutput) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, messages, workspace_docs, session_summary):
        self.calls += 1
        return self.decision


class StubCompressionBackend:
    def __init__(self) -> None:
        self.last_request = None

    def compress(self, request):
        self.last_request = request
        return CompressionResult(compressed_text="压缩结果", thinking="调试思考")


def build_controller(tmp_path: Path, decision: DecisionOutput):
    config = AgentConfig(
        workspace_root=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        show_debug_thinking=True,
    )
    workspace = WorkspaceManager(config)
    workspace.bootstrap()
    deps = ControllerDependencies(
        config=config,
        workspace=workspace,
        registry=build_default_registry(),
        decision_backend=StubDecisionBackend(decision),
        compression_backend=StubCompressionBackend(),
    )
    return NovelAgentController(deps), deps


def test_out_of_scope_always_returns_exact_text(tmp_path: Path):
    decision = DecisionOutput(
        domain="out_of_scope",
        action="reject",
        assistant_reply="不应出现",
    )
    controller, _ = build_controller(tmp_path, decision)
    session = SessionState()
    result = controller.handle_user_message(session, "帮我写代码")
    assert result.reply == "无法回答"
    assert result.domain == "out_of_scope"
    assert result.action == "reject"


def test_invalid_tool_name_is_blocked(tmp_path: Path):
    decision = DecisionOutput(
        domain="novel",
        action="call_tool",
        tool_name="write_code",
        tool_args={"raw_text": "abc"},
    )
    controller, deps = build_controller(tmp_path, decision)
    session = SessionState()
    result = controller.handle_user_message(session, "请压缩这段")
    assert result.action == "reject"
    assert result.tool_trace["status"] == "blocked_invalid_tool"
    assert deps.compression_backend.last_request is None


def test_novel_direct_reply_writes_memory(tmp_path: Path):
    decision = DecisionOutput(
        domain="novel",
        action="direct_reply",
        assistant_reply="这是小说人物关系分析。",
        memory_write=MemoryWrite(
            daily=["今天讨论了人物关系"],
            long_term=["用户偏好关注人物关系"],
        ),
    )
    controller, deps = build_controller(tmp_path, decision)
    session = SessionState()
    result = controller.handle_user_message(session, "分析这部小说的人物关系")
    assert result.reply == "这是小说人物关系分析。"
    daily_file = deps.workspace.ensure_daily_file()
    assert "今天讨论了人物关系" in daily_file.read_text(encoding="utf-8")
    assert "用户偏好关注人物关系" in (deps.workspace.root / "memory.md").read_text(encoding="utf-8")


def test_compress_tool_runs_and_returns_tool_output(tmp_path: Path):
    decision = DecisionOutput(
        domain="novel",
        action="call_tool",
        tool_name="compress_chapter",
        tool_args={"raw_text": "这是一章小说原文"},
    )
    controller, deps = build_controller(tmp_path, decision)
    session = SessionState()
    result = controller.handle_user_message(session, "请压缩这一章")
    assert result.reply == "压缩结果"
    assert result.action == "call_tool"
    assert deps.compression_backend.last_request is not None
    assert deps.compression_backend.last_request.raw_text == "这是一章小说原文"


def test_force_compress_bypasses_decision_backend_and_extracts_raw_text(tmp_path: Path):
    decision = DecisionOutput(
        domain="novel",
        action="direct_reply",
        assistant_reply="不应被使用",
    )
    controller, deps = build_controller(tmp_path, decision)
    session = SessionState()
    result = controller.handle_user_message(
        session,
        "请帮我压缩下面这一章小说，保持人物关系不变：\n\n第一章，林秋推开门，发现沈砚已经坐在窗边等她。",
    )
    assert result.reply == "压缩结果"
    assert result.action == "call_tool"
    assert deps.decision_backend.calls == 0
    assert deps.compression_backend.last_request is not None
    assert deps.compression_backend.last_request.raw_text == "第一章，林秋推开门，发现沈砚已经坐在窗边等她。"
