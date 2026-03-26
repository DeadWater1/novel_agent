from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from novel_agent.compaction import ContextCompactionManager, render_compact_summary_text
from novel_agent.config import AgentConfig
from novel_agent.controller import ControllerDependencies, NovelAgentController
from novel_agent.memory import SessionState, SessionStore
from novel_agent.registry import build_default_registry
from novel_agent.schemas import CompressionResult, DecisionOutput, DecisionReviewOutput, MemoryWrite
from novel_agent.session_meta import SessionMetaStore
from novel_agent.workspace import WorkspaceManager


class StubDecisionBackend:
    def __init__(self, decisions, reviews=None) -> None:
        if isinstance(decisions, list):
            self.decisions = list(decisions)
        else:
            self.decisions = [decisions]
        if reviews is None:
            self.reviews = [DecisionReviewOutput(verdict="accept", reason="default_accept")]
        elif isinstance(reviews, list):
            self.reviews = list(reviews)
        else:
            self.reviews = [reviews]
        self.calls = 0
        self.review_calls = 0
        self.last_messages = None
        self.last_workspace_docs = ""
        self.last_session_summary = ""
        self.last_compacted_session_context = ""
        self.last_recent_content_references = ""
        self.last_loop_events = None

    def decide(
        self,
        messages,
        workspace_docs,
        session_summary,
        compacted_session_context="",
        recent_content_references="",
        loop_events=None,
    ):
        self.calls += 1
        self.last_messages = messages
        self.last_workspace_docs = workspace_docs
        self.last_session_summary = session_summary
        self.last_compacted_session_context = compacted_session_context
        self.last_recent_content_references = recent_content_references
        self.last_loop_events = loop_events
        index = min(self.calls - 1, len(self.decisions) - 1)
        return self.decisions[index]

    def review_decision(
        self,
        *,
        messages,
        workspace_docs,
        session_summary,
        compacted_session_context="",
        recent_content_references="",
        loop_events=None,
        decision,
        user_text,
    ):
        self.review_calls += 1
        index = min(self.review_calls - 1, len(self.reviews) - 1)
        return self.reviews[index]

    def estimate_prompt_tokens(
        self,
        *,
        messages,
        workspace_docs,
        session_summary,
        compacted_session_context="",
        recent_content_references="",
        loop_events=None,
    ):
        merged = (
            str(messages)
            + workspace_docs
            + session_summary
            + compacted_session_context
            + recent_content_references
            + str(loop_events or [])
        )
        return max(len(merged) // 4, 1)


class StubCompressionBackend:
    def __init__(self) -> None:
        self.last_request = None

    def compress(self, request):
        self.last_request = request
        return CompressionResult(compressed_text="压缩结果", thinking="调试思考")


def build_controller(tmp_path: Path, decisions, reviews=None, **config_overrides):
    config = AgentConfig(
        workspace_root=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        show_debug_thinking=True,
        agent_max_loop_steps=4,
        **config_overrides,
    )
    workspace = WorkspaceManager(config)
    workspace.bootstrap()
    session_store = SessionStore(config.session_root, summary_max_chars=config.session_summary_max_chars)
    session_store.bootstrap()
    meta_store = SessionMetaStore(config.session_root)
    meta_store.bootstrap()
    compaction_manager = ContextCompactionManager(config, session_store, meta_store)
    compaction_manager.bootstrap()
    deps = ControllerDependencies(
        config=config,
        workspace=workspace,
        registry=build_default_registry(),
        decision_backend=StubDecisionBackend(decisions, reviews=reviews),
        compression_backend=StubCompressionBackend(),
        session_store=session_store,
        meta_store=meta_store,
        compaction_manager=compaction_manager,
    )
    return NovelAgentController(deps), deps


def _append_compression_turn(
    deps,
    session: SessionState,
    *,
    turn_index: int,
    user_content: str,
    assistant_content: str,
    timestamp: str | None = None,
) -> None:
    deps.session_store.append_events(
        session,
        [
            {
                "turn_index": turn_index,
                "event_type": "user_message",
                "role": "user",
                "content": user_content,
                **({"timestamp": timestamp} if timestamp else {}),
            },
            {
                "turn_index": turn_index,
                "event_type": "tool_result",
                "step_index": 1,
                "tool_name": "compress_chapter",
                "content": assistant_content,
                "payload": {"tool_trace": {"requested_tool": "compress_chapter", "status": "ok"}},
                **({"timestamp": timestamp} if timestamp else {}),
            },
            {
                "turn_index": turn_index,
                "event_type": "assistant_message",
                "role": "assistant",
                "content": assistant_content,
                **({"timestamp": timestamp} if timestamp else {}),
            },
        ],
    )


def test_out_of_scope_always_returns_exact_text(tmp_path: Path):
    decision = DecisionOutput(domain="out_of_scope", action="reject", assistant_reply="不应出现")
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
    assert deps.decision_backend.calls == 1
    daily_file = deps.workspace.ensure_daily_file()
    assert "今天讨论了人物关系" in daily_file.read_text(encoding="utf-8")
    assert "用户偏好关注人物关系" in (deps.workspace.root / "memory.md").read_text(encoding="utf-8")


def test_model_drives_compress_tool_instead_of_controller_shortcut(tmp_path: Path):
    decision = DecisionOutput(
        domain="novel",
        action="call_tool",
        tool_name="compress_chapter",
        tool_args={"raw_text": "这是一章小说原文"},
    )
    controller, deps = build_controller(tmp_path, decision, reviews=DecisionReviewOutput(verdict="accept", reason="tool_call"))
    session = SessionState()
    result = controller.handle_user_message(session, "请压缩这一章")
    assert result.reply == "压缩结果"
    assert result.action == "call_tool"
    assert deps.decision_backend.calls == 1
    assert deps.compression_backend.last_request is not None
    assert deps.compression_backend.last_request.raw_text == "这是一章小说原文"


def test_memory_search_can_happen_before_final_reply(tmp_path: Path):
    decisions = [
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_search",
            tool_args={"query": "人物关系"},
        ),
        DecisionOutput(
            domain="novel",
            action="direct_reply",
            assistant_reply="你之前偏好保留人物关系。",
        ),
    ]
    controller, deps = build_controller(tmp_path, decisions)
    deps.workspace.append_long_term_entries(["用户偏好压缩时保留人物关系"])
    session = SessionState()
    result = controller.handle_user_message(session, "我之前说过什么压缩偏好吗？")
    assert result.reply == "你之前偏好保留人物关系。"
    assert result.action == "call_tool"
    assert len(result.tool_trace["steps"]) == 1
    assert result.tool_trace["steps"][0]["requested_tool"] == "memory_search"


def test_direct_reply_review_can_trigger_retry_and_memory_lookup(tmp_path: Path):
    decisions = [
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="我猜第一次压缩是这个。"),
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_search",
            tool_args={"query": "第一次压缩内容", "search_mode": "lookup", "scope": "current_session"},
        ),
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="第一次压缩结果是：第一次压缩结果"),
    ]
    reviews = [
        DecisionReviewOutput(verdict="retry", reason="历史内容缺少证据，应先检索"),
        DecisionReviewOutput(verdict="accept", reason="已经检索"),
    ]
    controller, deps = build_controller(tmp_path, decisions, reviews=reviews)
    session = deps.session_store.create_session()
    _append_compression_turn(deps, session, turn_index=1, user_content="请压缩第一章原文", assistant_content="第一次压缩结果")
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None

    result = controller.handle_user_message(reloaded, "给我第一次压缩内容")
    assert result.reply == "第一次压缩结果是：第一次压缩结果"
    assert result.action == "call_tool"
    assert deps.decision_backend.calls == 3
    assert deps.decision_backend.review_calls == 1
    assert result.context_report["review_triggered"] is True
    assert result.context_report["review_verdict"] == "retry"
    event_types = [item["event_type"] for item in result.transcript_events]
    assert "decision_review" in event_types


def test_auto_recall_is_injected_into_workspace_docs(tmp_path: Path):
    decision = DecisionOutput(domain="novel", action="direct_reply", assistant_reply="沈砚和林秋曾是旧识。")
    controller, deps = build_controller(tmp_path, decision)
    deps.workspace.append_long_term_entries(["沈砚和林秋曾是旧识，后来因为误会分开。"])

    session = SessionState()
    controller.handle_user_message(session, "沈砚和林秋是什么关系？")

    assert "# Recalled Memory" in deps.decision_backend.last_workspace_docs
    assert "沈砚和林秋曾是旧识" in deps.decision_backend.last_workspace_docs


def test_memory_search_uses_compression_ledger_targets_for_lookup(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="call_tool", tool_name="memory_search", tool_args={"query": "第一次压缩内容"}),
    )
    session = deps.session_store.create_session()
    _append_compression_turn(deps, session, turn_index=1, user_content="请压缩第一章原文", assistant_content="第一次压缩结果")
    _append_compression_turn(deps, session, turn_index=2, user_content="请压缩第二章原文", assistant_content="第二次压缩结果")
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None

    execution = controller._execute_tool(
        reloaded,
        "memory_search",
        {"query": "第一次压缩内容", "search_mode": "lookup", "scope": "current_session"},
    )
    targets = [item["target"] for item in execution.payload["results"]]
    assert targets
    assert all("latest_compress" not in str(target) for target in targets)


def test_lookup_prefers_specific_compression_history_target_over_plain_compact_summary(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="call_tool", tool_name="memory_search", tool_args={"query": "第一个压缩文本"}),
    )
    session = deps.session_store.create_session()
    _append_compression_turn(
        deps,
        session,
        turn_index=1,
        user_content="请压缩第一章原文",
        assistant_content="第一次压缩结果全文，包含完整压缩后的章节内容。",
    )
    _append_compression_turn(
        deps,
        session,
        turn_index=2,
        user_content="请压缩第二章原文",
        assistant_content="第二次压缩结果全文，包含另一段完整章节内容。",
    )
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None

    execution = controller._execute_tool(
        reloaded,
        "memory_search",
        {"query": "第一个压缩文本", "search_mode": "lookup", "scope": "current_session"},
    )
    targets = [str(item["target"]) for item in execution.payload["results"]]
    assert targets
    assert targets[0].endswith("#compression_history:0")
    assert f"session_compact:{session.session_id}" not in targets[:1]


def test_memory_get_compression_history_target_returns_full_content(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="call_tool", tool_name="memory_get", tool_args={"target": "placeholder"}),
    )
    session = deps.session_store.create_session()
    _append_compression_turn(
        deps,
        session,
        turn_index=1,
        user_content="请压缩第一章原文",
        assistant_content="第一次压缩结果全文，包含完整压缩后的章节内容。",
    )
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None

    execution = controller._execute_tool(
        reloaded,
        "memory_get",
        {"target": f"session_compact:{session.session_id}#compression_history:0"},
    )
    assert execution.payload["resolved_target"] == f"session_compact:{session.session_id}#compression_history:0"
    assert execution.payload["content"] == "第一次压缩结果全文，包含完整压缩后的章节内容。"
    assert execution.terminal is True
    assert execution.final_reply == "第一次压缩结果全文，包含完整压缩后的章节内容。"


def test_compacted_session_context_uses_index_entries_instead_of_inline_compression_preview(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="call_tool", tool_name="memory_search", tool_args={"query": "placeholder"}),
    )
    session = deps.session_store.create_session()
    _append_compression_turn(
        deps,
        session,
        turn_index=1,
        user_content="请压缩第一章原文",
        assistant_content="第一次压缩结果全文，包含完整压缩后的章节内容。",
    )
    artifact = deps.compaction_manager.load_or_build_artifact(session)
    rendered = render_compact_summary_text(artifact)

    assert "Full compression text is not shown here" in rendered
    assert "preview=omitted" in rendered
    assert f"target=session_compact:{session.session_id}#compression_history:0" in rendered


def test_memory_get_default_delivers_full_text_to_user(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="call_tool", tool_name="memory_get", tool_args={"target": "placeholder"}),
    )
    session = deps.session_store.create_session()
    _append_compression_turn(
        deps,
        session,
        turn_index=1,
        user_content="请压缩第一章原文",
        assistant_content="这是第一次压缩后的完整章节内容。",
    )
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None
    deps.decision_backend.decisions[0].tool_args["target"] = f"session_compact:{session.session_id}#compression_history:0"

    result = controller.handle_user_message(reloaded, "给我第一次压缩的内容")
    assert result.reply == "这是第一次压缩后的完整章节内容。"
    assert result.action == "call_tool"
    assert result.tool_trace["final_tool"] == "memory_get"


def test_memory_get_observe_only_feeds_excerpt_back_to_model(tmp_path: Path):
    long_text = "完整压缩内容" * 80
    decisions = [
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_get",
            tool_args={"target": "placeholder", "delivery_mode": "observe"},
        ),
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="我已经基于读取结果继续分析。"),
    ]
    controller, deps = build_controller(tmp_path, decisions, memory_tool_max_chars=40)
    session = deps.session_store.create_session()
    _append_compression_turn(
        deps,
        session,
        turn_index=1,
        user_content="请压缩第一章原文",
        assistant_content=long_text,
    )
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None
    deps.decision_backend.decisions[0].tool_args["target"] = f"session_compact:{session.session_id}#compression_history:0"

    result = controller.handle_user_message(reloaded, "先读取第一次压缩内容，再继续分析")
    assert result.reply == "我已经基于读取结果继续分析。"
    assert result.action == "call_tool"
    assert deps.decision_backend.calls == 2
    assert deps.decision_backend.last_loop_events is not None
    rendered_loop = str(deps.decision_backend.last_loop_events)
    assert "preview" in rendered_loop
    assert long_text not in rendered_loop


def test_recent_content_reference_alias_is_available_to_model_and_memory_get(tmp_path: Path):
    decisions = [
        DecisionOutput(domain="novel", action="call_tool", tool_name="memory_get", tool_args={"target": "placeholder"}),
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="我已经看到最近内容引用。"),
    ]
    controller, deps = build_controller(tmp_path, decisions)
    session = deps.session_store.create_session()
    _append_compression_turn(
        deps,
        session,
        turn_index=1,
        user_content="请压缩第一章原文",
        assistant_content="这是第一次压缩后的完整章节内容。",
    )
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None
    deps.decision_backend.decisions[0].tool_args["target"] = f"session_compact:{session.session_id}#compression_history:0"

    first_result = controller.handle_user_message(reloaded, "给我第一次压缩的内容")
    assert first_result.reply == "这是第一次压缩后的完整章节内容。"

    second_result = controller.handle_user_message(reloaded, "继续看这个内容")
    assert second_result.reply == "我已经看到最近内容引用。"
    assert "content_ref:latest" in deps.decision_backend.last_recent_content_references

    execution = controller._execute_tool(reloaded, "memory_get", {"target": "content_ref:latest"})
    assert execution.final_reply == "这是第一次压缩后的完整章节内容。"


def test_agent_config_defaults_use_32k_input_budget():
    config = AgentConfig()
    assert config.decision_input_token_budget == 32768
    assert config.decision_output_max_new_tokens == 4096
    assert config.context_memory_flush_soft_threshold == 28672
    assert config.context_pruning_soft_budget == 30720
    assert config.context_auto_compact_token_threshold == 31744


def test_memory_search_recap_history_sessions_returns_compact_targets(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_search",
            tool_args={"query": "我们之前讨论了什么", "search_mode": "recap", "scope": "history_sessions"},
        ),
    )
    archived = SessionState()
    deps.session_store.append_events(
        archived,
        [
            {
                "turn_index": 1,
                "event_type": "user_message",
                "role": "user",
                "content": "我们先讨论人物关系，再压缩章节。",
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            },
            {
                "turn_index": 1,
                "event_type": "assistant_message",
                "role": "assistant",
                "content": "这次主要讨论人物关系和章节压缩。",
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            },
        ],
    )

    execution = controller._execute_tool(
        SessionState(),
        "memory_search",
        {"query": "我们之前讨论了什么", "search_mode": "recap", "scope": "history_sessions"},
    )
    assert execution.payload["results"]
    assert execution.payload["results"][0]["target"].startswith("session_compact:")
    assert execution.payload["results"][0]["source_kind"] == "session_compact"


def test_memory_search_recap_time_window_can_summarize_yesterday(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_search",
            tool_args={
                "query": "我们昨天讨论了什么",
                "search_mode": "recap",
                "scope": "time_window",
                "time_scope": {"from_days_ago": 1, "to_days_ago": 1},
            },
        ),
    )
    archived = SessionState()
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    deps.session_store.append_events(
        archived,
        [
            {"turn_index": 1, "event_type": "user_message", "role": "user", "content": "昨天我们聊剧情推进。", "timestamp": yesterday},
            {"turn_index": 1, "event_type": "assistant_message", "role": "assistant", "content": "重点是剧情推进和人物关系。", "timestamp": yesterday},
        ],
    )

    execution = controller._execute_tool(
        SessionState(),
        "memory_search",
        {
            "query": "我们昨天讨论了什么",
            "search_mode": "recap",
            "scope": "time_window",
            "time_scope": {"from_days_ago": 1, "to_days_ago": 1},
        },
    )
    assert execution.payload["results"]
    assert execution.payload["results"][0]["target"] == "session_compact:time_window:1-1#summary"


def test_context_report_records_memory_flush(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="继续讨论。"),
        memory_flush_soft_threshold_tokens=1,
        decision_reflection_enabled=False,
    )
    session = deps.session_store.create_session()
    _append_compression_turn(
        deps,
        session,
        turn_index=1,
        user_content="请压缩这一章，保留人物关系。",
        assistant_content="第一次压缩结果。",
    )
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None

    result = controller.handle_user_message(reloaded, "继续分析人物关系")
    assert result.context_report["memory_flush_applied"] is True
    assert deps.workspace.ensure_daily_file().read_text(encoding="utf-8")


def test_context_report_records_auto_compaction(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="这是总结。"),
        context_auto_compact_token_threshold=1,
        context_auto_compact_min_turns=1,
        decision_reflection_enabled=False,
    )
    session = SessionState()
    session.add_user_message("请分析这部小说的人物关系")
    session.add_assistant_message("这里重点是人物关系。")

    result = controller.handle_user_message(session, "再总结一下之前的内容")
    assert result.context_report["compaction_applied"] is True
    meta = deps.meta_store.load(session.session_id)
    assert meta is not None
    assert meta.latest_compaction_path.endswith(f"{session.session_id}.json")
