from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel

from novel_agent.compaction import ContextCompactionManager, render_compact_summary_text
from novel_agent.config import AgentConfig
from novel_agent.controller import ControllerDependencies, NovelAgentController
from novel_agent.embedding_index import EmbeddingIndexManager
from novel_agent.memory import SessionState, SessionStore
from novel_agent.registry import build_default_registry
from novel_agent.toolbox import ToolExecutionResult, ToolHandler, ToolRuntimeContext
from novel_agent.schemas import (
    BackendHealth,
    CompressionResult,
    DecisionOutput,
    DecisionReviewOutput,
    ExecutionPlanOutput,
    MemoryWrite,
    PlanStep,
)
from novel_agent.session_meta import SessionMetaStore
from novel_agent.workspace import WorkspaceManager


class StubDecisionBackend:
    def __init__(self, decisions, reviews=None, plans=None, allowed_tools: tuple[str, ...] | None = None) -> None:
        if isinstance(decisions, list):
            self.decisions = list(decisions)
        else:
            self.decisions = [decisions]
        self.allowed_tools = tuple(allowed_tools or ())
        if reviews is None:
            self.reviews = [DecisionReviewOutput(verdict="accept", reason="default_accept")]
        elif isinstance(reviews, list):
            self.reviews = list(reviews)
        else:
            self.reviews = [reviews]
        if plans is None:
            inferred_steps = [
                PlanStep(
                    step_index=index,
                    goal=(decision.user_goal or f"step {index}"),
                    preferred_action=decision.action if decision.action in ("direct_reply", "call_tool") else "direct_reply",
                    preferred_tool=(
                        decision.tool_name
                        if decision.action == "call_tool"
                        and decision.tool_name in self.allowed_tools
                        else None
                    ),
                )
                for index, decision in enumerate(self.decisions, start=1)
            ]
            self.plans = [ExecutionPlanOutput(user_goal=self.decisions[0].user_goal or "stub_goal", steps=inferred_steps)]
        elif isinstance(plans, list):
            self.plans = list(plans)
        else:
            self.plans = [plans]
        self.plan_calls = 0
        self.calls = 0
        self.review_calls = 0
        self.last_plan_output_text = ""
        self.last_messages = None
        self.last_workspace_docs = ""
        self.last_session_summary = ""
        self.last_compacted_session_context = ""
        self.last_recent_content_references = ""
        self.last_loop_events = None

    def plan_turn(
        self,
        *,
        user_text,
        messages,
        workspace_docs,
        session_summary,
        tool_prompt_docs="",
        tool_names=(),
        compacted_session_context="",
        recent_content_references="",
        loop_events=None,
    ):
        self.plan_calls += 1
        index = min(self.plan_calls - 1, len(self.plans) - 1)
        self.last_plan_output_text = '{"user_goal":"stub","steps":[]}'
        return self.plans[index]

    def decide(
        self,
        messages,
        workspace_docs,
        session_summary,
        tool_prompt_docs="",
        tool_names=(),
        compacted_session_context="",
        recent_content_references="",
        loop_events=None,
        execution_plan=None,
        current_step=None,
        completed_steps=None,
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
        execution_plan=None,
        current_step=None,
        completed_steps=None,
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
        tool_prompt_docs="",
        tool_names=(),
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


class StubEmbeddingBackend:
    name = "embedding_backend"

    def healthcheck(self) -> BackendHealth:
        return BackendHealth(ok=True, name=self.name, detail="stub")

    def similarity(self, query: str, text: str) -> float:
        return self.similarity_batch(query, [text])[0]

    def similarity_batch(self, query: str, texts: list[str]) -> list[float]:
        clean_query = query.strip()
        query_chars = set(clean_query)
        scores: list[float] = []
        for text in texts:
            clean_text = text.strip()
            if not clean_query or not clean_text:
                scores.append(0.0)
                continue
            overlap = len(query_chars & set(clean_text)) / max(len(query_chars), 1)
            exact_bonus = 1.0 if clean_query in clean_text else 0.0
            scores.append(overlap + exact_bonus)
        return scores

    def embed_query(self, query: str):
        return self.embed_texts([query], prompt_type="query")

    def embed_texts(self, texts: list[str], *, prompt_type: str = "document"):
        import torch

        vectors = []
        for text in texts:
            clean_text = text.strip()
            if not clean_text:
                vectors.append([0.0] * 128)
                continue
            vector = [0.0] * 128
            for char in clean_text:
                vector[ord(char) % 128] += 1.0
            if prompt_type == "query":
                vector[0] += 0.5
            vectors.append(vector)
        tensor = torch.tensor(vectors, dtype=torch.float32)
        return torch.nn.functional.normalize(tensor, p=2, dim=1)


def build_controller(tmp_path: Path, decisions, reviews=None, plans=None, registry=None, **config_overrides):
    config = AgentConfig(
        workspace_root=tmp_path / "workspace",
        session_root=tmp_path / "sessions",
        embedding_index_root=tmp_path / "runtime" / "embeddings",
        show_debug_thinking=True,
        agent_max_loop_steps=6,
        **config_overrides,
    )
    embedding_backend = StubEmbeddingBackend()
    embedding_index_manager = EmbeddingIndexManager(config, embedding_backend)
    embedding_index_manager.bootstrap()
    workspace = WorkspaceManager(
        config,
        embedding_backend=embedding_backend,
        embedding_index_manager=embedding_index_manager,
    )
    workspace.bootstrap()
    session_store = SessionStore(config.session_root, summary_max_chars=config.session_summary_max_chars)
    session_store.bootstrap()
    meta_store = SessionMetaStore(config.session_root)
    meta_store.bootstrap()
    compaction_manager = ContextCompactionManager(config, session_store, meta_store)
    compaction_manager.bootstrap()
    registry = registry or build_default_registry(config.tool_registry_enabled)
    deps = ControllerDependencies(
        config=config,
        workspace=workspace,
        registry=registry,
        decision_backend=StubDecisionBackend(decisions, reviews=reviews, plans=plans, allowed_tools=registry.names()),
        compression_backend=StubCompressionBackend(),
        embedding_backend=embedding_backend,
        embedding_index_manager=embedding_index_manager,
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


class FakeToolArgs(BaseModel):
    topic: str


class FakeToolHandler(ToolHandler):
    name = "fake_tool"
    description = "Fake tool for toolbox extensibility tests."
    args_model = FakeToolArgs
    prompt_doc = """
## fake_tool

- 名称：`fake_tool`
- 作用：返回一个固定格式的测试结果
- 输入：
  - `topic`: 必填
""".strip()

    def execute(
        self,
        session: SessionState,
        parsed_args: FakeToolArgs,
        runtime: ToolRuntimeContext,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=self.name,
            observation_text=f"topic={parsed_args.topic}",
            payload={"topic": parsed_args.topic},
            tool_trace={"requested_tool": self.name, "status": "ok"},
            terminal=False,
        )


def test_tool_validation_error_is_wrapped_without_controller_branching(tmp_path: Path):
    decision = DecisionOutput(
        domain="novel",
        action="call_tool",
        tool_name="embedding_similarity",
        tool_args={"query": "人物关系"},
    )
    controller, _ = build_controller(tmp_path, decision)
    session = SessionState()

    result = controller.handle_user_message(session, "比较这两段文本是不是相似")
    assert result.action == "reject"
    assert result.tool_trace["status"] == "invalid_args"
    assert result.tool_trace["requested_tool"] == "embedding_similarity"


def test_controller_can_dispatch_registered_fake_tool_without_new_branch(tmp_path: Path):
    registry = build_default_registry()
    registry.register(FakeToolHandler())
    decision = DecisionOutput(
        domain="novel",
        action="call_tool",
        tool_name="fake_tool",
        tool_args={"topic": "角色弧光"},
    )
    controller, _ = build_controller(tmp_path, decision, registry=registry)
    execution = controller._execute_tool(SessionState(), "fake_tool", {"topic": "角色弧光"})

    assert execution.tool_name == "fake_tool"
    assert execution.payload["topic"] == "角色弧光"
    assert execution.observation_text == "topic=角色弧光"


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
    context = deps.workspace.structured_memory.load_context()
    assert "用户偏好关注人物关系" in context.user_preferences


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
    assert deps.compression_backend.last_request.enable_thinking is True
    assert deps.compression_backend.last_request.max_new_tokens == 3584


def test_compress_tool_ignores_model_generation_overrides_and_uses_config_defaults(tmp_path: Path):
    decision = DecisionOutput(
        domain="novel",
        action="call_tool",
        tool_name="compress_chapter",
        tool_args={
            "raw_text": "这是一章小说原文",
            "max_new_tokens": 500,
            "temperature": 0.9,
            "seed": 123,
            "top_p": 0.6,
            "top_k": 5,
            "enable_thinking": False,
        },
    )
    controller, deps = build_controller(
        tmp_path,
        decision,
        reviews=DecisionReviewOutput(verdict="accept", reason="tool_call"),
        compression_temperature=0.33,
        compression_top_p=0.88,
        compression_top_k=17,
        compression_seed=77,
    )
    session = SessionState()
    controller.handle_user_message(session, "请压缩这一章")
    assert deps.compression_backend.last_request is not None
    assert deps.compression_backend.last_request.max_new_tokens == 3584
    assert deps.compression_backend.last_request.temperature == 0.33
    assert deps.compression_backend.last_request.seed == 77
    assert deps.compression_backend.last_request.top_p == 0.88
    assert deps.compression_backend.last_request.top_k == 17
    assert deps.compression_backend.last_request.enable_thinking is True


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
    plan = ExecutionPlanOutput(
        user_goal="先检索历史压缩内容再回答",
        steps=[
            PlanStep(step_index=1, goal="检索第一次压缩内容", preferred_action="call_tool", preferred_tool="memory_search"),
            PlanStep(step_index=2, goal="基于检索结果回答用户", preferred_action="direct_reply", preferred_tool=None),
        ],
    )
    decisions = [
        DecisionOutput(
            domain="novel",
            action="direct_reply",
            assistant_reply="我猜第一次压缩是这个。",
            needs_review=True,
            review_reason="依赖历史事实，需要先复核",
        ),
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
    controller, deps = build_controller(tmp_path, decisions, reviews=reviews, plans=plan)
    session = deps.session_store.create_session()
    _append_compression_turn(deps, session, turn_index=1, user_content="请压缩第一章原文", assistant_content="第一次压缩结果")
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None

    result = controller.handle_user_message(reloaded, "给我第一次压缩内容")
    assert result.reply == "第一次压缩结果是：第一次压缩结果"
    assert result.action == "call_tool"
    assert deps.decision_backend.calls == 3
    assert deps.decision_backend.review_calls == 0
    assert result.context_report["review_triggered"] is False
    event_types = [item["event_type"] for item in result.transcript_events]
    assert "premature_direct_reply_blocked" in event_types


def test_risk_gated_review_skips_low_risk_direct_reply(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="这是直接回答。"),
    )
    result = controller.handle_user_message(SessionState(), "直接回答即可")
    assert result.reply == "这是直接回答。"
    assert deps.decision_backend.review_calls == 0
    assert result.context_report["review_triggered"] is False


def test_risk_gated_review_runs_when_decision_marks_high_risk(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(
            domain="novel",
            action="direct_reply",
            assistant_reply="这条回答需要复核。",
            needs_review=True,
            review_reason="涉及历史事实一致性",
        ),
        reviews=DecisionReviewOutput(verdict="accept", reason="evidence_sufficient"),
    )
    result = controller.handle_user_message(SessionState(), "请确认历史设定")
    assert result.reply == "这条回答需要复核。"
    assert deps.decision_backend.review_calls == 1
    assert result.context_report["review_triggered"] is True
    assert result.context_report["review_verdict"] == "accept"


def test_review_mode_always_preserves_legacy_behavior(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="旧行为仍会复核。"),
        reviews=DecisionReviewOutput(verdict="accept", reason="legacy"),
        decision_review_mode="always",
    )
    controller.handle_user_message(SessionState(), "继续")
    assert deps.decision_backend.review_calls == 1


def test_review_mode_disabled_skips_review_even_for_high_risk(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(
            domain="novel",
            action="direct_reply",
            assistant_reply="即使高风险也不复核。",
            needs_review=True,
            review_reason="高风险",
        ),
        reviews=DecisionReviewOutput(verdict="retry", reason="should_not_run"),
        decision_review_mode="disabled",
    )
    result = controller.handle_user_message(SessionState(), "继续")
    assert result.reply == "即使高风险也不复核。"
    assert deps.decision_backend.review_calls == 0


def test_workspace_docs_no_longer_inject_auto_recall(tmp_path: Path):
    decision = DecisionOutput(domain="novel", action="direct_reply", assistant_reply="沈砚和林秋曾是旧识。")
    controller, deps = build_controller(tmp_path, decision, embedding_index_enabled=False)
    deps.workspace.append_long_term_entries(["沈砚和林秋曾是旧识，后来因为误会分开。"])

    session = SessionState()
    controller.handle_user_message(session, "沈砚和林秋是什么关系？")

    assert "# Recalled Memory" not in deps.decision_backend.last_workspace_docs


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
    specific_target = f"session_compact:{session.session_id}#compression_history:0"
    plain_target = f"session_compact:{session.session_id}"
    assert specific_target in targets
    if plain_target in targets:
        assert targets.index(specific_target) < targets.index(plain_target)


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
    assert execution.terminal is False
    assert execution.final_reply == ""


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


def test_memory_get_default_observes_and_explicit_deliver_returns_full_text_to_user(tmp_path: Path):
    controller, deps = build_controller(
        tmp_path,
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_get",
            tool_args={"target": "placeholder", "delivery_mode": "deliver"},
        ),
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


def test_non_final_direct_reply_is_blocked_and_controller_continues_current_step(tmp_path: Path):
    plan = ExecutionPlanOutput(
        user_goal="先检索证据再回答",
        steps=[
            PlanStep(step_index=1, goal="先检索相关证据", preferred_action="call_tool", preferred_tool="memory_search"),
            PlanStep(step_index=2, goal="基于证据回答用户", preferred_action="direct_reply", preferred_tool=None),
        ],
    )
    decisions = [
        DecisionOutput(
            domain="novel",
            action="direct_reply",
            assistant_reply="我先直接猜一个答案。",
        ),
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_search",
            tool_args={"query": "人物关系"},
        ),
        DecisionOutput(
            domain="novel",
            action="direct_reply",
            assistant_reply="最终答案：之前讨论重点是人物关系。",
        ),
    ]
    controller, deps = build_controller(tmp_path, decisions, plans=plan)
    deps.workspace.append_long_term_entries(["之前讨论重点是人物关系"])

    result = controller.handle_user_message(SessionState(), "我们之前重点讨论了什么？")

    assert result.reply == "最终答案：之前讨论重点是人物关系。"
    assert deps.decision_backend.calls == 3
    event_types = [item["event_type"] for item in result.transcript_events]
    assert "premature_direct_reply_blocked" in event_types
    assert result.tool_trace["steps"][0]["requested_tool"] == "memory_search"


def test_non_final_terminal_tool_is_deferred_to_observation_and_next_step_runs(tmp_path: Path):
    plan = ExecutionPlanOutput(
        user_goal="先读取压缩内容，再给结论",
        steps=[
            PlanStep(step_index=1, goal="读取最近一次压缩正文", preferred_action="call_tool", preferred_tool="memory_get"),
            PlanStep(step_index=2, goal="回答是否已读取成功", preferred_action="direct_reply", preferred_tool=None),
        ],
    )
    decisions = [
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_get",
            tool_args={"target": "placeholder", "delivery_mode": "deliver"},
        ),
        DecisionOutput(
            domain="novel",
            action="direct_reply",
            assistant_reply="是，已经读取到最近一次压缩内容。",
        ),
    ]
    controller, deps = build_controller(tmp_path, decisions, plans=plan)
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

    result = controller.handle_user_message(reloaded, "先读取最近一次压缩内容，再告诉我是否读取成功")

    assert result.reply == "是，已经读取到最近一次压缩内容。"
    assert deps.decision_backend.calls == 2
    assert result.tool_trace["steps"][0]["requested_tool"] == "memory_get"
    assert result.tool_trace.get("final_tool") is None
    event_types = [item["event_type"] for item in result.transcript_events]
    assert "terminal_tool_deferred" in event_types


def test_planner_can_split_multi_step_count_question_into_observe_then_reply(tmp_path: Path):
    plan = ExecutionPlanOutput(
        user_goal="询问最近一次压缩输出字数",
        steps=[
            PlanStep(step_index=1, goal="读取最近一次压缩输出正文", preferred_action="call_tool", preferred_tool="memory_get"),
            PlanStep(step_index=2, goal="根据正文返回字数", preferred_action="direct_reply", preferred_tool=None),
        ],
    )
    decisions = [
        DecisionOutput(
            domain="novel",
            user_goal="询问最近一次压缩输出字数",
            action="call_tool",
            step_index=1,
            tool_name="memory_get",
            tool_args={"target": "placeholder"},
        ),
        DecisionOutput(
            domain="novel",
            user_goal="询问最近一次压缩输出字数",
            action="direct_reply",
            step_index=2,
            assistant_reply="最近一次压缩输出共 14 个字。",
        ),
    ]
    controller, deps = build_controller(tmp_path, decisions, plans=plan)
    session = deps.session_store.create_session()
    _append_compression_turn(
        deps,
        session,
        turn_index=1,
        user_content="请压缩第一章原文",
        assistant_content="压缩结果一共十四字",
    )
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None
    deps.decision_backend.decisions[0].tool_args["target"] = f"session_compact:{session.session_id}#compression_history:0"

    result = controller.handle_user_message(reloaded, "最近一次的压缩输出总共有多少个字？")
    assert deps.decision_backend.plan_calls == 1
    assert deps.decision_backend.calls == 2
    assert result.reply == "最近一次压缩输出共 14 个字。"
    assert result.action == "call_tool"
    assert result.tool_trace["steps"][0]["requested_tool"] == "memory_get"
    event_types = [item["event_type"] for item in result.transcript_events]
    assert "plan_created" in event_types
    assert "plan_step_completed" in event_types


def test_embedding_similarity_tool_is_available_to_model(tmp_path: Path):
    controller, _ = build_controller(
        tmp_path,
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="unused"),
    )
    session = SessionState()
    execution = controller._execute_tool(
        session,
        "embedding_similarity",
        {
            "query": "人物关系",
            "texts": ["人物关系很复杂", "今天去看天气预报"],
            "top_k": 1,
        },
    )
    assert execution.tool_name == "embedding_similarity"
    assert execution.terminal is False
    assert execution.payload["items"]
    assert execution.payload["items"][0]["text"] == "人物关系很复杂"
    assert "score=" in execution.observation_text


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


def test_tool_only_plan_triggers_final_synthesis_direct_reply(tmp_path: Path):
    plan = ExecutionPlanOutput(
        user_goal="确认最后两次压缩内容是否相同",
        steps=[
            PlanStep(step_index=1, goal="获取最后两次压缩内容的文本", preferred_action="call_tool", preferred_tool="memory_get"),
            PlanStep(step_index=2, goal="比较两次压缩内容是否为同一段文本", preferred_action="call_tool", preferred_tool="embedding_similarity"),
        ],
    )
    repeated = "出了青仙古镇，叶辰踏上苍茫大地，一路思索。"
    decisions = [
        DecisionOutput(
            domain="novel",
            user_goal="确认最后两次压缩内容是否相同",
            action="call_tool",
            step_index=1,
            tool_name="memory_get",
            tool_args={"target": "placeholder"},
        ),
        DecisionOutput(
            domain="novel",
            user_goal="确认最后两次压缩内容是否相同",
            action="call_tool",
            step_index=2,
            tool_name="embedding_similarity",
            tool_args={"query": repeated, "texts": [repeated, repeated], "top_k": 2},
        ),
        DecisionOutput(
            domain="novel",
            user_goal="确认最后两次压缩内容是否相同",
            action="direct_reply",
            assistant_reply="是，最后两次压缩内容相同。",
        ),
    ]
    controller, deps = build_controller(tmp_path, decisions, plans=plan)
    session = deps.session_store.create_session()
    _append_compression_turn(deps, session, turn_index=1, user_content="请压缩第一章原文", assistant_content=repeated)
    _append_compression_turn(deps, session, turn_index=2, user_content="请压缩第二章原文", assistant_content=repeated)
    reloaded = deps.session_store.load_session(session.session_id)
    assert reloaded is not None
    deps.decision_backend.decisions[0].tool_args["target"] = f"session_compact:{session.session_id}#compression_history:1"

    result = controller.handle_user_message(reloaded, "最后两次压缩内容是否相同？")

    assert result.reply == "是，最后两次压缩内容相同。"
    assert result.action == "call_tool"
    assert deps.decision_backend.calls == 3
    assert result.tool_trace["status"] == "final_synthesis_completed"
    event_types = [item["event_type"] for item in result.transcript_events]
    assert "final_synthesis_started" in event_types
    assert "final_synthesis_completed" in event_types


def test_final_synthesis_retries_once_when_first_attempt_does_not_direct_reply(tmp_path: Path):
    plan = ExecutionPlanOutput(
        user_goal="根据记忆给出结论",
        steps=[PlanStep(step_index=1, goal="检索历史记忆", preferred_action="call_tool", preferred_tool="memory_search")],
    )
    decisions = [
        DecisionOutput(
            domain="novel",
            user_goal="根据记忆给出结论",
            action="call_tool",
            step_index=1,
            tool_name="memory_search",
            tool_args={"query": "人物关系"},
        ),
        DecisionOutput(
            domain="novel",
            user_goal="根据记忆给出结论",
            action="call_tool",
            tool_name="memory_search",
            tool_args={"query": "人物关系"},
        ),
        DecisionOutput(
            domain="novel",
            user_goal="根据记忆给出结论",
            action="direct_reply",
            assistant_reply="最终结论：之前重点讨论的是人物关系。",
        ),
    ]
    controller, deps = build_controller(tmp_path, decisions, plans=plan)
    deps.workspace.append_long_term_entries(["之前重点讨论的是人物关系"])

    result = controller.handle_user_message(SessionState(), "我们之前重点讨论了什么？")

    assert result.reply == "最终结论：之前重点讨论的是人物关系。"
    assert deps.decision_backend.calls == 3
    event_types = [item["event_type"] for item in result.transcript_events]
    assert "final_synthesis_retry" in event_types


def test_final_synthesis_uses_structured_evidence_instead_of_full_text(tmp_path: Path):
    long_text = "完整压缩内容" * 80
    plan = ExecutionPlanOutput(
        user_goal="基于读取内容给出结论",
        steps=[PlanStep(step_index=1, goal="读取最近一次压缩内容", preferred_action="call_tool", preferred_tool="memory_get")],
    )
    decisions = [
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_get",
            tool_args={"target": "placeholder"},
        ),
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="我已经基于证据完成最终回答。"),
    ]
    controller, deps = build_controller(tmp_path, decisions, plans=plan, memory_tool_max_chars=40)
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

    result = controller.handle_user_message(reloaded, "读取最近一次压缩内容，并给出结论")

    assert result.reply == "我已经基于证据完成最终回答。"
    rendered_loop = str(deps.decision_backend.last_loop_events)
    assert "final_synthesis_evidence" in rendered_loop
    assert "preview" in rendered_loop
    assert long_text not in rendered_loop


def test_recent_content_reference_alias_is_available_to_model_and_memory_get(tmp_path: Path):
    plans = [
        ExecutionPlanOutput(
            user_goal="给出第一次压缩内容",
            steps=[PlanStep(step_index=1, goal="直接读取并交付内容", preferred_action="call_tool", preferred_tool="memory_get")],
        ),
        ExecutionPlanOutput(
            user_goal="继续引用最近内容",
            steps=[PlanStep(step_index=1, goal="基于最近内容继续回答", preferred_action="direct_reply", preferred_tool=None)],
        ),
    ]
    decisions = [
        DecisionOutput(
            domain="novel",
            action="call_tool",
            tool_name="memory_get",
            tool_args={"target": "placeholder", "delivery_mode": "deliver"},
        ),
        DecisionOutput(domain="novel", action="direct_reply", assistant_reply="我已经看到最近内容引用。"),
    ]
    controller, deps = build_controller(tmp_path, decisions, plans=plans)
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

    execution = controller._execute_tool(
        reloaded,
        "memory_get",
        {"target": "content_ref:latest", "delivery_mode": "deliver"},
    )
    assert execution.final_reply == "这是第一次压缩后的完整章节内容。"


def test_agent_config_defaults_use_32k_input_budget():
    config = AgentConfig()
    assert config.decision_input_token_budget == 32768
    assert config.decision_output_max_new_tokens == 4096
    assert config.compression_max_new_tokens == 3584
    assert config.compression_answer_reserved_tokens == 1536
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
