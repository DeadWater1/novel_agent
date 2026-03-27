from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class MemoryWrite(BaseModel):
    daily: list[str] = Field(default_factory=list)
    long_term: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    step_index: int = 1
    goal: str = ""
    preferred_action: Literal["direct_reply", "call_tool"] = "direct_reply"
    preferred_tool: Literal["compress_chapter", "memory_search", "memory_get", "embedding_similarity"] | None = None

    @model_validator(mode="after")
    def _normalize_step_index(self) -> "PlanStep":
        self.step_index = max(int(self.step_index), 1)
        return self


class ExecutionPlanOutput(BaseModel):
    user_goal: str = ""
    steps: list[PlanStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_steps(self) -> "ExecutionPlanOutput":
        if not self.steps:
            raise ValueError("execution_plan_requires_at_least_one_step")
        for index, step in enumerate(self.steps, start=1):
            step.step_index = index
        return self


class DecisionOutput(BaseModel):
    domain: Literal["novel", "out_of_scope"]
    user_goal: str = ""
    action: Literal["direct_reply", "call_tool", "reject"]
    assistant_reply: str = ""
    step_index: int = 1
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    plan_update: ExecutionPlanOutput | None = None
    memory_write: MemoryWrite = Field(default_factory=MemoryWrite)


class DecisionReviewOutput(BaseModel):
    verdict: Literal["accept", "retry"] = "accept"
    reason: str = ""


class TimeScope(BaseModel):
    from_days_ago: int = 0
    to_days_ago: int = 0

    @model_validator(mode="after")
    def _normalize_bounds(self) -> "TimeScope":
        self.from_days_ago = max(self.from_days_ago, 0)
        self.to_days_ago = max(self.to_days_ago, 0)
        return self


class MemorySearchArgs(BaseModel):
    query: str
    search_mode: Literal["lookup", "recap"] = "lookup"
    scope: Literal["current_session", "history_sessions", "time_window"] = "current_session"
    time_scope: TimeScope | None = None
    max_results: int | None = None


class MemoryGetArgs(BaseModel):
    target: str
    delivery_mode: Literal["deliver", "observe"] = "observe"


class EmbeddingSimilarityArgs(BaseModel):
    query: str
    text: str | None = None
    texts: list[str] = Field(default_factory=list)
    top_k: int | None = None

    @model_validator(mode="after")
    def _normalize_targets(self) -> "EmbeddingSimilarityArgs":
        clean_single = (self.text or "").strip()
        clean_many = [item.strip() for item in self.texts if item and item.strip()]

        if clean_single and clean_many:
            raise ValueError("embedding_similarity accepts either text or texts, not both")
        if not clean_single and not clean_many:
            raise ValueError("embedding_similarity requires text or texts")

        self.text = clean_single or None
        self.texts = clean_many
        if self.top_k is not None:
            self.top_k = max(self.top_k, 1)
        return self


class CompactCompressionHistoryEntry(BaseModel):
    turn_index: int = 0
    user_request: str = ""
    compressed_preview: str = ""
    entities: list[str] = Field(default_factory=list)
    timestamp: str = ""
    full_content_target: str = ""
    ordinal_aliases: list[str] = Field(default_factory=list)


class CompactSummaryArtifact(BaseModel):
    session_id: str
    updated_at: str = ""
    transcript_path: str = ""
    source: Literal["llm", "rule_based"] = "rule_based"
    session_goal: str = ""
    discussion_topics: list[str] = Field(default_factory=list)
    compression_history: list[CompactCompressionHistoryEntry] = Field(default_factory=list)
    story_facts: list[str] = Field(default_factory=list)
    user_preferences: list[str] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    timeline_summary: list[str] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)


class CompressionRequest(BaseModel):
    raw_text: str
    max_new_tokens: int = 3584
    temperature: float = 0.2
    seed: int | None = None
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True


class CompressionResult(BaseModel):
    compressed_text: str
    thinking: str = ""


class CompressionLedgerEntry(BaseModel):
    session_id: str
    turn_index: int = 0
    timestamp: str = ""
    user_request_preview: str = ""
    compressed_preview: str = ""
    full_content_target: str = ""
    ordinal_aliases: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class RecentContentReference(BaseModel):
    alias: str = ""
    resolved_target: str = ""
    source_path: str = ""
    preview: str = ""
    content_length: int = 0
    created_turn_index: int = 0


class BackendHealth(BaseModel):
    ok: bool
    name: str
    detail: str = ""


class ContextReport(BaseModel):
    estimated_tokens: int = 0
    pruning_applied: bool = False
    compaction_applied: bool = False
    compaction_source: str = ""
    memory_flush_applied: bool = False
    recall_targets: list[str] = Field(default_factory=list)
    context_blocks: list[str] = Field(default_factory=list)
    review_triggered: bool = False
    review_verdict: str = ""
    review_reason: str = ""
    memory_flush_daily: list[str] = Field(default_factory=list)
    memory_flush_long_term: list[str] = Field(default_factory=list)


class ContextBuildResult(BaseModel):
    messages: list[dict[str, str]] = Field(default_factory=list)
    session_summary: str = ""
    compacted_session_context: str = ""
    recalled_memory: str = ""
    recent_content_references: str = ""
    workspace_docs: str = ""
    loop_events: list[dict[str, Any]] = Field(default_factory=list)
    context_report: ContextReport = Field(default_factory=ContextReport)


class AgentTurnResult(BaseModel):
    reply: str
    domain: Literal["novel", "out_of_scope"]
    action: Literal["direct_reply", "call_tool", "reject"]
    decision: dict[str, Any] = Field(default_factory=dict)
    tool_trace: dict[str, Any] = Field(default_factory=dict)
    thinking: str = ""
    memory_preview: dict[str, list[str]] = Field(default_factory=lambda: {"daily": [], "long_term": []})
    transcript_events: list[dict[str, Any]] = Field(default_factory=list)
    context_report: dict[str, Any] = Field(default_factory=dict)
