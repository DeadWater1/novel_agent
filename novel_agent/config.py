from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_ROOT = Path("/home/ubuntu/code/novel_agent")
DEFAULT_WORKSPACE_ROOT = DEFAULT_ROOT / "workspace"
DEFAULT_RUNTIME_ROOT = DEFAULT_ROOT / "runtime"
DEFAULT_SESSION_ROOT = DEFAULT_ROOT / "runtime" / "sessions"
DEFAULT_TRANSCRIPT_ROOT = DEFAULT_RUNTIME_ROOT / "transcripts"
DEFAULT_COMPACTION_ROOT = DEFAULT_RUNTIME_ROOT / "compactions"
DEFAULT_DECISION_MODEL_PATH = Path("/home/ubuntu/code/qwen/Qwen/Qwen3-14B")
DEFAULT_COMPRESSION_MODEL_PATH = Path("/home/ubuntu/code/qwen/output/Qwen3-4B_cot_beta_4/checkpoint-1334")
DEFAULT_SUMMARY_MODEL_PATH = DEFAULT_DECISION_MODEL_PATH
DEFAULT_DECISION_OUTPUT_MAX_NEW_TOKENS = 4096
DEFAULT_CONTEXT_MEMORY_FLUSH_SOFT_THRESHOLD = 28672


@dataclass(slots=True)
class AgentConfig:
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT
    session_root: Path = DEFAULT_SESSION_ROOT
    transcript_root: Path = DEFAULT_TRANSCRIPT_ROOT
    compaction_root: Path = DEFAULT_COMPACTION_ROOT
    decision_model_path: Path = DEFAULT_DECISION_MODEL_PATH
    summary_model_path: Path = DEFAULT_SUMMARY_MODEL_PATH
    compression_model_path: Path = DEFAULT_COMPRESSION_MODEL_PATH
    show_debug_thinking: bool = False
    decision_input_token_budget: int = 32768
    decision_output_max_new_tokens: int = DEFAULT_DECISION_OUTPUT_MAX_NEW_TOKENS
    decision_max_new_tokens: int = DEFAULT_DECISION_OUTPUT_MAX_NEW_TOKENS
    decision_temperature: float = 0.7
    decision_top_p: float = 0.9
    decision_top_k: int = 20
    decision_enable_thinking: bool = False
    summary_max_new_tokens: int = 1800
    summary_temperature: float = 0.1
    summary_top_p: float = 0.9
    summary_top_k: int = 20
    summary_enable_thinking: bool = False
    compression_max_new_tokens: int = 2800
    compression_temperature: float = 0.2
    compression_top_p: float = 0.95
    compression_top_k: int = 20
    compression_seed: int | None = None
    daily_memory_lookback_days: int = 2
    recent_message_limit: int = 8
    decision_reflection_enabled: bool = True
    decision_reflection_max_passes: int = 1
    context_recent_raw_messages: int = 8
    context_pruning_enabled: bool = True
    context_pruning_keep_recent_tool_results: int = 2
    context_pruning_soft_budget: int = 30720
    context_pruning_target_tokens: int = 28672
    context_micro_preview_chars: int = 160
    context_micro_keep_recent_messages: int = 8
    context_auto_compact_token_threshold: int = 31744
    context_auto_compact_min_turns: int = 6
    memory_flush_enabled: bool = True
    context_memory_flush_soft_threshold: int = DEFAULT_CONTEXT_MEMORY_FLUSH_SOFT_THRESHOLD
    memory_flush_soft_threshold_tokens: int = DEFAULT_CONTEXT_MEMORY_FLUSH_SOFT_THRESHOLD
    session_summary_max_chars: int = 1200
    agent_max_loop_steps: int = 4
    memory_search_max_results: int = 5
    archive_session_search_limit: int = 8
    archive_session_search_messages_per_session: int = 24
    memory_tool_max_chars: int = 4000
    long_term_memory_header: str = "# Long-Term Memory\n\n"
    app_title: str = "Novel Agent V3"
    out_of_scope_reply: str = "无法回答"
    enable_heartbeat: bool = True
    idle_heartbeat_interval_seconds: int = 180
    daily_memory_min_turns: int = 1
    long_term_repeat_threshold: int = 2
    tool_registry_enabled: tuple[str, ...] = field(
        default_factory=lambda: ("compress_chapter", "memory_search", "memory_get")
    )

    def __post_init__(self) -> None:
        if (
            self.decision_output_max_new_tokens == DEFAULT_DECISION_OUTPUT_MAX_NEW_TOKENS
            and self.decision_max_new_tokens != DEFAULT_DECISION_OUTPUT_MAX_NEW_TOKENS
        ):
            self.decision_output_max_new_tokens = max(self.decision_max_new_tokens, 1)
        self.decision_output_max_new_tokens = max(self.decision_output_max_new_tokens, 1)
        self.decision_max_new_tokens = self.decision_output_max_new_tokens

        if (
            self.context_memory_flush_soft_threshold == DEFAULT_CONTEXT_MEMORY_FLUSH_SOFT_THRESHOLD
            and self.memory_flush_soft_threshold_tokens != DEFAULT_CONTEXT_MEMORY_FLUSH_SOFT_THRESHOLD
        ):
            self.context_memory_flush_soft_threshold = max(self.memory_flush_soft_threshold_tokens, 1)

        self.decision_input_token_budget = max(self.decision_input_token_budget, 1)
        self.context_pruning_soft_budget = min(
            max(self.context_pruning_soft_budget, 1),
            self.decision_input_token_budget,
        )
        self.context_pruning_target_tokens = min(
            max(self.context_pruning_target_tokens, 1),
            self.context_pruning_soft_budget,
        )
        self.context_auto_compact_token_threshold = min(
            max(self.context_auto_compact_token_threshold, 1),
            self.decision_input_token_budget,
        )
        self.context_memory_flush_soft_threshold = min(
            max(self.context_memory_flush_soft_threshold, 1),
            self.context_pruning_target_tokens,
        )
        self.memory_flush_soft_threshold_tokens = self.context_memory_flush_soft_threshold
