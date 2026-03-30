from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal


DEFAULT_ROOT = Path("/home/ubuntu/code/novel_agent")
DEFAULT_WORKSPACE_ROOT = DEFAULT_ROOT / "workspace"
DEFAULT_RUNTIME_ROOT = DEFAULT_ROOT / "runtime"
DEFAULT_SESSION_ROOT = DEFAULT_ROOT / "runtime" / "sessions"
DEFAULT_TRANSCRIPT_ROOT = DEFAULT_RUNTIME_ROOT / "transcripts"
DEFAULT_COMPACTION_ROOT = DEFAULT_RUNTIME_ROOT / "compactions"
DEFAULT_EMBEDDING_INDEX_ROOT = DEFAULT_RUNTIME_ROOT / "embeddings"
DEFAULT_STRUCTURED_MEMORY_ROOT = DEFAULT_RUNTIME_ROOT / "structured_memory"
DEFAULT_DECISION_MODEL_PATH = Path("/home/ubuntu/code/qwen/Qwen/Qwen3-14B")
DEFAULT_EMBEDDING_MODEL_PATH = Path("/home/ubuntu/code/qwen/Qwen/Qwen3-Embedding-0.6B")
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
    embedding_index_root: Path = DEFAULT_EMBEDDING_INDEX_ROOT
    structured_memory_root: Path = DEFAULT_STRUCTURED_MEMORY_ROOT
    decision_model_path: Path = DEFAULT_DECISION_MODEL_PATH
    embedding_model_path: Path = DEFAULT_EMBEDDING_MODEL_PATH
    summary_model_path: Path = DEFAULT_SUMMARY_MODEL_PATH
    compression_model_path: Path = DEFAULT_COMPRESSION_MODEL_PATH
    generation_backend: Literal["local", "vllm"] = "local"
    decision_review_mode: Literal["always", "risk_gated", "disabled"] = "risk_gated"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.5
    vllm_max_model_len: int = 32768
    vllm_dtype: str = "auto"
    vllm_trust_remote_code: bool = True
    vllm_enforce_eager: bool = True
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
    compression_max_new_tokens: int = 3584
    compression_answer_reserved_tokens: int = 1536
    compression_temperature: float = 0.2
    compression_top_p: float = 0.95
    compression_top_k: int = 20
    compression_enable_thinking: bool = True
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
    agent_max_loop_steps: int = 16
    memory_search_max_results: int = 5
    archive_session_search_limit: int = 8
    archive_session_search_messages_per_session: int = 24
    memory_tool_max_chars: int = 4000
    embedding_max_length: int = 32768
    embedding_batch_size: int = 16
    embedding_index_enabled: bool = True
    embedding_index_batch_size: int = 32
    embedding_index_max_sessions_per_idle_run: int = 8
    long_term_memory_header: str = "# Long-Term Memory\n\n"
    app_title: str = "Novel Agent V5"
    out_of_scope_reply: str = "无法回答"
    enable_heartbeat: bool = True
    idle_heartbeat_interval_seconds: int = 180
    daily_memory_min_turns: int = 1
    long_term_repeat_threshold: int = 2
    tool_registry_enabled: tuple[str, ...] = field(
        default_factory=lambda: ("compress_chapter", "memory_search", "memory_get", "embedding_similarity")
    )

    @classmethod
    def from_env(cls) -> "AgentConfig":
        kwargs: dict[str, object] = {}
        generation_backend = os.getenv("NOVEL_AGENT_GENERATION_BACKEND", "").strip().lower()
        if generation_backend:
            kwargs["generation_backend"] = generation_backend

        review_mode = (
            os.getenv("NOVEL_AGENT_REVIEW_MODE", "").strip().lower()
            or os.getenv("NOVEL_AGENT_DECISION_REVIEW_MODE", "").strip().lower()
        )
        if review_mode:
            kwargs["decision_review_mode"] = review_mode

        for env_name, key in (
            ("NOVEL_AGENT_VLLM_TENSOR_PARALLEL_SIZE", "vllm_tensor_parallel_size"),
            ("NOVEL_AGENT_VLLM_MAX_MODEL_LEN", "vllm_max_model_len"),
        ):
            value = os.getenv(env_name, "").strip()
            if not value:
                continue
            try:
                kwargs[key] = int(value)
            except ValueError:
                pass

        for env_name, key in (
            ("NOVEL_AGENT_VLLM_GPU_MEMORY_UTILIZATION", "vllm_gpu_memory_utilization"),
        ):
            value = os.getenv(env_name, "").strip()
            if not value:
                continue
            try:
                kwargs[key] = float(value)
            except ValueError:
                pass

        dtype = os.getenv("NOVEL_AGENT_VLLM_DTYPE", "").strip()
        if dtype:
            kwargs["vllm_dtype"] = dtype

        for env_name, key in (
            ("NOVEL_AGENT_VLLM_TRUST_REMOTE_CODE", "vllm_trust_remote_code"),
            ("NOVEL_AGENT_VLLM_ENFORCE_EAGER", "vllm_enforce_eager"),
        ):
            value = os.getenv(env_name, "").strip().lower()
            if value in {"1", "true", "yes", "on"}:
                kwargs[key] = True
            elif value in {"0", "false", "no", "off"}:
                kwargs[key] = False

        return cls(**kwargs)

    def __post_init__(self) -> None:
        self.generation_backend = str(self.generation_backend or "local").strip().lower()
        if self.generation_backend not in {"local", "vllm"}:
            self.generation_backend = "local"

        self.decision_review_mode = str(self.decision_review_mode or "risk_gated").strip().lower()
        if self.decision_review_mode not in {"always", "risk_gated", "disabled"}:
            self.decision_review_mode = "risk_gated"

        self.vllm_tensor_parallel_size = max(int(self.vllm_tensor_parallel_size), 1)
        self.vllm_gpu_memory_utilization = min(max(float(self.vllm_gpu_memory_utilization), 0.1), 1.0)
        self.vllm_max_model_len = max(int(self.vllm_max_model_len), 1)
        self.vllm_dtype = str(self.vllm_dtype or "auto").strip() or "auto"
        self.vllm_trust_remote_code = bool(self.vllm_trust_remote_code)
        self.vllm_enforce_eager = bool(self.vllm_enforce_eager)

        if not self.decision_reflection_enabled:
            self.decision_review_mode = "disabled"

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
        self.compression_max_new_tokens = max(self.compression_max_new_tokens, 1)
        self.compression_answer_reserved_tokens = min(
            max(self.compression_answer_reserved_tokens, 1),
            self.compression_max_new_tokens,
        )
