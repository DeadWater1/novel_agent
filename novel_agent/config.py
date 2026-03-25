from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_ROOT = Path("/home/ubuntu/code/novel_agent")
DEFAULT_WORKSPACE_ROOT = DEFAULT_ROOT / "workspace"
DEFAULT_SESSION_ROOT = DEFAULT_ROOT / "runtime" / "sessions"
DEFAULT_DECISION_MODEL_PATH = Path("/home/ubuntu/code/Qwen/Qwen3-4B")
DEFAULT_COMPRESSION_MODEL_PATH = Path("/home/ubuntu/code/qwen/output/Qwen3-4B_cot_beta_4/checkpoint-1334")


@dataclass(slots=True)
class AgentConfig:
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT
    session_root: Path = DEFAULT_SESSION_ROOT
    decision_model_path: Path = DEFAULT_DECISION_MODEL_PATH
    compression_model_path: Path = DEFAULT_COMPRESSION_MODEL_PATH
    show_debug_thinking: bool = False
    decision_max_new_tokens: int = 768
    decision_temperature: float = 0.2
    decision_top_p: float = 0.9
    decision_top_k: int = 20
    decision_enable_thinking: bool = False
    compression_max_new_tokens: int = 2800
    compression_temperature: float = 0.2
    compression_top_p: float = 0.95
    compression_top_k: int = 20
    compression_seed: int = 42
    daily_memory_lookback_days: int = 2
    recent_message_limit: int = 8
    session_summary_max_chars: int = 1200
    long_term_memory_header: str = "# Long-Term Memory\n\n"
    app_title: str = "Novel Agent V1"
    out_of_scope_reply: str = "无法回答"
    tool_registry_enabled: tuple[str, ...] = field(default_factory=lambda: ("compress_chapter",))
