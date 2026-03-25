from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryWrite(BaseModel):
    daily: list[str] = Field(default_factory=list)
    long_term: list[str] = Field(default_factory=list)


class DecisionOutput(BaseModel):
    domain: Literal["novel", "out_of_scope"]
    user_goal: str = ""
    action: Literal["direct_reply", "call_tool", "reject"]
    assistant_reply: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    memory_write: MemoryWrite = Field(default_factory=MemoryWrite)


class CompressionRequest(BaseModel):
    raw_text: str
    max_new_tokens: int = 2800
    temperature: float = 0.2
    seed: int | None = None
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True


class CompressionResult(BaseModel):
    compressed_text: str
    thinking: str = ""


class BackendHealth(BaseModel):
    ok: bool
    name: str
    detail: str = ""


class AgentTurnResult(BaseModel):
    reply: str
    domain: Literal["novel", "out_of_scope"]
    action: Literal["direct_reply", "call_tool", "reject"]
    decision: dict[str, Any] = Field(default_factory=dict)
    tool_trace: dict[str, Any] = Field(default_factory=dict)
    thinking: str = ""
    memory_preview: dict[str, list[str]] = Field(default_factory=lambda: {"daily": [], "long_term": []})
    transcript_events: list[dict[str, Any]] = Field(default_factory=list)
