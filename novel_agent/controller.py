from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .config import AgentConfig
from .memory import SessionState
from .registry import ToolRegistry
from .schemas import AgentTurnResult, CompressionRequest, DecisionOutput, MemoryWrite
from .workspace import WorkspaceManager


COMPRESSION_KEYWORDS = (
    "压缩",
    "压一下",
    "压一压",
    "帮我压",
    "精简",
    "浓缩",
    "缩写",
    "缩短",
    "提炼",
    "精炼",
)

NOVEL_COMPRESSION_TARGET_KEYWORDS = (
    "小说",
    "章节",
    "这章",
    "这一章",
    "一章",
    "原文",
    "正文",
    "剧情",
    "人物",
    "设定",
    "世界观",
    "片段",
    "文段",
)

CODE_LIKE_KEYWORDS = (
    "python",
    "java",
    "javascript",
    "代码",
    "函数",
    "class ",
    "def ",
    "import ",
    "```",
)


@dataclass(slots=True)
class ControllerDependencies:
    config: AgentConfig
    workspace: WorkspaceManager
    registry: ToolRegistry
    decision_backend: Any
    compression_backend: Any


class NovelAgentController:
    def __init__(self, deps: ControllerDependencies) -> None:
        self.deps = deps

    def handle_user_message(self, session: SessionState, user_text: str) -> AgentTurnResult:
        session.add_user_message(user_text)
        forced_decision = self._build_forced_compress_decision(user_text)
        if forced_decision is not None:
            result = self._route_decision(session, forced_decision)
            session.last_decision = forced_decision.model_dump()
            session.last_tool_trace = result.tool_trace
            session.last_thinking = result.thinking
            session.refresh_summary(self.deps.config.session_summary_max_chars)
            return result

        docs_bundle = self.deps.workspace.load_workspace_docs(session.summary)
        recent_messages = session.recent_messages(self.deps.config.recent_message_limit)

        try:
            decision = self.deps.decision_backend.decide(
                messages=recent_messages,
                workspace_docs=docs_bundle,
                session_summary=session.summary,
            )
        except ValidationError as exc:
            return self._finalize_error(session, f"决策结果校验失败: {exc}")
        except Exception as exc:
            return self._finalize_error(session, f"决策后端不可用: {exc}")

        if not isinstance(decision, DecisionOutput):
            try:
                decision = DecisionOutput.model_validate(decision)
            except ValidationError as exc:
                return self._finalize_error(session, f"决策结果校验失败: {exc}")

        result = self._route_decision(session, decision)
        session.last_decision = decision.model_dump()
        session.last_tool_trace = result.tool_trace
        session.last_thinking = result.thinking
        session.refresh_summary(self.deps.config.session_summary_max_chars)
        return result

    def _build_forced_compress_decision(self, user_text: str) -> DecisionOutput | None:
        raw_text = self._extract_compression_raw_text(user_text)
        if not self._should_force_compress(user_text, raw_text):
            return None
        return DecisionOutput(
            domain="novel",
            user_goal="压缩小说章节",
            action="call_tool",
            assistant_reply="",
            tool_name="compress_chapter",
            tool_args={
                "raw_text": raw_text,
                "max_new_tokens": self.deps.config.compression_max_new_tokens,
                "temperature": self.deps.config.compression_temperature,
                "seed": self.deps.config.compression_seed,
                "top_p": self.deps.config.compression_top_p,
                "top_k": self.deps.config.compression_top_k,
                "enable_thinking": True,
            },
            memory_write=MemoryWrite(daily=["用户请求压缩小说章节"], long_term=[]),
        )

    def _should_force_compress(self, user_text: str, extracted_raw_text: str) -> bool:
        text = user_text.strip()
        if not text:
            return False
        if not any(keyword in text for keyword in COMPRESSION_KEYWORDS):
            return False
        has_attached_body = extracted_raw_text.strip() != text and len(extracted_raw_text.strip()) >= 20
        if has_attached_body and any(keyword in text for keyword in NOVEL_COMPRESSION_TARGET_KEYWORDS):
            return True
        if any(keyword in text for keyword in CODE_LIKE_KEYWORDS):
            return False
        if has_attached_body:
            return True
        return len(text) >= 120 and ("\n" in text or "“" in text or "”" in text)

    def _extract_compression_raw_text(self, user_text: str) -> str:
        text = user_text.strip()
        if not text:
            return text

        if "\n\n" in text:
            head, tail = text.split("\n\n", 1)
            if any(keyword in head for keyword in COMPRESSION_KEYWORDS) and len(tail.strip()) >= 20:
                return tail.strip()

        lines = [line.rstrip() for line in text.splitlines()]
        if len(lines) >= 2 and any(keyword in lines[0] for keyword in COMPRESSION_KEYWORDS):
            tail = "\n".join(lines[1:]).strip()
            if len(tail) >= 20:
                return tail

        for sep in ("：", ":"):
            if sep in text:
                head, tail = text.split(sep, 1)
                if any(keyword in head for keyword in COMPRESSION_KEYWORDS) and len(tail.strip()) >= 20:
                    return tail.strip()

        return text

    def _route_decision(self, session: SessionState, decision: DecisionOutput) -> AgentTurnResult:
        if decision.domain == "out_of_scope":
            return self._finalize_success(
                session=session,
                reply=self.deps.config.out_of_scope_reply,
                domain="out_of_scope",
                action="reject",
                decision=decision.model_dump(),
            )

        if decision.action == "direct_reply":
            reply = decision.assistant_reply.strip() or "当前请求暂时无法处理"
            return self._finalize_success(
                session=session,
                reply=reply,
                domain="novel",
                action="direct_reply",
                decision=decision.model_dump(),
                memory_write=decision.memory_write,
            )

        if decision.action == "call_tool":
            tool_name = decision.tool_name
            if not self.deps.registry.is_registered(tool_name):
                return self._finalize_error(
                    session,
                    "当前请求暂时无法处理",
                    decision=decision.model_dump(),
                    tool_trace={"requested_tool": tool_name, "status": "blocked_invalid_tool"},
                )

            spec = self.deps.registry.get(tool_name)
            missing = [name for name in spec.required_args if name not in decision.tool_args or not str(decision.tool_args[name]).strip()]
            if missing:
                return self._finalize_error(
                    session,
                    "当前请求暂时无法处理",
                    decision=decision.model_dump(),
                    tool_trace={"requested_tool": tool_name, "status": "missing_args", "missing_args": missing},
                )

            if tool_name == "compress_chapter":
                return self._run_compress_tool(session, decision)

            return self._finalize_error(
                session,
                "当前请求暂时无法处理",
                decision=decision.model_dump(),
                tool_trace={"requested_tool": tool_name, "status": "unsupported_tool"},
            )

        return self._finalize_error(session, "当前请求暂时无法处理", decision=decision.model_dump())

    def _run_compress_tool(self, session: SessionState, decision: DecisionOutput) -> AgentTurnResult:
        args = dict(decision.tool_args)
        request = CompressionRequest(
            raw_text=str(args["raw_text"]),
            max_new_tokens=int(args.get("max_new_tokens", self.deps.config.compression_max_new_tokens)),
            temperature=float(args.get("temperature", self.deps.config.compression_temperature)),
            seed=int(args.get("seed", self.deps.config.compression_seed)),
            top_p=float(args.get("top_p", self.deps.config.compression_top_p)),
            top_k=int(args.get("top_k", self.deps.config.compression_top_k)),
            enable_thinking=bool(args.get("enable_thinking", True)),
        )
        try:
            tool_result = self.deps.compression_backend.compress(request)
        except Exception as exc:
            return self._finalize_error(
                session,
                f"压缩工具不可用: {exc}",
                decision=decision.model_dump(),
                tool_trace={"requested_tool": "compress_chapter", "status": "backend_error"},
            )

        return self._finalize_success(
            session=session,
            reply=tool_result.compressed_text.strip() or "当前请求暂时无法处理",
            domain="novel",
            action="call_tool",
            decision=decision.model_dump(),
            tool_trace={"requested_tool": "compress_chapter", "status": "ok", "tool_args": request.model_dump()},
            thinking=tool_result.thinking,
            memory_write=decision.memory_write,
        )

    def _finalize_success(
        self,
        session: SessionState,
        reply: str,
        domain: str,
        action: str,
        decision: dict[str, Any],
        tool_trace: dict[str, Any] | None = None,
        thinking: str = "",
        memory_write: MemoryWrite | None = None,
    ) -> AgentTurnResult:
        session.add_assistant_message(reply)
        preview = {"daily": [], "long_term": []}
        if memory_write is not None:
            preview = {
                "daily": list(memory_write.daily),
                "long_term": list(memory_write.long_term),
            }
            self.deps.workspace.append_daily_entries(preview["daily"])
            self.deps.workspace.append_long_term_entries(preview["long_term"])
        return AgentTurnResult(
            reply=reply,
            domain=domain,
            action=action,
            decision=decision,
            tool_trace=tool_trace or {},
            thinking=thinking,
            memory_preview=preview,
        )

    def _finalize_error(
        self,
        session: SessionState,
        reply: str,
        decision: dict[str, Any] | None = None,
        tool_trace: dict[str, Any] | None = None,
    ) -> AgentTurnResult:
        session.add_assistant_message(reply)
        return AgentTurnResult(
            reply=reply,
            domain="novel",
            action="reject",
            decision=decision or {},
            tool_trace=tool_trace or {},
            thinking="",
            memory_preview={"daily": [], "long_term": []},
        )
