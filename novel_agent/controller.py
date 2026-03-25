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


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    observation_text: str
    payload: dict[str, Any]
    tool_trace: dict[str, Any]
    terminal: bool = False
    final_reply: str = ""
    thinking: str = ""


class NovelAgentController:
    def __init__(self, deps: ControllerDependencies) -> None:
        self.deps = deps

    def handle_user_message(self, session: SessionState, user_text: str) -> AgentTurnResult:
        session.add_user_message(user_text)
        turn_index = (len(session.messages) + 1) // 2
        transcript_events = [self._event(turn_index, "user_message", role="user", content=user_text)]
        loop_events: list[dict[str, Any]] = []
        aggregated_memory = MemoryWrite()
        tool_steps: list[dict[str, Any]] = []
        used_tool = False

        forced_decision = self._build_forced_compress_decision(user_text)
        if forced_decision is not None:
            transcript_events.append(self._decision_event(turn_index, 1, forced_decision))
            self._merge_memory_write(aggregated_memory, forced_decision.memory_write)
            result = self._handle_tool_step(
                session=session,
                turn_index=turn_index,
                step_index=1,
                decision=forced_decision,
                transcript_events=transcript_events,
                tool_steps=tool_steps,
                aggregated_memory=aggregated_memory,
            )
            if result is not None:
                return result

        for step_index in range(1, self.deps.config.agent_max_loop_steps + 1):
            docs_bundle = self.deps.workspace.load_workspace_docs(session.summary)
            recent_messages = session.recent_messages(self.deps.config.recent_message_limit)

            try:
                decision = self.deps.decision_backend.decide(
                    messages=recent_messages,
                    workspace_docs=docs_bundle,
                    session_summary=session.summary,
                    loop_events=loop_events,
                )
            except ValidationError as exc:
                return self._finalize_error(
                    session,
                    f"决策结果校验失败: {exc}",
                    transcript_events=transcript_events,
                )
            except Exception as exc:
                return self._finalize_error(
                    session,
                    f"决策后端不可用: {exc}",
                    transcript_events=transcript_events,
                )

            if not isinstance(decision, DecisionOutput):
                try:
                    decision = DecisionOutput.model_validate(decision)
                except ValidationError as exc:
                    return self._finalize_error(
                        session,
                        f"决策结果校验失败: {exc}",
                        transcript_events=transcript_events,
                    )

            transcript_events.append(self._decision_event(turn_index, step_index, decision))
            self._merge_memory_write(aggregated_memory, decision.memory_write)

            if decision.domain == "out_of_scope":
                return self._finalize_success(
                    session=session,
                    reply=self.deps.config.out_of_scope_reply,
                    domain="out_of_scope",
                    action="reject",
                    decision=decision.model_dump(),
                    transcript_events=transcript_events,
                    memory_write=aggregated_memory,
                    tool_trace={"steps": tool_steps},
                )

            if decision.action == "direct_reply":
                reply = decision.assistant_reply.strip() or "当前请求暂时无法处理"
                final_action = "call_tool" if used_tool else "direct_reply"
                return self._finalize_success(
                    session=session,
                    reply=reply,
                    domain="novel",
                    action=final_action,
                    decision=decision.model_dump(),
                    transcript_events=transcript_events,
                    memory_write=aggregated_memory,
                    tool_trace={"steps": tool_steps},
                )

            if decision.action == "reject":
                return self._finalize_success(
                    session=session,
                    reply="当前请求暂时无法处理",
                    domain="novel",
                    action="reject",
                    decision=decision.model_dump(),
                    transcript_events=transcript_events,
                    memory_write=aggregated_memory,
                    tool_trace={"steps": tool_steps},
                )

            if decision.action != "call_tool":
                return self._finalize_error(
                    session,
                    "当前请求暂时无法处理",
                    decision=decision.model_dump(),
                    transcript_events=transcript_events,
                    tool_trace={"steps": tool_steps},
                )

            result = self._handle_tool_step(
                session=session,
                turn_index=turn_index,
                step_index=step_index,
                decision=decision,
                transcript_events=transcript_events,
                tool_steps=tool_steps,
                aggregated_memory=aggregated_memory,
            )
            if result is not None:
                return result

            used_tool = True
            latest_step = tool_steps[-1]
            loop_events.append(
                {
                    "step_index": step_index,
                    "event_type": "tool_call",
                    "tool_name": latest_step["requested_tool"],
                    "tool_args": latest_step.get("tool_args", {}),
                }
            )
            loop_events.append(
                {
                    "step_index": step_index,
                    "event_type": "tool_result",
                    "tool_name": latest_step["requested_tool"],
                    "observation": latest_step.get("observation", ""),
                }
            )

        return self._finalize_error(
            session,
            "当前请求暂时无法处理",
            transcript_events=transcript_events,
            tool_trace={"steps": tool_steps, "status": "loop_exhausted"},
        )

    def _handle_tool_step(
        self,
        *,
        session: SessionState,
        turn_index: int,
        step_index: int,
        decision: DecisionOutput,
        transcript_events: list[dict[str, Any]],
        tool_steps: list[dict[str, Any]],
        aggregated_memory: MemoryWrite,
    ) -> AgentTurnResult | None:
        tool_name = decision.tool_name
        if not self.deps.registry.is_registered(tool_name):
            return self._finalize_error(
                session,
                "当前请求暂时无法处理",
                decision=decision.model_dump(),
                transcript_events=transcript_events,
                tool_trace={"steps": tool_steps, "requested_tool": tool_name, "status": "blocked_invalid_tool"},
            )

        spec = self.deps.registry.get(tool_name)
        assert spec is not None
        missing = [name for name in spec.required_args if name not in decision.tool_args or not str(decision.tool_args[name]).strip()]
        if missing:
            return self._finalize_error(
                session,
                "当前请求暂时无法处理",
                decision=decision.model_dump(),
                transcript_events=transcript_events,
                tool_trace={
                    "steps": tool_steps,
                    "requested_tool": tool_name,
                    "status": "missing_args",
                    "missing_args": missing,
                },
            )

        transcript_events.append(
            self._event(
                turn_index,
                "tool_call",
                step_index=step_index,
                tool_name=tool_name,
                tool_args=dict(decision.tool_args),
            )
        )

        try:
            execution = self._execute_tool(tool_name, dict(decision.tool_args))
        except Exception as exc:
            return self._finalize_error(
                session,
                f"{tool_name} 工具不可用: {exc}",
                decision=decision.model_dump(),
                transcript_events=transcript_events,
                tool_trace={"steps": tool_steps, "requested_tool": tool_name, "status": "backend_error"},
            )

        tool_steps.append(
            {
                "requested_tool": execution.tool_name,
                "status": execution.tool_trace.get("status", "ok"),
                "tool_args": dict(decision.tool_args),
                "observation": execution.observation_text,
            }
        )
        transcript_events.append(
            self._event(
                turn_index,
                "tool_result",
                step_index=step_index,
                tool_name=execution.tool_name,
                content=execution.observation_text,
                payload={"tool_trace": execution.tool_trace, "payload": execution.payload},
                thinking=execution.thinking,
            )
        )

        if execution.terminal:
            return self._finalize_success(
                session=session,
                reply=execution.final_reply.strip() or "当前请求暂时无法处理",
                domain="novel",
                action="call_tool",
                decision=decision.model_dump(),
                transcript_events=transcript_events,
                tool_trace={"steps": tool_steps, "final_tool": execution.tool_name},
                thinking=execution.thinking,
                memory_write=aggregated_memory,
            )
        return None

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> ToolExecutionResult:
        if tool_name == "compress_chapter":
            seed_value = tool_args.get("seed", self.deps.config.compression_seed)
            request = CompressionRequest(
                raw_text=str(tool_args["raw_text"]),
                max_new_tokens=int(tool_args.get("max_new_tokens", self.deps.config.compression_max_new_tokens)),
                temperature=float(tool_args.get("temperature", self.deps.config.compression_temperature)),
                seed=int(seed_value) if seed_value not in (None, "") else None,
                top_p=float(tool_args.get("top_p", self.deps.config.compression_top_p)),
                top_k=int(tool_args.get("top_k", self.deps.config.compression_top_k)),
                enable_thinking=bool(tool_args.get("enable_thinking", True)),
            )
            tool_result = self.deps.compression_backend.compress(request)
            return ToolExecutionResult(
                tool_name=tool_name,
                observation_text=tool_result.compressed_text.strip(),
                payload={"compressed_text": tool_result.compressed_text.strip()},
                tool_trace={"requested_tool": tool_name, "status": "ok", "tool_args": request.model_dump()},
                terminal=True,
                final_reply=tool_result.compressed_text.strip(),
                thinking=tool_result.thinking,
            )

        if tool_name == "memory_search":
            query = str(tool_args["query"]).strip()
            max_results = int(tool_args.get("max_results", self.deps.config.memory_search_max_results))
            results = self.deps.workspace.memory_search(query=query, max_results=max_results)
            if results:
                lines = [f"{idx}. {item['source_id']}: {item['snippet']}" for idx, item in enumerate(results, start=1)]
                observation = "\n".join(lines)
            else:
                observation = "memory_search 未找到相关记忆。"
            return ToolExecutionResult(
                tool_name=tool_name,
                observation_text=self._truncate_memory_text(observation),
                payload={"query": query, "results": results},
                tool_trace={"requested_tool": tool_name, "status": "ok", "result_count": len(results), "query": query},
                terminal=False,
            )

        if tool_name == "memory_get":
            target = str(tool_args["target"]).strip()
            memory_doc = self.deps.workspace.memory_get(target=target)
            content = memory_doc["content"] or "(empty)"
            observation = f"{memory_doc['target']}:\n{self._truncate_memory_text(content)}"
            return ToolExecutionResult(
                tool_name=tool_name,
                observation_text=observation,
                payload=memory_doc,
                tool_trace={"requested_tool": tool_name, "status": "ok", "target": target},
                terminal=False,
            )

        raise ValueError("unsupported_tool")

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
            tool_args=self._build_default_compress_tool_args(raw_text),
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

    def _merge_memory_write(self, target: MemoryWrite, incoming: MemoryWrite | None) -> None:
        if incoming is None:
            return
        for entry in incoming.daily:
            if entry not in target.daily:
                target.daily.append(entry)
        for entry in incoming.long_term:
            if entry not in target.long_term:
                target.long_term.append(entry)

    def _truncate_memory_text(self, text: str) -> str:
        clean = text.strip()
        max_chars = self.deps.config.memory_tool_max_chars
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars].rstrip() + "..."

    def _build_default_compress_tool_args(self, raw_text: str) -> dict[str, Any]:
        tool_args = {
            "raw_text": raw_text,
            "max_new_tokens": self.deps.config.compression_max_new_tokens,
            "temperature": self.deps.config.compression_temperature,
            "top_p": self.deps.config.compression_top_p,
            "top_k": self.deps.config.compression_top_k,
            "enable_thinking": True,
        }
        if self.deps.config.compression_seed is not None:
            tool_args["seed"] = self.deps.config.compression_seed
        return tool_args

    def _decision_event(self, turn_index: int, step_index: int, decision: DecisionOutput) -> dict[str, Any]:
        return self._event(
            turn_index,
            "agent_decision",
            step_index=step_index,
            payload=decision.model_dump(),
        )

    def _event(self, turn_index: int, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {"turn_index": turn_index, "event_type": event_type}
        event.update(payload)
        return event

    def _finalize_success(
        self,
        session: SessionState,
        reply: str,
        domain: str,
        action: str,
        decision: dict[str, Any],
        transcript_events: list[dict[str, Any]],
        tool_trace: dict[str, Any] | None = None,
        thinking: str = "",
        memory_write: MemoryWrite | None = None,
    ) -> AgentTurnResult:
        session.add_assistant_message(reply)
        transcript_events.append(
            self._event((len(session.messages) // 2), "assistant_message", role="assistant", content=reply)
        )
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
            transcript_events=transcript_events,
        )

    def _finalize_error(
        self,
        session: SessionState,
        reply: str,
        transcript_events: list[dict[str, Any]],
        decision: dict[str, Any] | None = None,
        tool_trace: dict[str, Any] | None = None,
    ) -> AgentTurnResult:
        session.add_assistant_message(reply)
        transcript_events.append(
            self._event((len(session.messages) // 2), "assistant_message", role="assistant", content=reply)
        )
        return AgentTurnResult(
            reply=reply,
            domain="novel",
            action="reject",
            decision=decision or {},
            tool_trace=tool_trace or {},
            thinking="",
            memory_preview={"daily": [], "long_term": []},
            transcript_events=transcript_events,
        )
