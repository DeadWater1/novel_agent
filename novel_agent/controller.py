from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Any

from pydantic import ValidationError

from .compaction import ContextCompactionManager, render_compact_summary_text
from .config import AgentConfig
from .context_engine import ContextEngine, ContextEngineDependencies
from .embedding_index import EmbeddingIndexItem
from .memory import SessionState, SessionStore
from .registry import ToolRegistry
from .schemas import (
    AgentTurnResult,
    CompressionRequest,
    ContextReport,
    DecisionOutput,
    DecisionReviewOutput,
    EmbeddingSimilarityArgs,
    ExecutionPlanOutput,
    MemoryGetArgs,
    MemorySearchArgs,
    MemoryWrite,
    PlanStep,
    RecentContentReference,
    TimeScope,
)
from .search_utils import extract_snippet, hybrid_search_score, hybrid_search_scores, mmr_rerank
from .session_meta import SessionMetaStore
from .workspace import WorkspaceManager


SESSION_SEARCH_CANDIDATE_MULTIPLIER = 4


@dataclass(slots=True)
class ControllerDependencies:
    config: AgentConfig
    workspace: WorkspaceManager
    registry: ToolRegistry
    decision_backend: Any
    compression_backend: Any
    embedding_backend: Any
    embedding_index_manager: Any | None = None
    session_store: SessionStore | None = None
    meta_store: SessionMetaStore | None = None
    compaction_manager: ContextCompactionManager | None = None


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    observation_text: str
    payload: dict[str, Any]
    tool_trace: dict[str, Any]
    terminal: bool = False
    final_reply: str = ""
    thinking: str = ""


@dataclass(slots=True)
class TurnPlanState:
    plan: ExecutionPlanOutput
    current_position: int = 0
    completed_steps: list[PlanStep] | None = None
    replan_used: bool = False

    def __post_init__(self) -> None:
        if self.completed_steps is None:
            self.completed_steps = []


class NovelAgentController:
    def __init__(self, deps: ControllerDependencies) -> None:
        self.deps = deps
        self.context_engine = ContextEngine(
            ContextEngineDependencies(
                config=deps.config,
                workspace=deps.workspace,
                decision_backend=deps.decision_backend,
                session_store=deps.session_store,
                meta_store=deps.meta_store,
                compaction_manager=deps.compaction_manager,
                search_memory_fn=self._search_memory_sources,
            )
        )

    def handle_user_message(self, session: SessionState, user_text: str) -> AgentTurnResult:
        session.add_user_message(user_text)
        session.refresh_summary(self.deps.config.session_summary_max_chars)
        turn_index = (len(session.messages) + 1) // 2
        transcript_events = [self._event(turn_index, "user_message", role="user", content=user_text)]
        loop_events: list[dict[str, Any]] = []
        aggregated_memory = MemoryWrite()
        tool_steps: list[dict[str, Any]] = []
        used_tool = False
        review_passes = 0
        context_report = ContextReport()
        planning_context = self.context_engine.build_turn_context(
            session=session,
            user_text=user_text,
            loop_events=[],
        )
        context_report = self._merge_context_report(context_report, planning_context.context_report)

        try:
            plan = self.deps.decision_backend.plan_turn(
                user_text=user_text,
                messages=planning_context.messages,
                workspace_docs=planning_context.workspace_docs,
                session_summary=planning_context.session_summary,
                compacted_session_context=planning_context.compacted_session_context,
                recent_content_references=planning_context.recent_content_references,
                loop_events=planning_context.loop_events,
            )
        except ValidationError as exc:
            return self._finalize_error(
                session,
                f"计划结果校验失败: {exc}",
                transcript_events=transcript_events,
                context_report=context_report,
            )
        except Exception as exc:
            return self._finalize_error(
                session,
                f"计划后端不可用: {exc}",
                transcript_events=transcript_events,
                context_report=context_report,
            )

        if not isinstance(plan, ExecutionPlanOutput):
            try:
                plan = ExecutionPlanOutput.model_validate(plan)
            except ValidationError as exc:
                return self._finalize_error(
                    session,
                    f"计划结果校验失败: {exc}",
                    transcript_events=transcript_events,
                    context_report=context_report,
                )

        plan_state = TurnPlanState(plan=plan)
        plan_created_event = self._plan_event(turn_index, "plan_created", plan)
        transcript_events.append(plan_created_event)
        loop_events.append(
            {
                "step_index": 0,
                "event_type": "plan_created",
                "payload": plan.model_dump(),
            }
        )

        for loop_step_index in range(1, self.deps.config.agent_max_loop_steps + 1):
            current_plan_step = self._current_plan_step(plan_state)
            if current_plan_step is None:
                return self._finalize_error(
                    session,
                    "当前请求暂时无法处理",
                    transcript_events=transcript_events,
                    tool_trace={"steps": tool_steps, "status": "plan_exhausted"},
                    context_report=context_report,
                )

            context_bundle = self.context_engine.build_turn_context(
                session=session,
                user_text=user_text,
                loop_events=loop_events,
                execution_plan=plan_state.plan.model_dump(),
                current_step=current_plan_step.model_dump(),
                completed_steps=[step.model_dump() for step in plan_state.completed_steps or []],
            )
            context_report = self._merge_context_report(context_report, context_bundle.context_report)

            try:
                decision = self.deps.decision_backend.decide(
                    messages=context_bundle.messages,
                    workspace_docs=context_bundle.workspace_docs,
                    session_summary=context_bundle.session_summary,
                    compacted_session_context=context_bundle.compacted_session_context,
                    recent_content_references=context_bundle.recent_content_references,
                    loop_events=context_bundle.loop_events,
                    execution_plan=plan_state.plan.model_dump(),
                    current_step=current_plan_step.model_dump(),
                    completed_steps=[step.model_dump() for step in plan_state.completed_steps or []],
                )
            except ValidationError as exc:
                return self._finalize_error(
                    session,
                    f"决策结果校验失败: {exc}",
                    transcript_events=transcript_events,
                    context_report=context_report,
                )
            except Exception as exc:
                return self._finalize_error(
                    session,
                    f"决策后端不可用: {exc}",
                    transcript_events=transcript_events,
                    context_report=context_report,
                )

            if not isinstance(decision, DecisionOutput):
                try:
                    decision = DecisionOutput.model_validate(decision)
                except ValidationError as exc:
                    return self._finalize_error(
                        session,
                        f"决策结果校验失败: {exc}",
                        transcript_events=transcript_events,
                        context_report=context_report,
                    )

            transcript_events.append(self._decision_event(turn_index, loop_step_index, decision))
            self._merge_memory_write(aggregated_memory, decision.memory_write)

            if decision.plan_update is not None:
                if not plan_state.replan_used:
                    plan_state.plan = decision.plan_update
                    plan_state.current_position = 0
                    plan_state.replan_used = True
                    current_plan_step = self._current_plan_step(plan_state)
                    transcript_events.append(self._plan_event(turn_index, "plan_updated", plan_state.plan))
                    loop_events.append(
                        {
                            "step_index": loop_step_index,
                            "event_type": "plan_updated",
                            "payload": plan_state.plan.model_dump(),
                        }
                    )
                else:
                    loop_events.append(
                        {
                            "step_index": loop_step_index,
                            "event_type": "plan_update_ignored",
                            "payload": decision.plan_update.model_dump(),
                        }
                    )

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
                    context_report=context_report,
                )

            if decision.action == "direct_reply":
                if self._should_review_direct_reply(review_passes=review_passes):
                    review = self._review_direct_reply(
                        user_text=user_text,
                        decision=decision,
                        context_bundle=context_bundle,
                        plan_state=plan_state,
                        current_step=current_plan_step,
                    )
                    context_report.review_triggered = True
                    context_report.review_verdict = review.verdict
                    context_report.review_reason = review.reason
                    transcript_events.append(
                        self._event(
                            turn_index,
                            "decision_review",
                            step_index=loop_step_index,
                            payload=review.model_dump(),
                        )
                    )
                    if review.verdict == "retry":
                        review_passes += 1
                        loop_events.append(
                            {
                                "step_index": loop_step_index,
                                "event_type": "decision_review",
                                "verdict": review.verdict,
                                "reason": review.reason,
                            }
                        )
                        continue
                plan_state.completed_steps.append(current_plan_step)
                transcript_events.append(self._plan_step_completed_event(turn_index, current_plan_step))
                loop_events.append(
                    {
                        "step_index": current_plan_step.step_index,
                        "event_type": "plan_step_completed",
                        "goal": current_plan_step.goal,
                    }
                )
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
                    context_report=context_report,
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
                    context_report=context_report,
                )

            if decision.action != "call_tool":
                return self._finalize_error(
                    session,
                    "当前请求暂时无法处理",
                    decision=decision.model_dump(),
                    transcript_events=transcript_events,
                    tool_trace={"steps": tool_steps},
                    context_report=context_report,
                )

            result = self._handle_tool_step(
                session=session,
                turn_index=turn_index,
                step_index=loop_step_index,
                decision=decision,
                transcript_events=transcript_events,
                tool_steps=tool_steps,
                aggregated_memory=aggregated_memory,
                context_report=context_report,
            )
            if result is not None:
                completed_event = self._plan_step_completed_event(turn_index, current_plan_step)
                if result.transcript_events is transcript_events:
                    result.transcript_events.append(completed_event)
                else:
                    transcript_events.append(completed_event)
                    result.transcript_events.append(completed_event)
                return result

            used_tool = True
            plan_state.completed_steps.append(current_plan_step)
            plan_state.current_position += 1
            transcript_events.append(self._plan_step_completed_event(turn_index, current_plan_step))
            latest_step = tool_steps[-1]
            loop_events.append(
                {
                    "step_index": loop_step_index,
                    "event_type": "tool_call",
                    "tool_name": latest_step["requested_tool"],
                    "tool_args": latest_step.get("tool_args", {}),
                }
            )
            loop_events.append(
                {
                    "step_index": loop_step_index,
                    "event_type": "tool_result",
                    "tool_name": latest_step["requested_tool"],
                    "observation": latest_step.get("observation", ""),
                }
            )
            loop_events.append(
                {
                    "step_index": current_plan_step.step_index,
                    "event_type": "plan_step_completed",
                    "goal": current_plan_step.goal,
                }
            )

        return self._finalize_error(
            session,
            "当前请求暂时无法处理",
            transcript_events=transcript_events,
            tool_trace={"steps": tool_steps, "status": "loop_exhausted"},
            context_report=context_report,
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
        context_report: ContextReport,
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
            execution = self._execute_tool(session, tool_name, dict(decision.tool_args))
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
                context_report=context_report,
            )
        return None

    def _execute_tool(self, session: SessionState, tool_name: str, tool_args: dict[str, Any]) -> ToolExecutionResult:
        if tool_name == "compress_chapter":
            request = CompressionRequest(
                raw_text=str(tool_args["raw_text"]),
                max_new_tokens=self.deps.config.compression_max_new_tokens,
                temperature=self.deps.config.compression_temperature,
                seed=self.deps.config.compression_seed,
                top_p=self.deps.config.compression_top_p,
                top_k=self.deps.config.compression_top_k,
                enable_thinking=self.deps.config.compression_enable_thinking,
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
            search_args = MemorySearchArgs.model_validate(tool_args)
            max_results = int(search_args.max_results or self.deps.config.memory_search_max_results)
            results = self._search_memory_sources(
                session=session,
                query=search_args.query.strip(),
                max_results=max_results,
                search_mode=search_args.search_mode,
                scope=search_args.scope,
                time_scope=search_args.time_scope,
            )
            if results:
                observation = self._format_memory_search_results(results, search_mode=search_args.search_mode)
            else:
                observation = "memory_search 未找到相关记忆。"
            return ToolExecutionResult(
                tool_name=tool_name,
                observation_text=self._truncate_memory_text(observation),
                payload={"search_args": search_args.model_dump(), "results": results},
                tool_trace={
                    "requested_tool": tool_name,
                    "status": "ok",
                    "result_count": len(results),
                    "query": search_args.query,
                    "search_mode": search_args.search_mode,
                    "scope": search_args.scope,
                },
                terminal=False,
            )

        if tool_name == "memory_get":
            memory_args = MemoryGetArgs.model_validate(tool_args)
            memory_doc = self._memory_get(session, target=memory_args.target)
            content = str(memory_doc.get("content") or "").strip()
            resolved_target = str(memory_doc.get("resolved_target") or memory_doc["target"])
            source_path = str(memory_doc.get("source_path") or "")
            observation_preview = self._memory_observation_preview(content or "(empty)")
            observation = (
                f"target={resolved_target} source={source_path or '(empty)'} length={len(content)}\n"
                f"preview:\n{observation_preview}"
            )
            self._remember_recent_content_reference(session, memory_doc)
            return ToolExecutionResult(
                tool_name=tool_name,
                observation_text=observation,
                payload={**memory_doc, "delivery_mode": memory_args.delivery_mode},
                tool_trace={
                    "requested_tool": tool_name,
                    "status": "ok",
                    "target": memory_args.target,
                    "resolved_target": resolved_target,
                    "delivery_mode": memory_args.delivery_mode,
                },
                terminal=memory_args.delivery_mode == "deliver",
                final_reply=(content or "未找到对应内容。") if memory_args.delivery_mode == "deliver" else "",
            )

        if tool_name == "embedding_similarity":
            similarity_args = EmbeddingSimilarityArgs.model_validate(tool_args)
            candidates = [similarity_args.text] if similarity_args.text else list(similarity_args.texts)
            scores = self.deps.embedding_backend.similarity_batch(similarity_args.query, candidates)
            ranked = [
                {
                    "index": index + 1,
                    "score": float(score),
                    "text": text,
                    "preview": extract_snippet(text, similarity_args.query, max_chars=220),
                }
                for index, (text, score) in enumerate(zip(candidates, scores))
            ]
            ranked.sort(key=lambda item: item["score"], reverse=True)
            if similarity_args.top_k is not None:
                ranked = ranked[: similarity_args.top_k]
            observation_lines = [
                f"{index}. score={item['score']:.4f} length={len(item['text'])}\npreview: {item['preview']}"
                for index, item in enumerate(ranked, start=1)
            ]
            observation = "\n".join(observation_lines) if observation_lines else "embedding_similarity 无可比较文本。"
            return ToolExecutionResult(
                tool_name=tool_name,
                observation_text=self._truncate_memory_text(observation),
                payload={
                    "query": similarity_args.query,
                    "items": ranked,
                },
                tool_trace={
                    "requested_tool": tool_name,
                    "status": "ok",
                    "candidate_count": len(candidates),
                },
                terminal=False,
            )

        raise ValueError("unsupported_tool")

    def _should_review_direct_reply(self, *, review_passes: int) -> bool:
        if not self.deps.config.decision_reflection_enabled:
            return False
        return review_passes < max(self.deps.config.decision_reflection_max_passes, 0)

    def _review_direct_reply(
        self,
        *,
        user_text: str,
        decision: DecisionOutput,
        context_bundle: Any,
        plan_state: TurnPlanState,
        current_step: PlanStep,
    ) -> DecisionReviewOutput:
        reviewer = getattr(self.deps.decision_backend, "review_decision", None)
        if not callable(reviewer):
            return DecisionReviewOutput(verdict="accept", reason="review_backend_unavailable")
        try:
            review = reviewer(
                messages=context_bundle.messages,
                workspace_docs=context_bundle.workspace_docs,
                session_summary=context_bundle.session_summary,
                compacted_session_context=context_bundle.compacted_session_context,
                recent_content_references=context_bundle.recent_content_references,
                loop_events=context_bundle.loop_events,
                execution_plan=plan_state.plan.model_dump(),
                current_step=current_step.model_dump(),
                completed_steps=[step.model_dump() for step in plan_state.completed_steps or []],
                decision=decision.model_dump(),
                user_text=user_text,
            )
        except Exception as exc:
            return DecisionReviewOutput(verdict="accept", reason=f"review_error:{exc}")
        if isinstance(review, DecisionReviewOutput):
            return review
        try:
            return DecisionReviewOutput.model_validate(review)
        except ValidationError:
            return DecisionReviewOutput(verdict="accept", reason="review_validation_failed")

    def _current_plan_step(self, plan_state: TurnPlanState) -> PlanStep | None:
        if plan_state.current_position < 0:
            return None
        if plan_state.current_position >= len(plan_state.plan.steps):
            return None
        return plan_state.plan.steps[plan_state.current_position]

    def _compaction_manager(self) -> ContextCompactionManager | None:
        return self.deps.compaction_manager

    def _search_memory_sources(
        self,
        *,
        session: SessionState,
        query: str,
        max_results: int,
        search_mode: str = "lookup",
        scope: str = "current_session",
        time_scope: TimeScope | None = None,
        exclude_latest_user_message: bool = True,
    ) -> list[dict[str, Any]]:
        if search_mode == "recap":
            return self._search_recap_sources(
                session=session,
                query=query,
                max_results=max_results,
                scope=scope,
                time_scope=time_scope,
            )
        return self._search_lookup_sources(
            session=session,
            query=query,
            max_results=max_results,
            scope=scope,
            time_scope=time_scope,
            exclude_latest_user_message=exclude_latest_user_message,
        )

    def _search_lookup_sources(
        self,
        *,
        session: SessionState,
        query: str,
        max_results: int,
        scope: str,
        time_scope: TimeScope | None,
        exclude_latest_user_message: bool,
    ) -> list[dict[str, Any]]:
        query_embedding = self._build_query_embedding(query)
        if scope == "current_session":
            return self._merge_ranked_lookup_tiers(
                [
                    self._search_session_context(
                        session=session,
                        query=query,
                        max_results=max_results * SESSION_SEARCH_CANDIDATE_MULTIPLIER,
                        exclude_latest_user_message=exclude_latest_user_message,
                    ),
                    self._search_current_compact_context(
                        session=session,
                        query=query,
                        max_results=max_results * SESSION_SEARCH_CANDIDATE_MULTIPLIER,
                        query_embedding=query_embedding,
                    ),
                    self.deps.workspace.memory_search(
                        query=query,
                        max_results=max_results * SESSION_SEARCH_CANDIDATE_MULTIPLIER,
                    ),
                ],
                max_results=max_results,
            )
        return self._merge_ranked_lookup_tiers(
            [
                self._search_compact_history_context(
                    session=session,
                    query=query,
                    max_results=max_results * SESSION_SEARCH_CANDIDATE_MULTIPLIER,
                    scope=scope,
                    time_scope=time_scope,
                    query_embedding=query_embedding,
                ),
                self._search_archived_session_context(
                    session=session,
                    query=query,
                    max_results=max_results * SESSION_SEARCH_CANDIDATE_MULTIPLIER,
                    scope=scope,
                    time_scope=time_scope,
                    query_embedding=query_embedding,
                ),
                self.deps.workspace.memory_search(
                    query=query,
                    max_results=max_results * SESSION_SEARCH_CANDIDATE_MULTIPLIER,
                ),
            ],
            max_results=max_results,
        )

    def _search_recap_sources(
        self,
        *,
        session: SessionState,
        query: str,
        max_results: int,
        scope: str,
        time_scope: TimeScope | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        manager = self._compaction_manager()
        if manager is None:
            return []

        if scope == "current_session":
            artifact = manager.load_or_build_artifact(session)
            results.append(self._build_recap_result_from_artifact(query=query, artifact=artifact))
            return results[:max_results]

        if scope == "time_window" and time_scope is not None:
            window_summary = manager.build_time_window_summary(
                from_days_ago=time_scope.from_days_ago,
                to_days_ago=time_scope.to_days_ago,
            )
            results.append(
                {
                    "source_id": window_summary["target"],
                    "source_kind": "time_window_summary",
                    "source_path": window_summary["time_range"],
                    "target": window_summary["target"],
                    "score": max(
                        hybrid_search_score(
                            query,
                            window_summary["content"],
                            embedding_backend=self.deps.embedding_backend,
                        ),
                        0.5,
                    ),
                    "summary_preview": window_summary["summary_preview"],
                    "topics": window_summary["topics"],
                    "time_range": window_summary["time_range"],
                    "session_id": "time_window",
                    "snippet": window_summary["summary_preview"],
                }
            )
            for artifact in manager.artifacts_for_time_window(time_scope.from_days_ago, time_scope.to_days_ago)[:max_results]:
                results.append(self._build_recap_result_from_artifact(query=query, artifact=artifact))
            return self._rerank_search_results(results, max_results=max_results)

        for artifact in self._history_compaction_artifacts(session, scope=scope, time_scope=time_scope):
            results.append(self._build_recap_result_from_artifact(query=query, artifact=artifact))
        return self._rerank_search_results(results, max_results=max_results)

    def _build_recap_result_from_artifact(self, *, query: str, artifact: Any) -> dict[str, Any]:
        manager = self._compaction_manager()
        assert manager is not None
        preview = manager.search_chunks(artifact)[0]["summary_preview"]
        return {
            "source_id": f"session_compact:{artifact.session_id}",
            "source_kind": "session_compact",
            "source_path": artifact.session_id,
            "target": f"session_compact:{artifact.session_id}",
            "score": hybrid_search_score(
                query,
                preview + "\n" + render_compact_summary_text(artifact),
                embedding_backend=self.deps.embedding_backend,
            ),
            "summary_preview": preview,
            "topics": ", ".join(artifact.discussion_topics),
            "time_range": artifact.updated_at[:10] if artifact.updated_at else "",
            "session_id": artifact.session_id,
            "snippet": preview,
        }

    def _search_session_context(
        self,
        *,
        session: SessionState,
        query: str,
        max_results: int,
        exclude_latest_user_message: bool = False,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        last_user_index = len(session.messages)
        eligible: list[tuple[int, Any]] = []
        for index, item in enumerate(session.messages, start=1):
            if exclude_latest_user_message and index == last_user_index and item.role == "user":
                continue
            eligible.append((index, item))
        scores = hybrid_search_scores(
            query,
            [item.content for _, item in eligible],
            embedding_backend=self.deps.embedding_backend,
        )
        for (index, item), score in zip(eligible, scores):
            snippet = extract_snippet(item.content, query, max_chars=220)
            if score <= 0:
                continue
            role = item.role or "unknown"
            recency_boost = 1.0 + 0.35 * (index / max(len(session.messages), 1))
            results.append(
                {
                    "source_id": f"session:{role}:{index}",
                    "source_kind": "session",
                    "source_path": "session",
                    "target": f"session:{role}:{index}",
                    "score": score * recency_boost,
                    "snippet": snippet,
                }
            )
        results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return results[:max_results]

    def _search_current_compact_context(
        self,
        *,
        session: SessionState,
        query: str,
        max_results: int,
        query_embedding: Any | None = None,
    ) -> list[dict[str, Any]]:
        manager = self._compaction_manager()
        if manager is None:
            return []
        artifact = manager.load_or_build_artifact(session)
        return self._search_compact_artifact_chunks(
            query=query,
            artifacts=[artifact],
            max_results=max_results,
            query_embedding=query_embedding,
        )

    def _search_archived_session_context(
        self,
        *,
        session: SessionState,
        query: str,
        max_results: int,
        scope: str,
        time_scope: TimeScope | None,
        query_embedding: Any | None = None,
    ) -> list[dict[str, Any]]:
        store = self.deps.session_store
        if store is None or max_results <= 0:
            return []

        results: list[dict[str, Any]] = []
        per_session_limit = max(self.deps.config.archive_session_search_messages_per_session, 1)
        for session_id in self._history_session_ids(session, scope=scope, time_scope=time_scope):
            archived_session = store.load_session(session_id)
            if archived_session is None:
                continue
            recent_messages = archived_session.messages[-per_session_limit:]
            if not recent_messages:
                continue
            base_index = max(len(archived_session.messages) - len(recent_messages), 0)
            items = [
                EmbeddingIndexItem.create(
                    source_id=f"session:{session_id}:{item.role}:{base_index + offset}",
                    source_kind="session_archive",
                    source_path=session_id,
                    target=f"session:{session_id}:{item.role}:{base_index + offset}",
                    text=item.content,
                )
                for offset, item in enumerate(recent_messages, start=1)
            ]
            scores = self._score_indexed_items(
                query=query,
                items=items,
                shard_path=self.deps.embedding_index_manager.session_shard_path(session_id)
                if self.deps.embedding_index_manager is not None
                else None,
                query_embedding=query_embedding,
            )
            for metadata, original, score in zip(items, recent_messages, scores):
                snippet = extract_snippet(original.content, query, max_chars=220)
                if score <= 0:
                    continue
                results.append(
                    {
                        "source_id": metadata.source_id,
                        "source_kind": metadata.source_kind,
                        "source_path": metadata.source_path,
                        "target": metadata.target,
                        "score": score,
                        "snippet": snippet,
                    }
                )
        results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return results[:max_results]

    def _search_compact_history_context(
        self,
        *,
        session: SessionState,
        query: str,
        max_results: int,
        scope: str,
        time_scope: TimeScope | None,
        query_embedding: Any | None = None,
    ) -> list[dict[str, Any]]:
        artifacts = self._history_compaction_artifacts(session, scope=scope, time_scope=time_scope)
        return self._search_compact_artifact_chunks(
            query=query,
            artifacts=artifacts,
            max_results=max_results,
            query_embedding=query_embedding,
        )

    def _search_compact_artifact_chunks(
        self,
        *,
        query: str,
        artifacts: list[Any],
        max_results: int,
        query_embedding: Any | None = None,
    ) -> list[dict[str, Any]]:
        manager = self._compaction_manager()
        if manager is None:
            return []
        results: list[dict[str, Any]] = []
        for artifact in artifacts:
            chunks = manager.search_chunks(artifact)
            items = [
                EmbeddingIndexItem.create(
                    source_id=chunk["target"],
                    source_kind=chunk["source_kind"],
                    source_path=chunk["source_path"],
                    target=chunk["target"],
                    text=chunk["text"],
                )
                for chunk in chunks
            ]
            scores = self._score_indexed_items(
                query=query,
                items=items,
                shard_path=self.deps.embedding_index_manager.compaction_shard_path(artifact.session_id)
                if self.deps.embedding_index_manager is not None
                else None,
                query_embedding=query_embedding,
            )
            for chunk, item, score in zip(chunks, items, scores):
                if score <= 0:
                    continue
                results.append(
                    {
                        "source_id": item.source_id,
                        "source_kind": item.source_kind,
                        "source_path": item.source_path,
                        "target": item.target,
                        "score": score,
                        "snippet": extract_snippet(chunk["text"], query, max_chars=220),
                        "summary_preview": chunk["summary_preview"],
                        "topics": chunk["topics"],
                        "time_range": chunk["time_range"],
                        "session_id": chunk["session_id"],
                    }
                )
        return self._rerank_search_results(
            self._prefer_specific_compact_targets(results),
            max_results=max_results,
        )

    def _build_query_embedding(self, query: str) -> Any | None:
        manager = self.deps.embedding_index_manager
        if manager is None or not self.deps.config.embedding_index_enabled:
            return None
        clean_query = query.strip()
        if not clean_query:
            return None
        return manager.build_query_embedding(clean_query)

    def _score_indexed_items(
        self,
        *,
        query: str,
        items: list[EmbeddingIndexItem],
        shard_path: Any | None,
        query_embedding: Any | None = None,
    ) -> list[float]:
        manager = self.deps.embedding_index_manager
        if manager is None or shard_path is None or not self.deps.config.embedding_index_enabled:
            return hybrid_search_scores(
                query,
                [item.text for item in items],
                embedding_backend=self.deps.embedding_backend,
            )
        if query_embedding is not None:
            return manager.score_items_with_query_embedding(
                query_embedding,
                items,
                shard_path=shard_path,
            )
        return manager.score_items(query, items, shard_path=shard_path)

    def _prefer_specific_compact_targets(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return []

        specific_targets_by_session: dict[str, list[dict[str, Any]]] = {}
        for item in results:
            target = str(item.get("target") or "")
            session_id = str(item.get("session_id") or "")
            if "#compression_history:" not in target or not session_id:
                continue
            specific_targets_by_session.setdefault(session_id, []).append(item)

        adjusted: list[dict[str, Any]] = []
        for item in results:
            payload = dict(item)
            target = str(payload.get("target") or "")
            session_id = str(payload.get("session_id") or "")
            score = float(payload.get("score", 0.0))
            specific_targets = specific_targets_by_session.get(session_id, [])
            if target.startswith("session_compact:") and "#compression_history:" in target:
                payload["score"] = score + 0.25
            elif target == f"session_compact:{session_id}" and specific_targets:
                payload["score"] = score * 0.35
            adjusted.append(payload)
        return adjusted

    def _history_compaction_artifacts(
        self,
        session: SessionState,
        *,
        scope: str,
        time_scope: TimeScope | None,
    ) -> list[Any]:
        manager = self._compaction_manager()
        if manager is None:
            return []
        if scope == "time_window" and time_scope is not None:
            return manager.artifacts_for_time_window(time_scope.from_days_ago, time_scope.to_days_ago)
        artifacts: list[Any] = []
        for session_id in self._history_session_ids(session, scope=scope, time_scope=time_scope):
            artifact = manager.load_or_build_artifact_for_session_id(session_id)
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts

    def _history_session_ids(
        self,
        session: SessionState,
        *,
        scope: str,
        time_scope: TimeScope | None,
    ) -> list[str]:
        store = self.deps.session_store
        if store is None:
            return []
        infos = store.list_session_infos()
        results: list[str] = []
        for item in infos:
            session_id = str(item.get("session_id", "")).strip()
            if not session_id or session_id == session.session_id:
                continue
            if scope == "time_window" and time_scope is not None:
                updated_at = str(item.get("updated_at", ""))
                if not self._timestamp_matches_time_scope(updated_at, time_scope):
                    continue
            results.append(session_id)
            if len(results) >= self.deps.config.archive_session_search_limit:
                break
        return results

    def _timestamp_matches_time_scope(self, timestamp: str, time_scope: TimeScope) -> bool:
        if not timestamp:
            return False
        date_text = timestamp[:10]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
            return False
        target_day = date.fromisoformat(date_text)
        older_bound = max(time_scope.from_days_ago, time_scope.to_days_ago)
        newer_bound = min(time_scope.from_days_ago, time_scope.to_days_ago)
        lower = date.today() - timedelta(days=older_bound)
        upper = date.today() - timedelta(days=newer_bound)
        return lower <= target_day <= upper

    def _rerank_search_results(self, results: list[dict[str, Any]], *, max_results: int) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen = set()
        for item in sorted(results, key=lambda payload: float(payload.get("score", 0.0)), reverse=True):
            key = (item["source_id"], item.get("target"), item.get("snippet"), item.get("summary_preview"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        if not deduped:
            return []
        return mmr_rerank(
            deduped[: max(max_results * SESSION_SEARCH_CANDIDATE_MULTIPLIER, max_results)],
            limit=max_results,
            text_key="snippet",
        )

    def _merge_ranked_lookup_tiers(
        self,
        tiers: list[list[dict[str, Any]]],
        *,
        max_results: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen = set()
        for tier_index, tier in enumerate(tiers):
            tier_bias = max(0.0, 0.06 * (len(tiers) - tier_index - 1))
            for item in self._rerank_search_results(
                tier,
                max_results=max_results * SESSION_SEARCH_CANDIDATE_MULTIPLIER,
            ):
                key = (item["source_id"], item.get("target"))
                if key in seen:
                    continue
                seen.add(key)
                payload = dict(item)
                payload["score"] = float(payload.get("score", 0.0)) + tier_bias
                merged.append(payload)
        return self._rerank_search_results(merged, max_results=max_results)

    def _format_memory_search_results(self, results: list[dict[str, Any]], *, search_mode: str) -> str:
        lines: list[str] = []
        for index, item in enumerate(results, start=1):
            source_kind = str(item.get("source_kind") or "memory")
            target = str(item.get("target") or item["source_id"])
            source_path = str(item.get("source_path") or "")
            score = float(item.get("score", 0.0))
            lines.append(f"{index}. [{source_kind}] target={target} score={score:.3f}")
            if source_path:
                lines.append(f"source: {source_path}")
            if search_mode == "recap":
                topics = str(item.get("topics") or "(empty)")
                time_range = str(item.get("time_range") or "(empty)")
                preview = str(item.get("summary_preview") or item.get("snippet") or "")
                lines.append(f"time_range: {time_range}")
                lines.append(f"topics: {topics}")
                lines.append(f"summary_preview: {preview}")
            else:
                lines.append(f"snippet: {item['snippet']}")
        return "\n".join(lines)

    def _memory_get(self, session: SessionState, *, target: str) -> dict[str, Any]:
        requested_target = target.strip()
        clean_target = self._resolve_recent_content_target(session, requested_target)
        if clean_target.startswith("session_compact:"):
            payload = self._memory_get_compact(clean_target)
        elif clean_target.startswith("session:"):
            payload = self._memory_get_session(session, clean_target)
        else:
            payload = self.deps.workspace.memory_get(target=clean_target)
        payload["target"] = requested_target
        payload["resolved_target"] = str(payload.get("resolved_target") or clean_target)
        return payload

    def _memory_get_compact(self, target: str) -> dict[str, Any]:
        manager = self._compaction_manager()
        if manager is None:
            raise ValueError("unsupported_memory_target")

        time_window_match = re.fullmatch(r"session_compact:time_window:(\d+)-(\d+)#summary", target)
        if time_window_match:
            from_days_ago = int(time_window_match.group(1))
            to_days_ago = int(time_window_match.group(2))
            summary = manager.build_time_window_summary(from_days_ago, to_days_ago)
            return {
                "target": target,
                "resolved_target": target,
                "source_path": summary["time_range"],
                "content": summary["content"],
                "line_start": 1,
                "line_end": max(len(str(summary["content"]).splitlines()), 1),
            }

        session_match = re.fullmatch(r"session_compact:([0-9a-f]{32})(?:#(?P<section>[^:]+):(?P<index>\d+))?", target)
        if not session_match:
            raise ValueError("unsupported_memory_target")

        session_id = session_match.group(1)
        artifact = manager.load_or_build_artifact_for_session_id(session_id)
        if artifact is None:
            content = ""
        else:
            section = session_match.group("section")
            index_text = session_match.group("index")
            if section == "compression_history" and index_text is not None:
                item_index = int(index_text)
                if 0 <= item_index < len(artifact.compression_history):
                    item = artifact.compression_history[item_index]
                    content = self._compression_full_content(session_id=session_id, turn_index=item.turn_index)
                    if not content:
                        content = "\n".join(
                            [
                                f"turn_index: {item.turn_index}",
                                f"user_request: {item.user_request}",
                                f"compressed_preview: {item.compressed_preview}",
                                f"entities: {', '.join(item.entities)}",
                                f"timestamp: {item.timestamp}",
                            ]
                        ).strip()
                else:
                    content = ""
            else:
                content = render_compact_summary_text(artifact)
        return {
            "target": target,
            "resolved_target": target,
            "source_path": session_id,
            "content": content,
            "line_start": 1,
            "line_end": max(len(content.splitlines()), 1),
        }

    def _compression_full_content(self, *, session_id: str, turn_index: int) -> str:
        store = self.deps.session_store
        if store is None:
            return ""
        for record in store.load_turn_records(session_id):
            if int(record.get("turn_index", 0)) != turn_index:
                continue
            tool_trace = record.get("tool_trace") or {}
            if tool_trace.get("requested_tool") != "compress_chapter":
                continue
            return str(record.get("assistant_reply", "")).strip()
        return ""

    def _latest_session_compression_record(self, session_id: str) -> dict[str, Any] | None:
        store = self.deps.session_store
        if store is None:
            return None
        records: list[dict[str, Any]] = []
        for record in store.load_turn_records(session_id):
            tool_trace = record.get("tool_trace") or {}
            if tool_trace.get("requested_tool") != "compress_chapter":
                continue
            assistant_reply = str(record.get("assistant_reply", "")).strip()
            if not assistant_reply:
                continue
            turn_index = int(record.get("turn_index", 0))
            records.append(
                {
                    "turn_index": turn_index,
                    "assistant_reply": assistant_reply,
                    "timestamp": str(record.get("timestamp", "")).strip(),
                }
            )
        if not records:
            return None
        records.sort(key=lambda item: (str(item.get("timestamp", "")), int(item.get("turn_index", 0))), reverse=True)
        return records[0]

    def _latest_session_compression_content(self, session_id: str) -> str:
        record = self._latest_session_compression_record(session_id)
        if record is None:
            return ""
        return str(record.get("assistant_reply", "")).strip()

    def _memory_get_session(self, session: SessionState, target: str) -> dict[str, Any]:
        if target == "session:latest_compress":
            content = self._latest_session_compression_content(session.session_id)
            return {
                "target": target,
                "resolved_target": target,
                "source_path": session.session_id,
                "content": content,
                "line_start": 1,
                "line_end": max(len(content.splitlines()), 1),
            }

        archived_latest = re.fullmatch(r"session:([0-9a-f]{32}):latest_compress", target)
        if archived_latest:
            session_id = archived_latest.group(1)
            content = self._latest_session_compression_content(session_id)
            return {
                "target": target,
                "resolved_target": target,
                "source_path": session_id,
                "content": content,
                "line_start": 1,
                "line_end": max(len(content.splitlines()), 1),
            }

        match = re.fullmatch(r"session:(?:(?P<session_id>[0-9a-f]{32}):)?(?P<role>user|assistant):(?P<index>\d+)", target)
        if not match:
            raise ValueError("unsupported_memory_target")
        target_session_id = match.group("session_id") or session.session_id
        role = str(match.group("role"))
        index = int(match.group("index"))
        target_session = session if target_session_id == session.session_id else None
        if target_session is None and self.deps.session_store is not None:
            target_session = self.deps.session_store.load_session(target_session_id)
        if target_session is None or index <= 0 or index > len(target_session.messages):
            content = ""
        else:
            message = target_session.messages[index - 1]
            content = message.content if message.role == role else ""
        return {
            "target": target,
            "resolved_target": target,
            "source_path": target_session_id,
            "content": content,
            "line_start": 1,
            "line_end": max(len(content.splitlines()), 1),
        }

    def _resolve_recent_content_target(self, session: SessionState, target: str) -> str:
        clean = target.strip()
        if not clean.startswith("content_ref:"):
            return clean
        references = self._recent_content_references(session)
        if not references:
            raise ValueError("unknown_content_reference")
        if clean == "content_ref:latest":
            return references[0].resolved_target
        for item in references:
            if item.alias == clean:
                return item.resolved_target
        raise ValueError("unknown_content_reference")

    def _remember_recent_content_reference(self, session: SessionState, memory_doc: dict[str, Any]) -> None:
        if self.deps.meta_store is None:
            return
        content = str(memory_doc.get("content") or "").strip()
        resolved_target = str(memory_doc.get("resolved_target") or "").strip()
        if not content or not resolved_target:
            return

        meta = self.deps.meta_store.get_or_create(session.session_id)
        references: list[RecentContentReference] = []
        for payload in meta.recent_content_references:
            try:
                reference = RecentContentReference.model_validate(payload)
            except ValidationError:
                continue
            if reference.resolved_target == resolved_target:
                continue
            references.append(reference)

        preview = self._memory_observation_preview(content)
        references.insert(
            0,
            RecentContentReference(
                alias="",
                resolved_target=resolved_target,
                source_path=str(memory_doc.get("source_path") or ""),
                preview=preview,
                content_length=len(content),
                created_turn_index=max((len(session.messages) + 1) // 2, 1),
            ),
        )
        references = references[:3]
        for index, item in enumerate(references):
            item.alias = "content_ref:latest" if index == 0 else f"content_ref:{index}"

        meta.recent_content_references = [item.model_dump() for item in references]
        self.deps.meta_store.save(meta)

    def _recent_content_references(self, session: SessionState) -> list[RecentContentReference]:
        if self.deps.meta_store is None:
            return []
        meta = self.deps.meta_store.load(session.session_id)
        if meta is None:
            return []
        references: list[RecentContentReference] = []
        for payload in meta.recent_content_references[:3]:
            try:
                references.append(RecentContentReference.model_validate(payload))
            except ValidationError:
                continue
        return references

    def _merge_memory_write(self, target: MemoryWrite, incoming: MemoryWrite | None) -> None:
        if incoming is None:
            return
        for entry in incoming.daily:
            if entry not in target.daily:
                target.daily.append(entry)
        for entry in incoming.long_term:
            if entry not in target.long_term:
                target.long_term.append(entry)

    def _merge_context_report(self, target: ContextReport, incoming: ContextReport) -> ContextReport:
        target.estimated_tokens = max(target.estimated_tokens, incoming.estimated_tokens)
        target.pruning_applied = target.pruning_applied or incoming.pruning_applied
        target.compaction_applied = target.compaction_applied or incoming.compaction_applied
        target.compaction_source = incoming.compaction_source or target.compaction_source
        target.memory_flush_applied = target.memory_flush_applied or incoming.memory_flush_applied
        target.review_triggered = target.review_triggered or incoming.review_triggered
        target.review_verdict = incoming.review_verdict or target.review_verdict
        target.review_reason = incoming.review_reason or target.review_reason
        for item in incoming.recall_targets:
            if item not in target.recall_targets:
                target.recall_targets.append(item)
        for item in incoming.context_blocks:
            if item not in target.context_blocks:
                target.context_blocks.append(item)
        for item in incoming.memory_flush_daily:
            if item not in target.memory_flush_daily:
                target.memory_flush_daily.append(item)
        for item in incoming.memory_flush_long_term:
            if item not in target.memory_flush_long_term:
                target.memory_flush_long_term.append(item)
        return target

    def _truncate_memory_text(self, text: str) -> str:
        clean = text.strip()
        max_chars = self.deps.config.memory_tool_max_chars
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars].rstrip() + "..."

    def _memory_observation_preview(self, text: str) -> str:
        clean = text.strip()
        if not clean:
            return "(empty)"
        preview_limit = min(
            self.deps.config.memory_tool_max_chars,
            max(self.deps.config.context_micro_preview_chars * 2, 240),
        )
        return extract_snippet(clean, clean, max_chars=preview_limit)

    def _decision_event(self, turn_index: int, step_index: int, decision: DecisionOutput) -> dict[str, Any]:
        return self._event(
            turn_index,
            "agent_decision",
            step_index=step_index,
            payload=decision.model_dump(),
        )

    def _plan_event(self, turn_index: int, event_type: str, plan: ExecutionPlanOutput) -> dict[str, Any]:
        return self._event(
            turn_index,
            event_type,
            payload=plan.model_dump(),
        )

    def _plan_step_completed_event(self, turn_index: int, step: PlanStep) -> dict[str, Any]:
        return self._event(
            turn_index,
            "plan_step_completed",
            step_index=step.step_index,
            payload=step.model_dump(),
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
        context_report: ContextReport | None = None,
    ) -> AgentTurnResult:
        session.add_assistant_message(reply)
        session.refresh_summary(self.deps.config.session_summary_max_chars)
        if context_report is not None:
            transcript_events.append(self._event((len(session.messages) // 2), "context_report", payload=context_report.model_dump()))
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
            context_report=(context_report.model_dump() if context_report is not None else {}),
        )

    def _finalize_error(
        self,
        session: SessionState,
        reply: str,
        transcript_events: list[dict[str, Any]],
        decision: dict[str, Any] | None = None,
        tool_trace: dict[str, Any] | None = None,
        context_report: ContextReport | None = None,
    ) -> AgentTurnResult:
        session.add_assistant_message(reply)
        session.refresh_summary(self.deps.config.session_summary_max_chars)
        if context_report is not None:
            transcript_events.append(self._event((len(session.messages) // 2), "context_report", payload=context_report.model_dump()))
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
            context_report=(context_report.model_dump() if context_report is not None else {}),
        )
