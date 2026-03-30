from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from .compaction import ContextCompactionManager, render_compact_summary_text
from .config import AgentConfig
from .maintenance import build_daily_memory_candidates, build_long_term_candidates
from .memory import SessionState, SessionStore
from .schemas import ContextBuildResult, ContextReport, RecentContentReference
from .session_meta import SessionMetaStore
from .workspace import WorkspaceManager

@dataclass(slots=True)
class ContextEngineDependencies:
    config: AgentConfig
    workspace: WorkspaceManager
    decision_backend: Any
    tool_prompt_docs: str = ""
    tool_names: tuple[str, ...] = ()
    session_store: SessionStore | None = None
    meta_store: SessionMetaStore | None = None
    compaction_manager: ContextCompactionManager | None = None
    search_memory_fn: Callable[..., list[dict[str, Any]]] | None = None


class ContextEngine:
    def __init__(self, deps: ContextEngineDependencies) -> None:
        self.deps = deps

    def build_turn_context(
        self,
        *,
        session: SessionState,
        user_text: str,
        loop_events: list[dict[str, Any]],
        execution_plan: dict[str, Any] | None = None,
        current_step: dict[str, Any] | None = None,
        completed_steps: list[dict[str, Any]] | None = None,
    ) -> ContextBuildResult:
        report = ContextReport(context_blocks=["messages", "session_summary"])
        active_messages = self._recent_messages(session)
        active_loop_events = list(loop_events)
        compacted_session_context = ""

        manager = self.deps.compaction_manager
        if manager is not None:
            compacted_session_context = manager.build_prompt_context(session)
            micro_context = manager.build_micro_compact_context(session)
            if micro_context.strip():
                compacted_session_context = "\n\n".join(
                    part for part in (compacted_session_context, micro_context) if part.strip()
                )
                report.pruning_applied = True
            if compacted_session_context.strip():
                report.context_blocks.append("compacted_session_context")

        recent_content_references = self._build_recent_content_reference_context(session=session)
        if recent_content_references.strip():
            report.context_blocks.append("recent_content_references")

        recalled_memory = ""
        workspace_docs = self.deps.workspace.load_workspace_docs(session.summary, recalled_memory)
        report.context_blocks.append("workspace_docs")

        estimated_tokens = self._estimate_tokens(
            messages=active_messages,
            workspace_docs=workspace_docs,
            session_summary=session.summary,
            compacted_session_context=compacted_session_context,
            recent_content_references=recent_content_references,
            loop_events=active_loop_events,
            execution_plan=execution_plan,
            current_step=current_step,
            completed_steps=completed_steps,
        )
        report.estimated_tokens = estimated_tokens

        self._maybe_flush_memory_before_compaction(session=session, estimated_tokens=estimated_tokens, report=report)

        active_loop_events, estimated_tokens = self._prune_loop_events_to_budget(
            messages=active_messages,
            workspace_docs=workspace_docs,
            session_summary=session.summary,
            compacted_session_context=compacted_session_context,
            recent_content_references=recent_content_references,
            loop_events=active_loop_events,
            execution_plan=execution_plan,
            current_step=current_step,
            completed_steps=completed_steps,
            report=report,
        )
        report.estimated_tokens = estimated_tokens

        if manager is not None:
            artifact = manager.maybe_auto_compact(session, estimated_tokens)
            if artifact is not None:
                compacted_session_context = render_compact_summary_text(artifact)
                micro_context = manager.build_micro_compact_context(session)
                if micro_context.strip():
                    compacted_session_context = "\n\n".join(
                        part for part in (compacted_session_context, micro_context) if part.strip()
                    )
                report.compaction_applied = True
                report.compaction_source = artifact.source
                report.context_blocks = [item for item in report.context_blocks if item != "compacted_session_context"]
                report.context_blocks.append("compacted_session_context")
                active_loop_events, estimated_tokens = self._prune_loop_events_to_budget(
                    messages=active_messages,
                    workspace_docs=workspace_docs,
                    session_summary=session.summary,
                    compacted_session_context=compacted_session_context,
                    recent_content_references=recent_content_references,
                    loop_events=active_loop_events,
                    execution_plan=execution_plan,
                    current_step=current_step,
                    completed_steps=completed_steps,
                    report=report,
                )
                report.estimated_tokens = estimated_tokens

        return ContextBuildResult(
            messages=active_messages,
            session_summary=session.summary,
            compacted_session_context=compacted_session_context,
            recalled_memory=recalled_memory,
            recent_content_references=recent_content_references,
            workspace_docs=workspace_docs,
            loop_events=active_loop_events,
            context_report=report,
        )

    def _recent_messages(self, session: SessionState) -> list[dict[str, str]]:
        raw_limit = max(self.deps.config.context_recent_raw_messages, 1)
        if len(session.messages) > raw_limit:
            return session.chat_history()[-raw_limit:]
        return session.chat_history()

    def _prune_loop_events(self, loop_events: list[dict[str, Any]], *, report: ContextReport) -> list[dict[str, Any]]:
        if not self.deps.config.context_pruning_enabled:
            return list(loop_events)

        keep_results = max(self.deps.config.context_pruning_keep_recent_tool_results, 0)
        result_indexes = [index for index, event in enumerate(loop_events) if event.get("event_type") == "tool_result"]
        prune_indexes = set(result_indexes[:-keep_results] if keep_results else result_indexes)
        if not prune_indexes:
            return list(loop_events)

        pruned: list[dict[str, Any]] = []
        for index, event in enumerate(loop_events):
            if index not in prune_indexes:
                pruned.append(dict(event))
                continue
            payload = dict(event)
            payload["observation"] = "(older tool result pruned by context engine)"
            pruned.append(payload)
        report.pruning_applied = True
        if "pruned_loop_events" not in report.context_blocks:
            report.context_blocks.append("pruned_loop_events")
        return pruned

    def _estimate_tokens(
        self,
        *,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        compacted_session_context: str,
        recent_content_references: str,
        loop_events: list[dict[str, Any]],
        execution_plan: dict[str, Any] | None = None,
        current_step: dict[str, Any] | None = None,
        completed_steps: list[dict[str, Any]] | None = None,
    ) -> int:
        estimator = getattr(self.deps.decision_backend, "estimate_prompt_tokens", None)
        if callable(estimator):
            try:
                return int(
                    estimator(
                        messages=messages,
                        workspace_docs=workspace_docs,
                        tool_prompt_docs=self.deps.tool_prompt_docs,
                        tool_names=self.deps.tool_names,
                        session_summary=session_summary,
                        compacted_session_context=compacted_session_context,
                        recent_content_references=recent_content_references,
                        loop_events=loop_events,
                        execution_plan=execution_plan,
                        current_step=current_step,
                        completed_steps=completed_steps,
                    )
                )
            except Exception:
                pass
        merged = (
            str(messages)
            + workspace_docs
            + session_summary
            + compacted_session_context
            + recent_content_references
            + str(execution_plan or {})
            + str(current_step or {})
            + str(completed_steps or [])
            + str(loop_events)
        )
        return max(len(merged) // 2, 1)

    def _maybe_flush_memory_before_compaction(
        self,
        *,
        session: SessionState,
        estimated_tokens: int,
        report: ContextReport,
    ) -> None:
        if not self.deps.config.memory_flush_enabled:
            return
        if estimated_tokens < self.deps.config.context_memory_flush_soft_threshold:
            return
        if self.deps.meta_store is None or self.deps.session_store is None:
            return

        meta = self.deps.meta_store.get_or_create(session.session_id)
        current_turn_index = max(len(session.messages) // 2, 1)
        if meta.last_memory_flush_turn_index > meta.last_compaction_turn_index:
            return
        turn_records = self.deps.session_store.load_turn_records(session.session_id)
        if not turn_records:
            return

        recent_records = turn_records[-max(self.deps.config.context_pruning_keep_recent_tool_results + 1, 2) :]
        daily_entries = build_daily_memory_candidates(recent_records)
        long_term_entries = build_long_term_candidates(
            turn_records,
            repeat_threshold=self.deps.config.long_term_repeat_threshold,
        )
        if not daily_entries and not long_term_entries:
            meta.last_memory_flush_turn_index = current_turn_index
            self.deps.meta_store.save(meta)
            return

        self.deps.workspace.append_daily_entries(daily_entries)
        self.deps.workspace.append_long_term_entries(long_term_entries)
        meta.last_memory_flush_turn_index = current_turn_index
        self.deps.meta_store.save(meta)

        report.memory_flush_applied = True
        report.memory_flush_daily = daily_entries
        report.memory_flush_long_term = long_term_entries
        if "memory_flush" not in report.context_blocks:
            report.context_blocks.append("memory_flush")

    def _build_recent_content_reference_context(self, *, session: SessionState) -> str:
        references = self._recent_content_references(session)
        if not references:
            return ""
        lines = ["# Recent Content References", "以下仅提供最近可展开内容的元信息，不包含全文。"]
        for item in references:
            lines.append(
                (
                    f"- alias={item.alias} target={item.resolved_target} "
                    f"source={item.source_path or '(empty)'} length={item.content_length}"
                )
            )
            if item.preview:
                lines.append(f"preview: {item.preview}")
        return "\n".join(lines)

    def _recent_content_references(self, session: SessionState) -> list[RecentContentReference]:
        if self.deps.meta_store is None:
            return []
        meta = self.deps.meta_store.load(session.session_id)
        if meta is None:
            return []
        results: list[RecentContentReference] = []
        for payload in meta.recent_content_references[:3]:
            try:
                results.append(RecentContentReference.model_validate(payload))
            except ValidationError:
                continue
        return results

    def _prune_loop_events_to_budget(
        self,
        *,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        compacted_session_context: str,
        recent_content_references: str,
        loop_events: list[dict[str, Any]],
        execution_plan: dict[str, Any] | None = None,
        current_step: dict[str, Any] | None = None,
        completed_steps: list[dict[str, Any]] | None = None,
        report: ContextReport,
    ) -> tuple[list[dict[str, Any]], int]:
        pruned = list(loop_events)
        estimated_tokens = self._estimate_tokens(
            messages=messages,
            workspace_docs=workspace_docs,
            session_summary=session_summary,
            compacted_session_context=compacted_session_context,
            recent_content_references=recent_content_references,
            loop_events=pruned,
            execution_plan=execution_plan,
            current_step=current_step,
            completed_steps=completed_steps,
        )
        if not self.deps.config.context_pruning_enabled:
            return pruned, estimated_tokens
        if estimated_tokens <= self.deps.config.context_pruning_soft_budget:
            return pruned, estimated_tokens

        keep_results = max(self.deps.config.context_pruning_keep_recent_tool_results, 0)
        result_indexes = [index for index, event in enumerate(pruned) if event.get("event_type") == "tool_result"]
        indexes_to_prune = result_indexes[:-keep_results] if keep_results else result_indexes
        for index in indexes_to_prune:
            payload = dict(pruned[index])
            payload["observation"] = "(older tool result pruned by context engine)"
            pruned[index] = payload
            estimated_tokens = self._estimate_tokens(
                messages=messages,
                workspace_docs=workspace_docs,
                session_summary=session_summary,
                compacted_session_context=compacted_session_context,
                recent_content_references=recent_content_references,
                loop_events=pruned,
                execution_plan=execution_plan,
                current_step=current_step,
                completed_steps=completed_steps,
            )
            if estimated_tokens <= self.deps.config.context_pruning_target_tokens:
                break

        if pruned != loop_events:
            report.pruning_applied = True
            if "pruned_loop_events" not in report.context_blocks:
                report.context_blocks.append("pruned_loop_events")
        return pruned, estimated_tokens
