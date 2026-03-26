from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .maintenance import build_long_term_candidates
from .memory import SessionState, SessionStore
from .schemas import CompactCompressionHistoryEntry, CompactSummaryArtifact
from .search_utils import extract_snippet
from .session_meta import SessionMetaStore, utc_now_iso


TOPIC_HINTS = (
    ("人物", "人物关系"),
    ("剧情", "剧情推进"),
    ("设定", "设定"),
    ("世界观", "世界观"),
    ("压缩", "章节压缩"),
)

ENTITY_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,4}")


def render_compact_summary_text(artifact: CompactSummaryArtifact) -> str:
    parts = ["# Compacted Session Context", f"Session Goal: {artifact.session_goal or '(empty)'}"]

    if artifact.discussion_topics:
        parts.append("Discussion Topics:")
        parts.extend(f"- {item}" for item in artifact.discussion_topics)

    if artifact.compression_history:
        parts.append("Compression History:")
        parts.append("- Note: entries below are recall indexes only. Full compression text is not shown here; use memory_get on target when you need the exact content.")
        for index, item in enumerate(artifact.compression_history, start=1):
            target = item.full_content_target or f"session_compact:{artifact.session_id}#compression_history:{index - 1}"
            alias_text = f" aliases={','.join(item.ordinal_aliases)}" if item.ordinal_aliases else ""
            entity_text = f" entities={','.join(item.entities[:4])}" if item.entities else ""
            request_text = item.user_request.strip() or "(empty)"
            parts.append(
                f"- {index}. turn={item.turn_index} target={target} request={request_text}{alias_text}{entity_text} preview=omitted"
            )

    if artifact.story_facts:
        parts.append("Story Facts:")
        parts.extend(f"- {item}" for item in artifact.story_facts)

    if artifact.user_preferences:
        parts.append("User Preferences:")
        parts.extend(f"- {item}" for item in artifact.user_preferences)

    if artifact.open_loops:
        parts.append("Open Loops:")
        parts.extend(f"- {item}" for item in artifact.open_loops)

    if artifact.timeline_summary:
        parts.append("Timeline Summary:")
        parts.extend(f"- {item}" for item in artifact.timeline_summary)

    if artifact.search_hints:
        parts.append("Search Hints:")
        parts.extend(f"- {item}" for item in artifact.search_hints)

    return "\n".join(parts).strip()


class ContextCompactionManager:
    def __init__(
        self,
        config: AgentConfig,
        session_store: SessionStore,
        meta_store: SessionMetaStore,
        summary_backend: Any | None = None,
    ) -> None:
        self.config = config
        self.session_store = session_store
        self.meta_store = meta_store
        self.summary_backend = summary_backend
        self.transcript_root = Path(config.transcript_root)
        self.compaction_root = Path(config.compaction_root)

    def bootstrap(self) -> None:
        self.transcript_root.mkdir(parents=True, exist_ok=True)
        self.compaction_root.mkdir(parents=True, exist_ok=True)

    def transcript_dir(self, session_id: str) -> Path:
        return self.transcript_root / session_id

    def transcript_path(self, session_id: str, turn_index: int) -> Path:
        return self.transcript_dir(session_id) / f"transcript_{turn_index:04d}.jsonl"

    def compaction_path(self, session_id: str) -> Path:
        return self.compaction_root / f"{session_id}.json"

    def load_compaction(self, session_id: str) -> CompactSummaryArtifact | None:
        path = self.compaction_path(session_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return CompactSummaryArtifact.model_validate(payload)

    def save_compaction(self, artifact: CompactSummaryArtifact) -> Path:
        self.bootstrap()
        path = self.compaction_path(artifact.session_id)
        path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_or_build_artifact(self, session: SessionState) -> CompactSummaryArtifact:
        artifact = self.load_compaction(session.session_id)
        if artifact is not None:
            return artifact
        return self._build_fallback_artifact(session)

    def load_or_build_artifact_for_session_id(self, session_id: str) -> CompactSummaryArtifact | None:
        artifact = self.load_compaction(session_id)
        if artifact is not None:
            return artifact
        session = self.session_store.load_session(session_id)
        if session is None:
            return None
        return self._build_fallback_artifact(session, updated_at=self._session_updated_at(session_id))

    def build_prompt_context(self, session: SessionState) -> str:
        meta = self.meta_store.load(session.session_id)
        if meta is not None and meta.cached_compact_summary.strip():
            return meta.cached_compact_summary.strip()
        artifact = self.load_or_build_artifact(session)
        return render_compact_summary_text(artifact)

    def build_micro_compact_context(self, session: SessionState) -> str:
        keep = max(self.config.context_micro_keep_recent_messages, 0)
        older_messages = session.messages[:-keep] if keep else list(session.messages)
        if not older_messages:
            return ""

        total_messages = len(older_messages)
        previews: list[str] = []
        for item in older_messages[-4:]:
            role = "用户" if item.role == "user" else "助手"
            previews.append(f"- {role}: {extract_snippet(item.content, item.content, self.config.context_micro_preview_chars)}")
        return "\n".join(
            [
                "# Micro Compact",
                f"Earlier Messages: {total_messages}",
                "Earlier Message Previews:",
                *previews,
            ]
        ).strip()

    def maybe_auto_compact(self, session: SessionState, estimated_tokens: int) -> CompactSummaryArtifact | None:
        if estimated_tokens < self.config.context_auto_compact_token_threshold:
            return None
        turn_index = max(len(session.messages) // 2, 1)
        if turn_index < self.config.context_auto_compact_min_turns:
            return None
        meta = self.meta_store.get_or_create(session.session_id)
        if not meta.dirty_compaction and meta.last_compaction_turn_index >= turn_index:
            return None
        return self.compact_session(session, force=True)

    def compact_session(self, session: SessionState, force: bool = False) -> CompactSummaryArtifact:
        self.bootstrap()
        turn_index = max(len(session.messages) // 2, 1)
        meta = self.meta_store.get_or_create(session.session_id)
        if (
            not force
            and not meta.dirty_compaction
            and meta.last_compaction_turn_index >= turn_index
            and meta.latest_compaction_path
        ):
            cached = self.load_compaction(session.session_id)
            if cached is not None:
                return cached

        artifact = self._summarize_with_backend(session)
        if artifact is None:
            artifact = self._build_fallback_artifact(session)

        transcript_path = self._snapshot_transcript(session, turn_index)
        artifact.transcript_path = str(transcript_path)
        artifact.updated_at = utc_now_iso()
        compaction_path = self.save_compaction(artifact)

        meta.cached_compact_summary = render_compact_summary_text(artifact)
        meta.latest_transcript_path = str(transcript_path)
        meta.latest_compaction_path = str(compaction_path)
        meta.last_compaction_turn_index = turn_index
        meta.dirty_compaction = False
        self.meta_store.save(meta)
        return artifact

    def search_chunks(self, artifact: CompactSummaryArtifact) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        chunks.append(
            {
                "target": f"session_compact:{artifact.session_id}",
                "source_kind": "session_compact",
                "source_path": artifact.session_id,
                "text": render_compact_summary_text(artifact),
                "summary_preview": self._artifact_preview(artifact),
                "topics": ", ".join(artifact.discussion_topics),
                "time_range": artifact.updated_at[:10] if artifact.updated_at else "",
                "session_id": artifact.session_id,
            }
        )
        for index, item in enumerate(artifact.compression_history):
            target = item.full_content_target or f"session_compact:{artifact.session_id}#compression_history:{index}"
            alias_text = " ".join(item.ordinal_aliases)
            chunks.append(
                {
                    "target": target,
                    "source_kind": "session_compact",
                    "source_path": artifact.session_id,
                    "text": (
                        f"{item.user_request}\n"
                        f"{item.compressed_preview}\n"
                        f"{alias_text}\n"
                        f"{target}\n"
                        f"{' '.join(item.entities)}"
                    ),
                    "summary_preview": item.compressed_preview,
                    "topics": ", ".join([*item.entities, *item.ordinal_aliases]),
                    "time_range": item.timestamp[:10] if item.timestamp else "",
                    "session_id": artifact.session_id,
                }
            )
        return chunks

    def build_time_window_summary(self, from_days_ago: int, to_days_ago: int) -> dict[str, str]:
        artifacts = self.artifacts_for_time_window(from_days_ago, to_days_ago)
        start_day = date.today() - timedelta(days=from_days_ago)
        end_day = date.today() - timedelta(days=to_days_ago)
        topics: list[str] = []
        lines: list[str] = []
        compression_count = 0
        for artifact in artifacts:
            topics.extend(artifact.discussion_topics)
            compression_count += len(artifact.compression_history)
            preview = self._artifact_preview(artifact)
            lines.append(f"- {artifact.updated_at[:10] if artifact.updated_at else artifact.session_id}: {preview}")
        body = "\n".join(
            [
                f"时间范围：{start_day.isoformat()} 到 {end_day.isoformat()}",
                f"命中会话数：{len(artifacts)}",
                f"章节压缩次数：{compression_count}",
                f"主题：{', '.join(sorted(set(topics))[:8]) or '(empty)'}",
                "摘要：",
                *(lines or ["- (empty)"]),
            ]
        ).strip()
        return {
            "target": f"session_compact:time_window:{from_days_ago}-{to_days_ago}#summary",
            "content": body,
            "summary_preview": extract_snippet(body, body, 260),
            "topics": ", ".join(sorted(set(topics))[:8]),
            "time_range": f"{start_day.isoformat()}..{end_day.isoformat()}",
        }

    def artifacts_for_time_window(self, from_days_ago: int, to_days_ago: int) -> list[CompactSummaryArtifact]:
        infos = self.session_store.list_session_infos()
        older_bound = max(from_days_ago, to_days_ago)
        newer_bound = min(from_days_ago, to_days_ago)
        upper = date.today() - timedelta(days=newer_bound)
        lower = date.today() - timedelta(days=older_bound)
        results: list[CompactSummaryArtifact] = []
        for item in infos:
            updated_at = str(item.get("updated_at", ""))
            date_text = updated_at[:10]
            if not _looks_like_date(date_text):
                continue
            target_day = date.fromisoformat(date_text)
            if not (lower <= target_day <= upper):
                continue
            session_id = str(item.get("session_id", "")).strip()
            artifact = self.load_or_build_artifact_for_session_id(session_id)
            if artifact is not None:
                results.append(artifact)
        return results

    def _snapshot_transcript(self, session: SessionState, turn_index: int) -> Path:
        target_dir = self.transcript_dir(session.session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = self.transcript_path(session.session_id, turn_index)
        with path.open("w", encoding="utf-8") as handle:
            for index, item in enumerate(session.messages, start=1):
                payload = {
                    "message_index": index,
                    "role": item.role,
                    "content": item.content,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def _summarize_with_backend(self, session: SessionState) -> CompactSummaryArtifact | None:
        if self.summary_backend is None:
            return None
        try:
            artifact = self.summary_backend.summarize(
                session_id=session.session_id,
                messages=session.chat_history(),
                turn_records=self._turn_records_for_session(session),
                session_summary=session.summary,
            )
        except Exception:
            return None
        if artifact.session_id != session.session_id:
            artifact.session_id = session.session_id
        return artifact

    def _turn_records_for_session(self, session: SessionState) -> list[dict[str, Any]]:
        records = self.session_store.load_turn_records(session.session_id)
        if records:
            return records
        return self._infer_turn_records_from_messages(session)

    def _build_fallback_artifact(self, session: SessionState, updated_at: str | None = None) -> CompactSummaryArtifact:
        turn_records = self._turn_records_for_session(session)
        discussion_topics = self._discussion_topics(session)
        compression_history = self._compression_history(session.session_id, turn_records)
        timeline_summary = [
            f"当前会话共 {len(turn_records)} 轮。",
            f"章节压缩次数：{len(compression_history)}。",
        ]
        user_preferences = build_long_term_candidates(turn_records, repeat_threshold=1)
        story_facts = self._story_facts(turn_records)
        open_loops = self._open_loops(session)
        artifact = CompactSummaryArtifact(
            session_id=session.session_id,
            updated_at=(updated_at or utc_now_iso()),
            source="rule_based",
            session_goal=self._session_goal(session),
            discussion_topics=discussion_topics,
            compression_history=compression_history,
            story_facts=story_facts,
            user_preferences=user_preferences,
            open_loops=open_loops,
            timeline_summary=timeline_summary,
            search_hints=self._search_hints(discussion_topics, compression_history, user_preferences, story_facts),
        )
        return artifact

    def _infer_turn_records_from_messages(self, session: SessionState) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        messages = session.messages
        turn_index = 0
        index = 0
        while index < len(messages):
            current = messages[index]
            if current.role != "user":
                index += 1
                continue
            turn_index += 1
            assistant_reply = ""
            if index + 1 < len(messages) and messages[index + 1].role == "assistant":
                assistant_reply = messages[index + 1].content
            records.append(
                {
                    "turn_index": turn_index,
                    "user_message": current.content,
                    "assistant_reply": assistant_reply,
                    "tool_trace": {},
                    "action": "direct_reply",
                }
            )
            index += 2
        return records

    def _session_goal(self, session: SessionState) -> str:
        for item in reversed(session.messages):
            if item.role == "user" and item.content.strip():
                return extract_snippet(item.content, item.content, 120)
        return ""

    def _discussion_topics(self, session: SessionState) -> list[str]:
        joined = "\n".join(item.content for item in session.messages[-12:])
        topics = [label for keyword, label in TOPIC_HINTS if keyword in joined]
        if not topics and joined.strip():
            topics.append("小说对话")
        return list(dict.fromkeys(topics))

    def _compression_history(self, session_id: str, turn_records: list[dict[str, Any]]) -> list[CompactCompressionHistoryEntry]:
        results: list[CompactCompressionHistoryEntry] = []
        for record in turn_records:
            tool_trace = record.get("tool_trace") or {}
            if tool_trace.get("requested_tool") != "compress_chapter":
                continue
            user_message = str(record.get("user_message", "")).strip()
            assistant_reply = str(record.get("assistant_reply", "")).strip()
            results.append(
                CompactCompressionHistoryEntry(
                    turn_index=int(record.get("turn_index", 0)),
                    user_request=extract_snippet(user_message, user_message, 100),
                    compressed_preview=extract_snippet(assistant_reply, assistant_reply, 160),
                    entities=self._extract_entities(f"{user_message}\n{assistant_reply}"),
                    timestamp=str(record.get("timestamp", "")),
                )
            )
        recent = results[-6:]
        for index, item in enumerate(recent):
            aliases = [f"第{index + 1}次", f"第{index + 1}个"]
            if index == 0:
                aliases.extend(["第一次", "第1次", "第一个", "第1个", "最早一次", "最早一个"])
            if index == 1:
                aliases.extend(["第二次", "第2次", "第二个", "第2个"])
            if index == len(recent) - 1:
                aliases.extend(["最近一次", "最新一次", "上一次"])
            aliases.extend(["全部压缩", "所有压缩"])
            deduped_aliases = list(dict.fromkeys(alias for alias in aliases if alias))
            item.ordinal_aliases = deduped_aliases
            item.full_content_target = f"session_compact:{session_id}#compression_history:{index}"
        return recent

    def _story_facts(self, turn_records: list[dict[str, Any]]) -> list[str]:
        candidates: list[str] = []
        for record in turn_records[-6:]:
            assistant_reply = str(record.get("assistant_reply", "")).strip()
            if not assistant_reply:
                continue
            snippet = extract_snippet(assistant_reply, assistant_reply, 120)
            if snippet not in candidates:
                candidates.append(snippet)
            if len(candidates) >= 4:
                break
        return candidates

    def _open_loops(self, session: SessionState) -> list[str]:
        if not session.messages:
            return []
        if session.messages[-1].role == "user":
            return [extract_snippet(session.messages[-1].content, session.messages[-1].content, 120)]
        return []

    def _search_hints(
        self,
        discussion_topics: list[str],
        compression_history: list[CompactCompressionHistoryEntry],
        user_preferences: list[str],
        story_facts: list[str],
    ) -> list[str]:
        hints: list[str] = []
        hints.extend(discussion_topics)
        hints.extend(user_preferences)
        hints.extend(item.compressed_preview for item in compression_history[:2] if item.compressed_preview)
        hints.extend(story_facts[:2])
        clean: list[str] = []
        seen = set()
        for item in hints:
            value = item.strip()
            if not value or value in seen:
                continue
            clean.append(value)
            seen.add(value)
        return clean[:10]

    def _extract_entities(self, text: str) -> list[str]:
        counts: Counter[str] = Counter()
        for token in ENTITY_PATTERN.findall(text):
            if token in {"今天", "昨天", "用户", "助手", "当前会话", "章节压缩", "剧情推进"}:
                continue
            counts[token] += 1
        return [item for item, _ in counts.most_common(6)]

    def _artifact_preview(self, artifact: CompactSummaryArtifact) -> str:
        lines = [
            artifact.session_goal,
            *artifact.discussion_topics[:3],
            *(item.compressed_preview for item in artifact.compression_history[:2]),
            *artifact.timeline_summary[:2],
        ]
        merged = "；".join(item.strip() for item in lines if item and item.strip())
        return extract_snippet(merged, merged, 220) if merged else "(empty)"

    def _session_updated_at(self, session_id: str) -> str:
        for item in self.session_store.list_session_infos():
            if str(item.get("session_id", "")).strip() == session_id:
                return str(item.get("updated_at", "")).strip()
        return ""


def _looks_like_date(text: str) -> bool:
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True
