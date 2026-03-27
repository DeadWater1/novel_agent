from __future__ import annotations

import time

from .compaction import ContextCompactionManager
from .config import AgentConfig
from .embedding_index import EmbeddingIndexItem, EmbeddingIndexManager
from .maintenance import build_daily_memory_candidates, build_long_term_candidates, rebuild_session_summary
from .memory import SessionState, SessionStore
from .session_meta import SessionMetaStore, utc_now_iso
from .workspace import WorkspaceManager


class HeartbeatManager:
    def __init__(
        self,
        config: AgentConfig,
        session_store: SessionStore,
        meta_store: SessionMetaStore,
        workspace: WorkspaceManager,
        compaction_manager: ContextCompactionManager | None = None,
        embedding_index_manager: EmbeddingIndexManager | None = None,
    ) -> None:
        self.config = config
        self.session_store = session_store
        self.meta_store = meta_store
        self.workspace = workspace
        self.compaction_manager = compaction_manager
        self.embedding_index_manager = embedding_index_manager
        self._last_idle_run = 0.0

    def run_turn_heartbeat(self, session: SessionState) -> None:
        if not self.config.enable_heartbeat:
            return

        meta = self.meta_store.get_or_create(session.session_id)
        session.summary = rebuild_session_summary(session, max_chars=self.config.session_summary_max_chars)
        now = utc_now_iso()
        meta.last_activity_at = now
        meta.last_heartbeat_at = now
        meta.cached_summary = session.summary
        meta.dirty_summary = True
        meta.dirty_daily_memory = True
        meta.dirty_long_term = True
        meta.dirty_compaction = self.compaction_manager is not None
        meta.dirty_embedding = self.embedding_index_manager is not None
        self.meta_store.save(meta)

    def maybe_run_idle_heartbeat(self, max_sessions: int | None = None) -> int:
        if not self.config.enable_heartbeat:
            return 0
        now = time.monotonic()
        if self._last_idle_run and now - self._last_idle_run < self.config.idle_heartbeat_interval_seconds:
            return 0
        processed = self.run_idle_heartbeat_once(max_sessions=max_sessions)
        self._last_idle_run = now
        return processed

    def run_idle_heartbeat_once(self, max_sessions: int | None = None) -> int:
        if not self.config.enable_heartbeat:
            return 0

        processed = 0
        workspace_synced = False
        embedding_processed = 0
        embedding_limit = max(self.config.embedding_index_max_sessions_per_idle_run, 0)
        for meta in self.meta_store.list_dirty_sessions():
            if max_sessions is not None and processed >= max_sessions:
                break

            session = self.session_store.load_session(meta.session_id)
            if session is None:
                continue

            turn_records = self.session_store.load_turn_records(meta.session_id)
            new_records = [
                record for record in turn_records if int(record.get("turn_index", 0)) > meta.last_memory_turn_index
            ]

            if meta.dirty_summary:
                session.summary = rebuild_session_summary(session, max_chars=self.config.session_summary_max_chars)
                meta.cached_summary = session.summary
                meta.dirty_summary = False

            if meta.dirty_daily_memory and len(new_records) >= self.config.daily_memory_min_turns:
                daily_candidates = build_daily_memory_candidates(new_records)
                self.workspace.append_daily_entries(daily_candidates)
                meta.dirty_daily_memory = False

            if meta.dirty_long_term:
                long_term_candidates = build_long_term_candidates(
                    turn_records,
                    repeat_threshold=self.config.long_term_repeat_threshold,
                )
                self.workspace.append_long_term_entries(long_term_candidates)
                meta.dirty_long_term = False

            if self.embedding_index_manager is not None and not workspace_synced:
                self._sync_workspace_embedding_index()
                workspace_synced = True

            if meta.dirty_compaction and self.compaction_manager is not None:
                self.meta_store.save(meta)
                self.compaction_manager.compact_session(session)
                meta = self.meta_store.get_or_create(session.session_id)

            if (
                meta.dirty_embedding
                and self.embedding_index_manager is not None
                and (embedding_limit <= 0 or embedding_processed < embedding_limit)
            ):
                self._sync_session_embedding_indexes(session.session_id, meta)
                meta = self.meta_store.get_or_create(session.session_id)
                embedding_processed += 1

            if new_records:
                meta.last_memory_turn_index = max(int(record.get("turn_index", 0)) for record in new_records)

            meta.last_heartbeat_at = utc_now_iso()
            self.meta_store.save(meta)
            processed += 1

        if self.embedding_index_manager is not None and not workspace_synced:
            self._sync_workspace_embedding_index()

        return processed

    def _sync_workspace_embedding_index(self) -> None:
        if self.embedding_index_manager is None or not self.config.embedding_index_enabled:
            return
        chunks = self.workspace._iter_memory_chunks()
        items = [
            EmbeddingIndexItem.create(
                source_id=chunk.source_id,
                source_kind=chunk.source_kind,
                source_path=chunk.source_path,
                target=chunk.target,
                text=chunk.text,
            )
            for chunk in chunks
        ]
        self.embedding_index_manager.refresh_embeddings(
            self.embedding_index_manager.workspace_shard_path(),
            items,
        )

    def _sync_session_embedding_indexes(self, session_id: str, meta) -> None:
        if self.embedding_index_manager is None or not self.config.embedding_index_enabled:
            return

        session = self.session_store.load_session(session_id)
        if session is None:
            return

        recent_messages = session.messages[-self.config.archive_session_search_messages_per_session :]
        base_index = max(len(session.messages) - len(recent_messages), 0)
        session_items = [
            EmbeddingIndexItem.create(
                source_id=f"session:{session_id}:{item.role}:{base_index + offset}",
                source_kind="session_archive",
                source_path=session_id,
                target=f"session:{session_id}:{item.role}:{base_index + offset}",
                text=item.content,
            )
            for offset, item in enumerate(recent_messages, start=1)
        ]
        session_shard = self.embedding_index_manager.session_shard_path(session_id)
        self.embedding_index_manager.refresh_embeddings(session_shard, session_items)
        meta.latest_session_embedding_path = str(session_shard)

        if self.compaction_manager is not None:
            artifact = self.compaction_manager.load_or_build_artifact_for_session_id(session_id)
            if artifact is not None:
                compaction_items = [
                    EmbeddingIndexItem.create(
                        source_id=chunk["target"],
                        source_kind=chunk["source_kind"],
                        source_path=chunk["source_path"],
                        target=chunk["target"],
                        text=chunk["text"],
                    )
                    for chunk in self.compaction_manager.search_chunks(artifact)
                ]
                compaction_shard = self.embedding_index_manager.compaction_shard_path(session_id)
                self.embedding_index_manager.refresh_embeddings(compaction_shard, compaction_items)
                meta.latest_compaction_embedding_path = str(compaction_shard)

        current_turn_index = max(len(session.messages) // 2, 1)
        meta.last_embedding_turn_index = current_turn_index
        meta.last_embedding_at = utc_now_iso()
        meta.dirty_embedding = False
        self.meta_store.save(meta)
