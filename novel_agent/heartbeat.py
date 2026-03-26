from __future__ import annotations

import time

from .compaction import ContextCompactionManager
from .config import AgentConfig
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
    ) -> None:
        self.config = config
        self.session_store = session_store
        self.meta_store = meta_store
        self.workspace = workspace
        self.compaction_manager = compaction_manager
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

            if meta.dirty_compaction and self.compaction_manager is not None:
                self.meta_store.save(meta)
                self.compaction_manager.compact_session(session)
                meta = self.meta_store.get_or_create(session.session_id)

            if new_records:
                meta.last_memory_turn_index = max(int(record.get("turn_index", 0)) for record in new_records)

            meta.last_heartbeat_at = utc_now_iso()
            self.meta_store.save(meta)
            processed += 1

        return processed
