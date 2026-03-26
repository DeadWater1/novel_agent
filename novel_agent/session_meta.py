from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SessionMeta:
    session_id: str
    last_activity_at: str = ""
    last_heartbeat_at: str = ""
    last_memory_turn_index: int = 0
    last_memory_flush_turn_index: int = 0
    last_compaction_turn_index: int = 0
    dirty_summary: bool = False
    dirty_daily_memory: bool = False
    dirty_long_term: bool = False
    dirty_compaction: bool = False
    cached_summary: str = ""
    cached_compact_summary: str = ""
    latest_transcript_path: str = ""
    latest_compaction_path: str = ""
    recent_content_references: list[dict[str, Any]] = field(default_factory=list)


class SessionMetaStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def bootstrap(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def meta_path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.meta.json"

    def load(self, session_id: str) -> SessionMeta | None:
        path = self.meta_path(session_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SessionMeta(**payload)

    def get_or_create(self, session_id: str) -> SessionMeta:
        self.bootstrap()
        meta = self.load(session_id)
        if meta is not None:
            return meta
        meta = SessionMeta(session_id=session_id)
        self.save(meta)
        return meta

    def save(self, meta: SessionMeta) -> None:
        self.bootstrap()
        path = self.meta_path(meta.session_id)
        path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def touch_activity(self, session_id: str) -> SessionMeta:
        meta = self.get_or_create(session_id)
        meta.last_activity_at = utc_now_iso()
        self.save(meta)
        return meta

    def list_all(self) -> list[SessionMeta]:
        self.bootstrap()
        results: list[SessionMeta] = []
        for path in sorted(self.root.glob("*.meta.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            results.append(SessionMeta(**payload))
        return results

    def list_dirty_sessions(self) -> list[SessionMeta]:
        results = []
        for meta in self.list_all():
            if meta.dirty_summary or meta.dirty_daily_memory or meta.dirty_long_term or meta.dirty_compaction:
                results.append(meta)
        return results
