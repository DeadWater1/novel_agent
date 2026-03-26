from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class Message:
    role: str
    content: str


@dataclass(slots=True)
class SessionState:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    last_decision: dict | None = None
    last_tool_trace: dict | None = None
    last_thinking: str = ""

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))

    def recent_messages(self, limit: int) -> list[dict[str, str]]:
        selected = self.messages[-limit:]
        return [{"role": item.role, "content": item.content} for item in selected]

    def chat_history(self) -> list[dict[str, str]]:
        return [{"role": item.role, "content": item.content} for item in self.messages]

    def refresh_summary(self, max_chars: int = 1200) -> None:
        tail = self.messages[-8:]
        rendered = []
        for item in tail:
            prefix = "用户" if item.role == "user" else "助手"
            rendered.append(f"{prefix}: {item.content.strip()}")
        summary = "\n".join(rendered).strip()
        if len(summary) > max_chars:
            summary = summary[-max_chars:]
        self.summary = summary


class SessionStore:
    def __init__(self, root: Path, summary_max_chars: int = 1200) -> None:
        self.root = Path(root)
        self.summary_max_chars = summary_max_chars

    def bootstrap(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        index_path = self.index_path()
        if not index_path.exists():
            index_path.write_text("{}\n", encoding="utf-8")

    def create_session(self) -> SessionState:
        self.bootstrap()
        return SessionState()

    def index_path(self) -> Path:
        return self.root / "sessions.json"

    def session_path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"

    def append_events(self, session: SessionState, events: list[dict[str, Any]]) -> None:
        self.bootstrap()
        if not events:
            return
        path = self.session_path(session.session_id)
        index = self._load_index()
        session_info = index.get(session.session_id, {})
        next_event_index = int(session_info.get("event_count", 0))
        with path.open("a", encoding="utf-8") as handle:
            for event in events:
                next_event_index += 1
                payload = dict(event)
                payload["session_id"] = session.session_id
                payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
                payload.setdefault("event_index", next_event_index)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        last_event = events[-1]
        index[session.session_id] = {
            "session_id": session.session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "event_count": next_event_index,
            "last_turn_index": int(last_event.get("turn_index", 0)),
            "last_event_type": str(last_event.get("event_type", "")),
        }
        self._save_index(index)

    def load_session(self, session_id: str) -> SessionState | None:
        path = self.session_path(session_id)
        if not path.exists():
            return None

        session = SessionState(session_id=session_id)
        records = self.load_events(session_id)
        if records and any("event_type" in record for record in records):
            for payload in records:
                event_type = str(payload.get("event_type", ""))
                if event_type == "user_message":
                    content = str(payload.get("content", "")).strip()
                    if content:
                        session.add_user_message(content)
                elif event_type == "assistant_message":
                    content = str(payload.get("content", "")).strip()
                    if content:
                        session.add_assistant_message(content)
                elif event_type == "agent_decision":
                    session.last_decision = payload.get("payload") or None
                elif event_type == "tool_result":
                    tool_trace = payload.get("payload", {}).get("tool_trace")
                    if tool_trace is not None:
                        session.last_tool_trace = tool_trace
                    session.last_thinking = str(payload.get("thinking", "")) or session.last_thinking
        else:
            for payload in records:
                user_message = str(payload.get("user_message", "")).strip()
                assistant_reply = str(payload.get("assistant_reply", "")).strip()
                if user_message:
                    session.add_user_message(user_message)
                if assistant_reply:
                    session.add_assistant_message(assistant_reply)
                session.last_decision = payload.get("decision") or None
                session.last_tool_trace = payload.get("tool_trace") or None
                session.last_thinking = str(payload.get("thinking", ""))

        session.refresh_summary(self.summary_max_chars)
        return session

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        path = self.session_path(session_id)
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def load_turn_records(self, session_id: str) -> list[dict[str, Any]]:
        events = self.load_events(session_id)
        if not events:
            return []
        if not any("event_type" in event for event in events):
            return events

        grouped: dict[int, dict[str, Any]] = {}
        for payload in events:
            turn_index = int(payload.get("turn_index", 0))
            if turn_index <= 0:
                continue
            current = grouped.setdefault(
                turn_index,
                {
                    "turn_index": turn_index,
                    "user_message": "",
                    "assistant_reply": "",
                    "action": "direct_reply",
                    "decision": {},
                    "tool_trace": {},
                    "timestamp": "",
                },
            )
            event_type = str(payload.get("event_type", ""))
            timestamp = str(payload.get("timestamp", "")).strip()
            if timestamp:
                current["timestamp"] = timestamp
            if event_type == "user_message":
                current["user_message"] = str(payload.get("content", "")).strip()
            elif event_type == "assistant_message":
                current["assistant_reply"] = str(payload.get("content", "")).strip()
            elif event_type == "agent_decision":
                decision = payload.get("payload") or {}
                current["decision"] = decision
                if decision.get("action") == "call_tool":
                    current["action"] = "call_tool"
            elif event_type == "tool_call":
                current["action"] = "call_tool"
                current["tool_trace"] = {
                    "requested_tool": payload.get("tool_name"),
                    "status": "started",
                    "tool_args": payload.get("tool_args") or {},
                }
            elif event_type == "tool_result":
                current["action"] = "call_tool"
                current["tool_trace"] = (payload.get("payload") or {}).get("tool_trace", current["tool_trace"])

        return [grouped[key] for key in sorted(grouped)]

    def load_latest_session(self) -> SessionState | None:
        self.bootstrap()
        index = self._load_index()
        if index:
            latest = max(index.values(), key=lambda item: item.get("updated_at", ""))
            return self.load_session(str(latest["session_id"]))
        files = sorted(self.root.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not files:
            return None
        return self.load_session(files[0].stem)

    def list_session_infos(self) -> list[dict[str, Any]]:
        self.bootstrap()
        index = self._load_index()
        if index:
            return sorted(index.values(), key=lambda item: item.get("updated_at", ""), reverse=True)

        files = sorted(self.root.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
        results: list[dict[str, Any]] = []
        for path in files:
            results.append(
                {
                    "session_id": path.stem,
                    "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "event_count": 0,
                    "last_turn_index": 0,
                    "last_event_type": "",
                }
            )
        return results

    def _load_index(self) -> dict[str, dict[str, Any]]:
        self.bootstrap()
        path = self.index_path()
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    def _save_index(self, payload: dict[str, dict[str, Any]]) -> None:
        self.bootstrap()
        self.index_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
