from __future__ import annotations

import json
from pathlib import Path

from novel_agent.memory import SessionStore


def test_session_store_appends_event_transcript_and_updates_index(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions", summary_max_chars=300)
    session = store.create_session()
    store.append_events(
        session,
        [
            {"turn_index": 1, "event_type": "user_message", "role": "user", "content": "第一轮用户输入"},
            {"turn_index": 1, "event_type": "assistant_message", "role": "assistant", "content": "第一轮助手回复"},
        ],
    )
    store.append_events(
        session,
        [
            {"turn_index": 2, "event_type": "user_message", "role": "user", "content": "第二轮用户输入"},
            {
                "turn_index": 2,
                "event_type": "tool_call",
                "tool_name": "memory_search",
                "tool_args": {"query": "人物关系"},
            },
            {
                "turn_index": 2,
                "event_type": "tool_result",
                "tool_name": "memory_search",
                "content": "long_term: 用户偏好压缩时保留人物关系",
                "payload": {"tool_trace": {"requested_tool": "memory_search", "status": "ok"}},
            },
            {"turn_index": 2, "event_type": "assistant_message", "role": "assistant", "content": "第二轮助手回复"},
        ],
    )

    path = store.session_path(session.session_id)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    assert first["event_type"] == "user_message"
    assert first["content"] == "第一轮用户输入"
    assert last["event_type"] == "assistant_message"
    assert last["content"] == "第二轮助手回复"

    index = json.loads(store.index_path().read_text(encoding="utf-8"))
    assert index[session.session_id]["event_count"] == 6
    assert index[session.session_id]["last_turn_index"] == 2


def test_session_store_restores_session_from_event_transcript(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions", summary_max_chars=300)
    session = store.create_session()
    store.append_events(
        session,
        [
            {"turn_index": 1, "event_type": "user_message", "role": "user", "content": "林秋出场"},
            {
                "turn_index": 1,
                "event_type": "agent_decision",
                "payload": {"user_goal": "分析人物", "action": "direct_reply"},
            },
            {
                "turn_index": 1,
                "event_type": "tool_result",
                "tool_name": "memory_search",
                "content": "今天讨论了林秋",
                "payload": {"tool_trace": {"requested_tool": "memory_search", "status": "ok"}},
                "thinking": "调试信息",
            },
            {"turn_index": 1, "event_type": "assistant_message", "role": "assistant", "content": "这是人物分析"},
        ],
    )

    restored = store.load_session(session.session_id)
    assert restored is not None
    assert restored.session_id == session.session_id
    assert restored.chat_history() == [
        {"role": "user", "content": "林秋出场"},
        {"role": "assistant", "content": "这是人物分析"},
    ]
    assert restored.last_decision == {"user_goal": "分析人物", "action": "direct_reply"}
    assert restored.last_tool_trace == {"requested_tool": "memory_search", "status": "ok"}
    assert restored.last_thinking == "调试信息"
    assert "用户: 林秋出场" in restored.summary
