from __future__ import annotations

from collections import Counter
from typing import Any

from .memory import SessionState


DAILY_TOPIC_PATTERNS = (
    ("人物", "今天讨论了小说人物关系"),
    ("剧情", "今天讨论了剧情推进"),
    ("设定", "今天讨论了小说设定"),
    ("世界观", "今天讨论了世界观设定"),
)


LONG_TERM_PREFERENCE_PATTERNS = (
    (("保留人物关系", "人物关系不变"), "用户偏好压缩时保留人物关系"),
    (("保留剧情顺序", "剧情顺序不变"), "用户偏好压缩时保留剧情顺序"),
    (("保留世界观", "世界观不变"), "用户偏好压缩时保留世界观设定"),
    (("第一人称",), "用户偏好保留第一人称叙事视角"),
    (("第三人称",), "用户偏好保留第三人称叙事视角"),
)


def rebuild_session_summary(session: SessionState, max_chars: int = 1200) -> str:
    total_turns = len(session.messages) // 2
    conversation_focus = []
    joined = "\n".join(item.content for item in session.messages[-8:])

    if "压缩" in joined or "compress_chapter" in joined:
        conversation_focus.append("当前重点: 章节压缩")
    if "人物" in joined:
        conversation_focus.append("讨论焦点: 人物关系")
    if "剧情" in joined:
        conversation_focus.append("讨论焦点: 剧情推进")
    if "设定" in joined or "世界观" in joined:
        conversation_focus.append("讨论焦点: 设定/世界观")

    recent_lines = []
    for item in session.messages[-6:]:
        prefix = "用户" if item.role == "user" else "助手"
        recent_lines.append(f"{prefix}: {item.content.strip()}")

    parts = [f"当前会话已累计 {total_turns} 轮对话。"]
    parts.extend(conversation_focus)
    if recent_lines:
        parts.append("最近对话：")
        parts.extend(recent_lines)

    summary = "\n".join(parts).strip()
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary


def build_daily_memory_candidates(turn_records: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    if not turn_records:
        return candidates

    compression_count = 0
    for record in turn_records:
        user_message = str(record.get("user_message", ""))
        action = str(record.get("action", ""))
        tool_trace = record.get("tool_trace") or {}
        if action == "call_tool" and tool_trace.get("requested_tool") == "compress_chapter":
            compression_count += 1
        for keyword, sentence in DAILY_TOPIC_PATTERNS:
            if keyword in user_message and sentence not in candidates:
                candidates.append(sentence)

    if compression_count:
        candidates.append(f"今天执行了 {compression_count} 次章节压缩任务")

    if not candidates:
        candidates.append(f"今天新增了 {len(turn_records)} 轮小说相关交流")
    return candidates


def build_long_term_candidates(turn_records: list[dict[str, Any]], repeat_threshold: int = 2) -> list[str]:
    combined_text = "\n".join(
        f"{record.get('user_message', '')}\n{record.get('assistant_reply', '')}" for record in turn_records
    )
    if not combined_text.strip():
        return []

    counts: Counter[str] = Counter()
    for patterns, normalized in LONG_TERM_PREFERENCE_PATTERNS:
        for pattern in patterns:
            if pattern in combined_text:
                counts[normalized] += combined_text.count(pattern)

    candidates: list[str] = []
    for normalized, count in counts.items():
        if count >= repeat_threshold:
            candidates.append(normalized)
    return candidates
