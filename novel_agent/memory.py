from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Message:
    role: str
    content: str


@dataclass(slots=True)
class SessionState:
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
