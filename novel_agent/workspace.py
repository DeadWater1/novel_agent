from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from .config import AgentConfig


WORKSPACE_FILE_ORDER = (
    "agent.md",
    "tools.md",
    "soul.md",
    "identity.md",
    "user.md",
    "memory.md",
)


WORKSPACE_DEFAULTS = {
    "agent.md": "# Agent\n",
    "tools.md": "# Tools\n",
    "soul.md": "# Soul\n",
    "identity.md": "# Identity\n",
    "user.md": "# User\n",
    "memory.md": "# Long-Term Memory\n",
}


class WorkspaceManager:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.root = Path(config.workspace_root)
        self.daily_root = self.root / "memory"

    def bootstrap(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.daily_root.mkdir(parents=True, exist_ok=True)
        for name, content in WORKSPACE_DEFAULTS.items():
            path = self.root / name
            if not path.exists():
                path.write_text(content, encoding="utf-8")
        self.ensure_daily_file()

    def ensure_daily_file(self, day: date | None = None) -> Path:
        target_day = day or date.today()
        path = self.daily_root / f"{target_day.isoformat()}.md"
        if not path.exists():
            path.write_text(f"# Daily Memory - {target_day.isoformat()}\n\n", encoding="utf-8")
        return path

    def read_file(self, relative_name: str) -> str:
        path = self.root / relative_name
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def load_workspace_docs(self, session_summary: str = "") -> str:
        parts: list[str] = []
        for name in WORKSPACE_FILE_ORDER:
            content = self.read_file(name)
            if content:
                parts.append(content)
        for content in self.load_recent_daily_memories():
            if content:
                parts.append(content)
        if session_summary.strip():
            parts.append(f"# Session Summary\n\n{session_summary.strip()}")
        return "\n\n".join(parts).strip()

    def load_recent_daily_memories(self) -> list[str]:
        results: list[str] = []
        today = date.today()
        for offset in range(self.config.daily_memory_lookback_days):
            target = today - timedelta(days=offset)
            path = self.ensure_daily_file(target)
            results.append(path.read_text(encoding="utf-8").strip())
        return results

    def append_daily_entries(self, entries: list[str], day: date | None = None) -> None:
        clean_entries = [entry.strip() for entry in entries if entry and entry.strip()]
        if not clean_entries:
            return
        path = self.ensure_daily_file(day)
        current = path.read_text(encoding="utf-8")
        with path.open("w", encoding="utf-8") as handle:
            handle.write(current.rstrip() + "\n\n")
            for entry in clean_entries:
                handle.write(f"- {entry}\n")

    def append_long_term_entries(self, entries: list[str]) -> None:
        clean_entries = [entry.strip() for entry in entries if entry and entry.strip()]
        if not clean_entries:
            return
        path = self.root / "memory.md"
        current = path.read_text(encoding="utf-8") if path.exists() else self.config.long_term_memory_header
        with path.open("w", encoding="utf-8") as handle:
            handle.write(current.rstrip() + "\n\n")
            for entry in clean_entries:
                handle.write(f"- {entry}\n")
