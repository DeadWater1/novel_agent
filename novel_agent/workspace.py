from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

from .config import AgentConfig


WORKSPACE_FILE_ORDER = (
    "agent.md",
    "tools.md",
    "soul.md",
    "identity.md",
    "user.md",
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
        parts.append(self.build_memory_access_note())
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
        clean_entries = self._dedupe_entries(entries, self.ensure_daily_file(day))
        if not clean_entries:
            return
        path = self.ensure_daily_file(day)
        current = path.read_text(encoding="utf-8")
        with path.open("w", encoding="utf-8") as handle:
            handle.write(current.rstrip() + "\n\n")
            for entry in clean_entries:
                handle.write(f"- {entry}\n")

    def append_long_term_entries(self, entries: list[str]) -> None:
        clean_entries = self._dedupe_entries(entries, self.root / "memory.md")
        if not clean_entries:
            return
        path = self.root / "memory.md"
        current = path.read_text(encoding="utf-8") if path.exists() else self.config.long_term_memory_header
        with path.open("w", encoding="utf-8") as handle:
            handle.write(current.rstrip() + "\n\n")
            for entry in clean_entries:
                handle.write(f"- {entry}\n")

    def _dedupe_entries(self, entries: list[str], path: Path) -> list[str]:
        existing = self._read_bullet_entries(path)
        clean_entries: list[str] = []
        seen = set(existing)
        for entry in entries:
            cleaned = entry.strip()
            if not cleaned or cleaned in seen:
                continue
            clean_entries.append(cleaned)
            seen.add(cleaned)
        return clean_entries

    def _read_bullet_entries(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        entries = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                entries.add(stripped[2:].strip())
        return entries

    def build_memory_access_note(self) -> str:
        targets = ["long_term", "daily_latest", "today", "yesterday"]
        recent_days = []
        today = date.today()
        for offset in range(self.config.daily_memory_lookback_days):
            target = today - timedelta(days=offset)
            recent_days.append(f"daily:{target.isoformat()}")
        if recent_days:
            targets.extend(recent_days)
        targets_text = ", ".join(targets)
        return (
            "# Memory Access\n\n"
            "长期记忆和日记式记忆默认不会被完整注入上下文。\n"
            "需要记忆内容时，请使用 memory_search 或 memory_get 工具。\n"
            f"当前常用 memory_get target: {targets_text}"
        )

    def memory_search(self, query: str, max_results: int | None = None) -> list[dict[str, str]]:
        clean_query = query.strip()
        if not clean_query:
            return []

        results: list[dict[str, str]] = []
        limit = max_results or self.config.memory_search_max_results
        for source_id, content in self._iter_memory_sources():
            score, snippet = self._score_memory_match(clean_query, content)
            if score <= 0:
                continue
            results.append(
                {
                    "source_id": source_id,
                    "score": str(score),
                    "snippet": snippet,
                }
            )
        results.sort(key=lambda item: int(item["score"]), reverse=True)
        return results[:limit]

    def memory_get(self, target: str) -> dict[str, str]:
        clean_target = target.strip()
        if clean_target == "long_term":
            path = self.root / "memory.md"
        elif clean_target in {"daily_latest", "today"}:
            path = self.ensure_daily_file()
        elif clean_target == "yesterday":
            path = self.ensure_daily_file(date.today() - timedelta(days=1))
        elif clean_target.startswith("daily:"):
            day_text = clean_target.split(":", 1)[1]
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_text):
                raise ValueError("invalid_daily_target")
            year, month, day = (int(item) for item in day_text.split("-"))
            path = self.ensure_daily_file(date(year, month, day))
        else:
            raise ValueError("unsupported_memory_target")

        return {
            "target": clean_target,
            "source_path": str(path),
            "content": path.read_text(encoding="utf-8").strip() if path.exists() else "",
        }

    def _iter_memory_sources(self) -> list[tuple[str, str]]:
        sources: list[tuple[str, str]] = []
        long_term = self.root / "memory.md"
        if long_term.exists():
            sources.append(("long_term", long_term.read_text(encoding="utf-8").strip()))
        for path in sorted(self.daily_root.glob("*.md"), reverse=True):
            sources.append((f"daily:{path.stem}", path.read_text(encoding="utf-8").strip()))
        return sources

    def _score_memory_match(self, query: str, content: str) -> tuple[int, str]:
        text = content.strip()
        if not text:
            return 0, ""

        lowered_text = text.lower()
        tokens = [query.strip()]
        tokens.extend(
            token
            for token in re.split(r"[\s,，。；;、/]+", query)
            if token and token.strip() and token.strip() != query.strip()
        )
        unique_tokens = []
        seen = set()
        for token in tokens:
            cleaned = token.strip().lower()
            if not cleaned or cleaned in seen:
                continue
            unique_tokens.append(cleaned)
            seen.add(cleaned)

        score = 0
        best_token = ""
        best_index = -1
        for token in unique_tokens:
            idx = lowered_text.find(token)
            if idx == -1:
                continue
            score += max(1, len(token))
            if best_index == -1 or idx < best_index:
                best_token = token
                best_index = idx

        if score <= 0:
            return 0, ""

        start = max(0, best_index - 80)
        end = min(len(text), best_index + max(len(best_token), 1) + 120)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return score, snippet
