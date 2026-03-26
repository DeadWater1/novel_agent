from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .config import AgentConfig
from .search_utils import extract_snippet, format_line_target, hybrid_search_score, mmr_rerank, recency_multiplier


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

MEMORY_TARGET_PATTERN = re.compile(r"^(?P<base>[^#]+?)(?:#L(?P<start>\d+)(?:-L?(?P<end>\d+))?)?$")
MEMORY_CHUNK_MAX_CHARS = 520
MEMORY_CHUNK_WINDOW_LINES = 10
MEMORY_CHUNK_OVERLAP_LINES = 2
MEMORY_SEARCH_CANDIDATE_MULTIPLIER = 4
MEMORY_SNIPPET_MAX_CHARS = 240


@dataclass(slots=True)
class MemoryChunk:
    source_id: str
    source_kind: str
    source_path: str
    target: str
    text: str
    line_start: int
    line_end: int


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

    def load_workspace_docs(self, session_summary: str = "", recalled_memory: str = "") -> str:
        parts: list[str] = []
        for name in WORKSPACE_FILE_ORDER:
            content = self.read_file(name)
            if content:
                parts.append(content)
        parts.append(self.build_memory_access_note())
        if recalled_memory.strip():
            parts.append(f"# Recalled Memory\n\n{recalled_memory.strip()}")
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
            "系统会自动召回一小部分高相关记忆，但需要更完整的内容时仍应使用 memory_search 或 memory_get。\n"
            "memory_search 支持 lookup 与 recap 两种模式，会返回 snippet 或 summary_preview、target、source_path 等信息；如果结果不够，请把返回的 target 直接交给 memory_get。\n"
            "memory_get 支持 long_term、daily_latest、today、yesterday、daily:YYYY-MM-DD，"
            "也支持在其后附加 #L起始行-结束行，例如 daily:2026-03-26#L3-L12；"
            "session 检索结果也可以直接读取，例如 session:latest_compress、session:SESSION_ID:assistant:4、session_compact:SESSION_ID、session_compact:SESSION_ID#compression_history:0、session_compact:time_window:1-1#summary、content_ref:latest。"
            "memory_get 默认直接交付全文；如果只需要内部观察，可以显式传 delivery_mode=observe。\n"
            f"当前常用 memory_get target: {targets_text}"
        )

    def memory_search(self, query: str, max_results: int | None = None) -> list[dict[str, str | float | int]]:
        clean_query = query.strip()
        if not clean_query:
            return []

        limit = max_results or self.config.memory_search_max_results
        candidates: list[dict[str, str | float | int]] = []
        for chunk in self._iter_memory_chunks():
            score = hybrid_search_score(clean_query, chunk.text)
            if score <= 0:
                continue
            score *= recency_multiplier(chunk.source_id)
            candidates.append(
                {
                    "source_id": chunk.source_id,
                    "source_kind": chunk.source_kind,
                    "source_path": chunk.source_path,
                    "target": chunk.target,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "score": score,
                    "snippet": extract_snippet(chunk.text, clean_query, max_chars=MEMORY_SNIPPET_MAX_CHARS),
                    "text": chunk.text,
                }
            )

        if not candidates:
            return []

        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        reranked = mmr_rerank(
            candidates[: max(limit * MEMORY_SEARCH_CANDIDATE_MULTIPLIER, limit)],
            limit=limit,
            text_key="text",
        )
        for item in reranked:
            item.pop("text", None)
        return reranked

    def memory_get(self, target: str) -> dict[str, str | int]:
        clean_target = target.strip()
        base_target, line_start, line_end = self._parse_memory_target(clean_target)
        path, resolved_base = self._resolve_memory_target(base_target)
        content = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        if line_start is not None and content:
            content = self._slice_lines(content, line_start=line_start, line_end=line_end or line_start)
        resolved_target = (
            format_line_target(resolved_base, line_start, line_end or line_start) if line_start is not None else resolved_base
        )
        return {
            "target": clean_target,
            "resolved_target": resolved_target,
            "source_path": str(path),
            "content": content,
            "line_start": line_start or 1,
            "line_end": line_end or line_start or max(len(content.splitlines()), 1),
        }

    def _parse_memory_target(self, target: str) -> tuple[str, int | None, int | None]:
        match = MEMORY_TARGET_PATTERN.fullmatch(target)
        if not match:
            raise ValueError("unsupported_memory_target")
        base_target = str(match.group("base") or "").strip()
        line_start = int(match.group("start")) if match.group("start") else None
        line_end = int(match.group("end")) if match.group("end") else line_start
        return base_target, line_start, line_end

    def _resolve_memory_target(self, base_target: str) -> tuple[Path, str]:
        clean_target = base_target.strip()
        if clean_target == "long_term":
            return self.root / "memory.md", "long_term"
        if clean_target in {"daily_latest", "today"}:
            path = self.ensure_daily_file()
            return path, f"daily:{path.stem}"
        if clean_target == "yesterday":
            path = self.ensure_daily_file(date.today() - timedelta(days=1))
            return path, f"daily:{path.stem}"
        if clean_target.startswith("daily:"):
            day_text = clean_target.split(":", 1)[1]
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_text):
                raise ValueError("invalid_daily_target")
            year, month, day = (int(item) for item in day_text.split("-"))
            path = self.ensure_daily_file(date(year, month, day))
            return path, clean_target
        raise ValueError("unsupported_memory_target")

    def _iter_memory_chunks(self) -> list[MemoryChunk]:
        chunks: list[MemoryChunk] = []
        for source_id, source_kind, source_path, content in self._iter_memory_sources():
            chunks.extend(self._chunk_memory_text(source_id, source_kind, source_path, content))
        return chunks

    def _iter_memory_sources(self) -> list[tuple[str, str, str, str]]:
        sources: list[tuple[str, str, str, str]] = []
        long_term = self.root / "memory.md"
        if long_term.exists():
            sources.append(("long_term", "long_term", "memory.md", long_term.read_text(encoding="utf-8").strip()))
        for path in sorted(self.daily_root.glob("*.md"), reverse=True):
            sources.append(
                (
                    f"daily:{path.stem}",
                    "daily",
                    str(Path("memory") / path.name),
                    path.read_text(encoding="utf-8").strip(),
                )
            )
        return sources

    def _chunk_memory_text(
        self,
        source_id: str,
        source_kind: str,
        source_path: str,
        content: str,
    ) -> list[MemoryChunk]:
        stripped = content.strip()
        if not stripped:
            return []

        lines = stripped.splitlines()
        blocks: list[tuple[int, int, str, str]] = []
        current_lines: list[tuple[int, str]] = []
        current_heading = ""
        block_heading = ""

        def flush_block() -> None:
            nonlocal current_lines, block_heading
            if not current_lines:
                return
            line_start = current_lines[0][0]
            line_end = current_lines[-1][0]
            raw_text = "\n".join(line for _, line in current_lines).strip()
            if raw_text:
                blocks.extend(self._split_block(block_heading, raw_text, line_start, line_end))
            current_lines = []
            block_heading = current_heading

        for line_number, line in enumerate(lines, start=1):
            stripped_line = line.strip()
            if stripped_line.startswith("#"):
                flush_block()
                current_heading = stripped_line.lstrip("#").strip()
                block_heading = current_heading
                continue
            if not stripped_line:
                flush_block()
                continue
            if not current_lines:
                block_heading = current_heading
            current_lines.append((line_number, line.rstrip()))
        flush_block()

        chunks: list[MemoryChunk] = []
        for line_start, line_end, heading, body in blocks:
            base_target = source_id
            target = format_line_target(base_target, line_start, line_end)
            chunk_text = body if not heading or body.startswith(heading) else f"{heading}\n{body}"
            chunks.append(
                MemoryChunk(
                    source_id=source_id,
                    source_kind=source_kind,
                    source_path=source_path,
                    target=target,
                    text=chunk_text.strip(),
                    line_start=line_start,
                    line_end=line_end,
                )
            )
        return chunks

    def _split_block(self, heading: str, body: str, line_start: int, line_end: int) -> list[tuple[int, int, str, str]]:
        clean_body = body.strip()
        if len(clean_body) <= MEMORY_CHUNK_MAX_CHARS and (line_end - line_start + 1) <= MEMORY_CHUNK_WINDOW_LINES:
            return [(line_start, line_end, heading, clean_body)]

        lines = clean_body.splitlines()
        if not lines:
            return []

        windows: list[tuple[int, int, str, str]] = []
        start_index = 0
        while start_index < len(lines):
            end_index = min(start_index + MEMORY_CHUNK_WINDOW_LINES, len(lines))
            window_lines = lines[start_index:end_index]
            if not window_lines:
                break
            window_start = line_start + start_index
            window_end = window_start + len(window_lines) - 1
            windows.append((window_start, window_end, heading, "\n".join(window_lines).strip()))
            if end_index >= len(lines):
                break
            start_index = max(end_index - MEMORY_CHUNK_OVERLAP_LINES, start_index + 1)
        return windows

    def _slice_lines(self, content: str, *, line_start: int, line_end: int) -> str:
        lines = content.splitlines()
        start_index = max(line_start - 1, 0)
        end_index = min(line_end, len(lines))
        if start_index >= len(lines) or start_index >= end_index:
            return ""
        return "\n".join(lines[start_index:end_index]).strip()
