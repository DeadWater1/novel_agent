from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryFact(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: str
    content: str
    confidence: float = 0.8
    source_session_id: str = ""
    turn_index: int = 0
    created_at: str = Field(default_factory=utc_now_iso)
    tags: list[str] = Field(default_factory=list)
    date: str = Field(default_factory=lambda: date.today().isoformat())


class MemoryContext(BaseModel):
    user_preferences: list[str] = Field(default_factory=list)
    story_constraints: list[str] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    updated_at: str = ""


class MemoryDigest(BaseModel):
    date: str
    generated_at: str = Field(default_factory=utc_now_iso)
    lines: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class StructuredMemoryStore:
    root: Path
    digests_root: Path = Path()

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.digests_root = self.root / "digests"

    def bootstrap(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.digests_root.mkdir(parents=True, exist_ok=True)
        if not self.context_path().exists():
            self.save_context(MemoryContext())
        if not self.facts_path().exists():
            self.facts_path().write_text("", encoding="utf-8")

    def context_path(self) -> Path:
        return self.root / "context.json"

    def facts_path(self) -> Path:
        return self.root / "facts.jsonl"

    def digest_path(self, day: date | str) -> Path:
        if isinstance(day, date):
            day_text = day.isoformat()
        else:
            day_text = str(day)
        return self.digests_root / f"{day_text}.md"

    def load_context(self) -> MemoryContext:
        self.bootstrap()
        path = self.context_path()
        if not path.exists():
            return MemoryContext()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return MemoryContext.model_validate(payload)

    def save_context(self, context: MemoryContext) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.digests_root.mkdir(parents=True, exist_ok=True)
        context.updated_at = utc_now_iso()
        self.context_path().write_text(context.model_dump_json(indent=2), encoding="utf-8")

    def merge_context_entries(self, *, section: str, entries: list[str]) -> list[str]:
        clean_entries = [item.strip() for item in entries if item and item.strip()]
        if not clean_entries:
            return []
        context = self.load_context()
        current = list(getattr(context, section, []))
        added: list[str] = []
        for entry in clean_entries:
            if entry in current:
                continue
            current.append(entry)
            added.append(entry)
        setattr(context, section, current)
        self.save_context(context)
        return added

    def list_facts(self) -> list[MemoryFact]:
        self.bootstrap()
        results: list[MemoryFact] = []
        path = self.facts_path()
        if not path.exists():
            return results
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                results.append(MemoryFact.model_validate_json(line))
        return results

    def append_facts(self, facts: list[MemoryFact]) -> list[MemoryFact]:
        clean_facts = [item for item in facts if item.content.strip()]
        if not clean_facts:
            return []
        existing = {(item.kind, item.content.strip()) for item in self.list_facts()}
        written: list[MemoryFact] = []
        with self.facts_path().open("a", encoding="utf-8") as handle:
            for fact in clean_facts:
                key = (fact.kind, fact.content.strip())
                if key in existing:
                    continue
                existing.add(key)
                handle.write(fact.model_dump_json() + "\n")
                written.append(fact)
        return written

    def get_fact(self, fact_id: str) -> MemoryFact | None:
        for item in self.list_facts():
            if item.id == fact_id:
                return item
        return None

    def load_digest(self, day: date | str) -> MemoryDigest:
        self.bootstrap()
        path = self.digest_path(day)
        if not path.exists():
            day_text = day.isoformat() if isinstance(day, date) else str(day)
            return MemoryDigest(date=day_text, lines=[])
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                lines.append(stripped[2:].strip())
        day_text = path.stem
        return MemoryDigest(date=day_text, lines=lines)

    def append_digest_entries(self, entries: list[str], day: date | None = None) -> list[str]:
        self.bootstrap()
        target_day = day or date.today()
        digest = self.load_digest(target_day)
        clean_entries = [item.strip() for item in entries if item and item.strip()]
        added: list[str] = []
        for entry in clean_entries:
            if entry in digest.lines:
                continue
            digest.lines.append(entry)
            added.append(entry)
        self.save_digest(digest)
        return added

    def save_digest(self, digest: MemoryDigest) -> None:
        self.bootstrap()
        path = self.digest_path(digest.date)
        lines = [f"# Memory Digest - {digest.date}", "", f"generated_at: {digest.generated_at}", ""]
        lines.extend(f"- {item}" for item in digest.lines)
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def latest_digest(self) -> MemoryDigest:
        self.bootstrap()
        digests = sorted(self.digests_root.glob("*.md"), reverse=True)
        if not digests:
            return self.load_digest(date.today())
        return self.load_digest(digests[0].stem)
