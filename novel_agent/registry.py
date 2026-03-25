from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ToolSpec:
    name: str
    description: str
    required_args: tuple[str, ...]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str | None) -> ToolSpec | None:
        if not name:
            return None
        return self._tools.get(name)

    def is_registered(self, name: str | None) -> bool:
        return self.get(name) is not None

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools.keys())


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="compress_chapter",
            description="Compress a novel chapter while preserving facts and relationships.",
            required_args=("raw_text",),
        )
    )
    registry.register(
        ToolSpec(
            name="memory_search",
            description="Search long-term and daily memory files for relevant novel context.",
            required_args=("query",),
        )
    )
    registry.register(
        ToolSpec(
            name="memory_get",
            description="Fetch a specific memory document, such as long_term or a daily memory file.",
            required_args=("target",),
        )
    )
    return registry
