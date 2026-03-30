from __future__ import annotations

from dataclasses import dataclass

from .toolbox import (
    CompressChapterHandler,
    EmbeddingSimilarityHandler,
    MemoryGetHandler,
    MemorySearchHandler,
    ToolHandler,
)


@dataclass(slots=True, frozen=True)
class ToolDescriptor:
    name: str
    description: str
    required_args: tuple[str, ...]
    prompt_doc: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, handler: ToolHandler) -> None:
        self._tools[handler.name] = handler

    def get(self, name: str | None) -> ToolDescriptor | None:
        handler = self.get_handler(name)
        if handler is None:
            return None
        return ToolDescriptor(
            name=handler.name,
            description=handler.description,
            required_args=handler.required_args,
            prompt_doc=handler.render_prompt_doc(),
        )

    def get_handler(self, name: str | None) -> ToolHandler | None:
        if not name:
            return None
        return self._tools.get(name)

    def is_registered(self, name: str | None) -> bool:
        return self.get_handler(name) is not None

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools.keys())

    def descriptors(self) -> tuple[ToolDescriptor, ...]:
        return tuple(self.get(name) for name in self.names() if self.get(name) is not None)

    def render_prompt_docs(self) -> str:
        if not self._tools:
            return "## 可用工具\n\n当前没有可用工具。"
        sections = ["## 可用工具"]
        for name in self.names():
            handler = self._tools[name]
            sections.append(handler.render_prompt_doc())
        return "\n\n".join(sections).strip()

    def render_tool_name_text(self, separator: str = "、") -> str:
        if not self._tools:
            return "(none)"
        return separator.join(self.names())


def build_default_registry(enabled_tools: tuple[str, ...] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    handlers: tuple[ToolHandler, ...] = (
        CompressChapterHandler(),
        MemorySearchHandler(),
        MemoryGetHandler(),
        EmbeddingSimilarityHandler(),
    )
    enabled = set(enabled_tools) if enabled_tools is not None else None
    for handler in handlers:
        if enabled is not None and handler.name not in enabled:
            continue
        registry.register(handler)
    return registry
