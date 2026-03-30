from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .compaction import ContextCompactionManager
from .config import AgentConfig
from .memory import SessionState, SessionStore
from .schemas import CompressionRequest, EmbeddingSimilarityArgs, MemoryGetArgs, MemorySearchArgs
from .search_utils import extract_snippet
from .session_meta import SessionMetaStore
from .workspace import WorkspaceManager


class CompressChapterArgs(BaseModel):
    raw_text: str


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    observation_text: str
    payload: dict[str, Any]
    tool_trace: dict[str, Any]
    terminal: bool = False
    final_reply: str = ""
    thinking: str = ""


@dataclass(slots=True)
class ToolRuntimeContext:
    config: AgentConfig
    workspace: WorkspaceManager
    compression_backend: Any
    embedding_backend: Any
    embedding_index_manager: Any | None = None
    session_store: SessionStore | None = None
    meta_store: SessionMetaStore | None = None
    compaction_manager: ContextCompactionManager | None = None
    search_memory_sources: Any | None = None
    format_memory_search_results: Any | None = None
    memory_get: Any | None = None
    remember_recent_content_reference: Any | None = None
    truncate_memory_text: Any | None = None
    memory_observation_preview: Any | None = None


class ToolHandler:
    name: str = ""
    description: str = ""
    args_model: type[BaseModel] = BaseModel
    prompt_doc: str = ""

    @property
    def required_args(self) -> tuple[str, ...]:
        fields = getattr(self.args_model, "model_fields", {})
        return tuple(name for name, field in fields.items() if field.is_required())

    def parse_args(self, tool_args: dict[str, Any]) -> BaseModel:
        return self.args_model.model_validate(tool_args)

    def render_prompt_doc(self) -> str:
        return self.prompt_doc.strip()

    def execute(
        self,
        session: SessionState,
        parsed_args: BaseModel,
        runtime: ToolRuntimeContext,
    ) -> ToolExecutionResult:
        raise NotImplementedError


class CompressChapterHandler(ToolHandler):
    name = "compress_chapter"
    description = "Compress a novel chapter while preserving facts and relationships."
    args_model = CompressChapterArgs
    prompt_doc = """
## compress_chapter

- 名称：`compress_chapter`
- 作用：对小说章节进行压缩
- 输入：
  - `raw_text`: 原始章节文本，必填
- 说明：生成参数由系统配置统一管理，决策模型不应尝试覆盖
- 输出：
  - `compressed_text`
  - `thinking`（仅调试可见）
""".strip()

    def execute(
        self,
        session: SessionState,
        parsed_args: CompressChapterArgs,
        runtime: ToolRuntimeContext,
    ) -> ToolExecutionResult:
        request = CompressionRequest(
            raw_text=parsed_args.raw_text,
            max_new_tokens=runtime.config.compression_max_new_tokens,
            temperature=runtime.config.compression_temperature,
            seed=runtime.config.compression_seed,
            top_p=runtime.config.compression_top_p,
            top_k=runtime.config.compression_top_k,
            enable_thinking=runtime.config.compression_enable_thinking,
        )
        tool_result = runtime.compression_backend.compress(request)
        content = tool_result.compressed_text.strip()
        return ToolExecutionResult(
            tool_name=self.name,
            observation_text=content,
            payload={"compressed_text": content},
            tool_trace={"requested_tool": self.name, "status": "ok", "tool_args": request.model_dump()},
            terminal=True,
            final_reply=content,
            thinking=tool_result.thinking,
        )


class MemorySearchHandler(ToolHandler):
    name = "memory_search"
    description = "Search current-session, history-session, or time-window memory and return ranked evidence or recap targets."
    args_model = MemorySearchArgs
    prompt_doc = """
## memory_search

- 名称：`memory_search`
- 作用：检索长期记忆和日记式记忆中的相关小说信息
- 输入：
  - `query`: 查询关键词或问题，必填
  - `search_mode`: `"lookup"` 或 `"recap"`
  - `scope`: `"current_session"`、`"history_sessions"` 或 `"time_window"`
  - `time_scope`: 当 `scope="time_window"` 时可用，格式为 `{"from_days_ago": int, "to_days_ago": int}`
  - `max_results`: 可选，返回条数上限
- 输出：
  - `lookup` 返回 snippet 级证据
  - `recap` 返回 session/time-window 级摘要，并包含 `target`、`source_path`、`summary_preview`、`topics` 与 `time_range`
""".strip()

    def execute(
        self,
        session: SessionState,
        parsed_args: MemorySearchArgs,
        runtime: ToolRuntimeContext,
    ) -> ToolExecutionResult:
        assert callable(runtime.search_memory_sources)
        assert callable(runtime.format_memory_search_results)
        max_results = int(parsed_args.max_results or runtime.config.memory_search_max_results)
        results = runtime.search_memory_sources(
            session=session,
            query=parsed_args.query.strip(),
            max_results=max_results,
            search_mode=parsed_args.search_mode,
            scope=parsed_args.scope,
            time_scope=parsed_args.time_scope,
        )
        if results:
            observation = runtime.format_memory_search_results(results, search_mode=parsed_args.search_mode)
        else:
            observation = "memory_search 未找到相关记忆。"
        truncate = runtime.truncate_memory_text or (lambda text: text)
        return ToolExecutionResult(
            tool_name=self.name,
            observation_text=truncate(observation),
            payload={"search_args": parsed_args.model_dump(), "results": results},
            tool_trace={
                "requested_tool": self.name,
                "status": "ok",
                "result_count": len(results),
                "query": parsed_args.query,
                "search_mode": parsed_args.search_mode,
                "scope": parsed_args.scope,
            },
            terminal=False,
        )


class MemoryGetHandler(ToolHandler):
    name = "memory_get"
    description = "Fetch a memory target returned by memory_search or recent content references, optionally delivering full text immediately."
    args_model = MemoryGetArgs
    prompt_doc = """
## memory_get

- 名称：`memory_get`
- 作用：读取指定 target 对应的完整内容或完整摘要
- 输入：
  - `target`: 必填，可选值包括 `long_term`、`context:user_preferences`、`context:story_constraints`、`context:open_loops`、`fact:MEMORY_ID`、`digest:YYYY-MM-DD`，也兼容 `daily_latest`、`today`、`yesterday`、`daily:YYYY-MM-DD`、`session:...`、`session_compact:...`、`content_ref:latest`
  - `delivery_mode`: 可选，默认 `observe`；若明确要把全文直接交付给用户，可显式传 `deliver`
- 约束：当读取内容只是后续比较、判断、总结所需证据时，应保持 `observe`；只有在最后一步明确要把全文直接展示给用户时，才允许 `deliver`
- 输出：
  - target 对应的完整内容或摘要
""".strip()

    def execute(
        self,
        session: SessionState,
        parsed_args: MemoryGetArgs,
        runtime: ToolRuntimeContext,
    ) -> ToolExecutionResult:
        assert callable(runtime.memory_get)
        memory_doc = runtime.memory_get(session, target=parsed_args.target)
        content = str(memory_doc.get("content") or "").strip()
        resolved_target = str(memory_doc.get("resolved_target") or memory_doc["target"])
        source_path = str(memory_doc.get("source_path") or "")
        preview_fn = runtime.memory_observation_preview or (lambda text: text)
        observation_preview = preview_fn(content or "(empty)")
        observation = (
            f"target={resolved_target} source={source_path or '(empty)'} length={len(content)}\n"
            f"preview:\n{observation_preview}"
        )
        if callable(runtime.remember_recent_content_reference):
            runtime.remember_recent_content_reference(session, memory_doc)
        return ToolExecutionResult(
            tool_name=self.name,
            observation_text=observation,
            payload={**memory_doc, "delivery_mode": parsed_args.delivery_mode},
            tool_trace={
                "requested_tool": self.name,
                "status": "ok",
                "target": parsed_args.target,
                "resolved_target": resolved_target,
                "delivery_mode": parsed_args.delivery_mode,
            },
            terminal=parsed_args.delivery_mode == "deliver",
            final_reply=(content or "未找到对应内容。") if parsed_args.delivery_mode == "deliver" else "",
        )


class EmbeddingSimilarityHandler(ToolHandler):
    name = "embedding_similarity"
    description = "Compute semantic similarity scores between a query and one or more texts with the local embedding model."
    args_model = EmbeddingSimilarityArgs
    prompt_doc = """
## embedding_similarity

- 名称：`embedding_similarity`
- 作用：使用本地 embedding 模型比较 query 与候选文本的语义相似度
- 输入：
  - `query`: 必填
  - `text`: 可选，单条候选文本
  - `texts`: 可选，多条候选文本列表；与 `text` 二选一
  - `top_k`: 可选，仅保留最高分前 k 条
- 输出：
  - 相似度分数列表（用于后续判断，不直接等于最终答案）
""".strip()

    def execute(
        self,
        session: SessionState,
        parsed_args: EmbeddingSimilarityArgs,
        runtime: ToolRuntimeContext,
    ) -> ToolExecutionResult:
        candidates = [parsed_args.text] if parsed_args.text else list(parsed_args.texts)
        scores = runtime.embedding_backend.similarity_batch(parsed_args.query, candidates)
        ranked = [
            {
                "index": index + 1,
                "score": float(score),
                "text": text,
                "preview": extract_snippet(text, parsed_args.query, max_chars=220),
            }
            for index, (text, score) in enumerate(zip(candidates, scores))
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        if parsed_args.top_k is not None:
            ranked = ranked[: parsed_args.top_k]
        clean_query = parsed_args.query.strip()
        exact_match_indexes = [index + 1 for index, text in enumerate(candidates) if text.strip() == clean_query]
        observation_lines = [
            f"{index}. score={item['score']:.4f} length={len(item['text'])}\npreview: {item['preview']}"
            for index, item in enumerate(ranked, start=1)
        ]
        observation = "\n".join(observation_lines) if observation_lines else "embedding_similarity 无可比较文本。"
        truncate = runtime.truncate_memory_text or (lambda text: text)
        return ToolExecutionResult(
            tool_name=self.name,
            observation_text=truncate(observation),
            payload={
                "query": parsed_args.query,
                "items": ranked,
                "candidate_count": len(candidates),
                "top_score": float(ranked[0]["score"]) if ranked else 0.0,
                "best_match_index": ranked[0]["index"] if ranked else None,
                "exact_match_indexes": exact_match_indexes,
                "exact_match_all": bool(candidates) and len(exact_match_indexes) == len(candidates),
            },
            tool_trace={
                "requested_tool": self.name,
                "status": "ok",
                "candidate_count": len(candidates),
            },
            terminal=False,
        )
