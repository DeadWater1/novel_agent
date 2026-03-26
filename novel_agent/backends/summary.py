from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..schemas import BackendHealth, CompactSummaryArtifact
from ..utils import extract_json_object, split_think_and_answer
from .base import BaseBackend
from .decision import _import_llm_dependencies, _load_model_with_runtime, _model_device


SUMMARY_SYSTEM_PROMPT = """
你是一个会话压缩器。你的任务是把小说 Agent 的长对话压缩成结构化 JSON，供后续 memory_search 和开放性回顾问题使用。

必须遵守：
1. 只能基于提供的 messages / turn_records / session_summary，总结已有事实，不能编造。
2. 优先提取：讨论主题、压缩历史、稳定剧情事实、用户偏好、未完成问题、检索提示词。
3. `compression_history` 只保留真正执行过章节压缩的 turn。
4. `compressed_preview` 必须是压缩结果的短摘要，不要重复整段正文。
5. 所有字段都必须输出；没有内容时用空字符串、空数组。
6. 只输出 JSON，不要输出 JSON 以外的文字。

JSON schema:
{
  "session_id": "string",
  "updated_at": "string",
  "transcript_path": "string",
  "source": "llm",
  "session_goal": "string",
  "discussion_topics": ["string"],
  "compression_history": [
    {
      "turn_index": 0,
      "user_request": "string",
      "compressed_preview": "string",
      "entities": ["string"],
      "timestamp": "string",
      "full_content_target": "string",
      "ordinal_aliases": ["string"]
    }
  ],
  "story_facts": ["string"],
  "user_preferences": ["string"],
  "open_loops": ["string"],
  "timeline_summary": ["string"],
  "search_hints": ["string"]
}
""".strip()


class LocalSummaryBackend(BaseBackend):
    name = "summary_backend"

    def __init__(self, config: AgentConfig, shared_backend: Any | None = None) -> None:
        self.config = config
        self.model_path = Path(config.summary_model_path)
        self.shared_backend = shared_backend
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._load_error: str | None = None

    def healthcheck(self) -> BackendHealth:
        if not self.model_path.exists():
            return BackendHealth(ok=False, name=self.name, detail=f"model_path_not_found:{self.model_path}")
        try:
            self._ensure_loaded(load_model=False)
        except Exception as exc:
            return BackendHealth(ok=False, name=self.name, detail=str(exc))
        return BackendHealth(ok=True, name=self.name, detail=str(self.model_path))

    def _ensure_loaded(self, load_model: bool = True) -> None:
        if self.shared_backend is not None and Path(getattr(self.shared_backend, "model_path", "")) == self.model_path:
            self.shared_backend._ensure_loaded(load_model=load_model)
            self._torch = self.shared_backend._torch
            self._tokenizer = self.shared_backend._tokenizer
            self._model = self.shared_backend._model
            return

        if self._tokenizer is not None and (self._model is not None or not load_model):
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        if not self.model_path.exists():
            self._load_error = f"model_path_not_found:{self.model_path}"
            raise RuntimeError(self._load_error)
        torch, AutoModelForCausalLM, AutoTokenizer = _import_llm_dependencies()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
        if load_model:
            self._model = _load_model_with_runtime(AutoModelForCausalLM, self.model_path, torch)

    def summarize(
        self,
        *,
        session_id: str,
        messages: list[dict[str, str]],
        turn_records: list[dict[str, Any]],
        session_summary: str,
    ) -> CompactSummaryArtifact:
        self._ensure_loaded(load_model=True)
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        request_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"session_id={session_id}\n\n"
                    f"session_summary:\n{session_summary or '(empty)'}\n\n"
                    f"messages:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                    f"turn_records:\n{json.dumps(turn_records, ensure_ascii=False)}"
                ),
            },
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                request_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.summary_enable_thinking,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                request_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        model_inputs = self._tokenizer([prompt], return_tensors="pt").to(_model_device(self._model, self._torch))
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.config.summary_max_new_tokens,
            temperature=self.config.summary_temperature,
            top_p=self.config.summary_top_p,
            top_k=self.config.summary_top_k,
            do_sample=bool(self.config.summary_temperature > 0),
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        _, answer = split_think_and_answer(self._tokenizer, output_ids)
        payload = extract_json_object(answer)
        payload.setdefault("session_id", session_id)
        payload.setdefault("source", "llm")
        return CompactSummaryArtifact.model_validate(payload)
