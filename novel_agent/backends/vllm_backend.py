from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..prompts import (
    build_decision_review_system_prompt,
    build_decision_system_prompt,
    build_plan_system_prompt,
)
from ..schemas import (
    BackendHealth,
    CompactSummaryArtifact,
    CompressionRequest,
    CompressionResult,
    DecisionOutput,
    DecisionReviewOutput,
    ExecutionPlanOutput,
)
from ..utils import extract_answer_text, extract_json_object, extract_think_text, resolve_seed
from .base import BaseBackend
from .compression import COMPRESSION_SYSTEM_PROMPT
from .decision import _import_llm_dependencies
from .summary import SUMMARY_SYSTEM_PROMPT


def _import_vllm_dependencies() -> tuple[Any, Any]:
    try:
        module = importlib.import_module("vllm")
    except Exception as exc:
        raise RuntimeError("vllm is required") from exc
    return module.LLM, module.SamplingParams


def _estimate_text_tokens(*parts: Any) -> int:
    merged = "".join(str(part or "") for part in parts)
    return max(len(merged) // 4, 1)


class _VLLMSharedModel:
    def __init__(self, config: AgentConfig, model_path: Path) -> None:
        self.config = config
        self.model_path = Path(model_path)
        self._tokenizer = None
        self._llm = None
        self._load_error: str | None = None

    def ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        if not self.model_path.exists():
            self._load_error = f"model_path_not_found:{self.model_path}"
            raise RuntimeError(self._load_error)
        _, _, AutoTokenizer = _import_llm_dependencies()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=self.config.vllm_trust_remote_code,
        )
        return self._tokenizer

    def ensure_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        if not self.model_path.exists():
            self._load_error = f"model_path_not_found:{self.model_path}"
            raise RuntimeError(self._load_error)
        LLM, _ = _import_vllm_dependencies()
        self._llm = LLM(
            model=str(self.model_path),
            tokenizer=str(self.model_path),
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            max_model_len=self.config.vllm_max_model_len,
            dtype=self.config.vllm_dtype,
            trust_remote_code=self.config.vllm_trust_remote_code,
            enforce_eager=self.config.vllm_enforce_eager,
        )
        return self._llm


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


class _BaseVLLMBackend(BaseBackend):
    model_path: Path

    def __init__(
        self,
        config: AgentConfig,
        *,
        name: str,
        model_path: Path,
        shared_model: _VLLMSharedModel | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self.model_path = Path(model_path)
        self._shared_model = shared_model or _VLLMSharedModel(config, self.model_path)

    def healthcheck(self) -> BackendHealth:
        if not self.model_path.exists():
            return BackendHealth(ok=False, name=self.name, detail=f"model_path_not_found:{self.model_path}")
        try:
            self._shared_model.ensure_tokenizer()
            _import_vllm_dependencies()
        except Exception as exc:
            return BackendHealth(ok=False, name=self.name, detail=str(exc))
        return BackendHealth(ok=True, name=self.name, detail=str(self.model_path))

    def _generate_text(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
        seed: int | None = None,
    ) -> str:
        tokenizer = self._shared_model.ensure_tokenizer()
        llm = self._shared_model.ensure_llm()
        _, SamplingParams = _import_vllm_dependencies()
        prompt = _apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
        sampling_params = SamplingParams(
            max_tokens=max(max_tokens, 1),
            temperature=max(float(temperature), 0.0),
            top_p=float(top_p),
            top_k=int(top_k),
            seed=seed,
        )
        outputs = llm.generate([prompt], sampling_params)
        if not outputs:
            raise RuntimeError("vllm_empty_response")
        candidates = getattr(outputs[0], "outputs", [])
        if not candidates:
            raise RuntimeError("vllm_empty_candidates")
        text = str(getattr(candidates[0], "text", "") or "")
        if not text.strip():
            raise RuntimeError("vllm_empty_text")
        return text


class VLLMDecisionBackend(_BaseVLLMBackend):
    name = "decision_backend"

    def __init__(self, config: AgentConfig, shared_model: _VLLMSharedModel | None = None) -> None:
        super().__init__(
            config,
            name=self.name,
            model_path=Path(config.decision_model_path),
            shared_model=shared_model,
        )
        self.last_decision_thinking: str = ""
        self.last_review_thinking: str = ""
        self.last_plan_output_text: str = ""
        self.last_decision_output_text: str = ""
        self.last_review_output_text: str = ""

    def plan_turn(
        self,
        *,
        user_text: str,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        tool_prompt_docs: str = "",
        tool_names: tuple[str, ...] = (),
        compacted_session_context: str = "",
        recent_content_references: str = "",
        loop_events: list[dict[str, Any]] | None = None,
    ) -> ExecutionPlanOutput:
        output_text = self._generate_text(
            messages=[
                {"role": "system", "content": build_plan_system_prompt(workspace_docs, tool_prompt_docs, tool_names)},
                {
                    "role": "user",
                    "content": (
                        "请先为当前用户问题生成一个显式执行计划 JSON。\n\n"
                        f"User Message:\n{user_text}\n\n"
                        f"Session Summary:\n{session_summary or '(empty)'}\n\n"
                        f"Compacted Session Context:\n{compacted_session_context or '(empty)'}\n\n"
                        f"Recent Content References:\n{recent_content_references or '(empty)'}\n\n"
                        f"Messages:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                        f"Loop Events:\n{json.dumps(loop_events or [], ensure_ascii=False)}"
                    ),
                },
            ],
            max_tokens=self.config.decision_output_max_new_tokens,
            temperature=self.config.decision_temperature,
            top_p=self.config.decision_top_p,
            top_k=self.config.decision_top_k,
            enable_thinking=self.config.decision_enable_thinking,
        )
        self.last_plan_output_text = output_text
        return ExecutionPlanOutput.model_validate(extract_json_object(extract_answer_text(output_text)))

    def decide(
        self,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        tool_prompt_docs: str = "",
        tool_names: tuple[str, ...] = (),
        compacted_session_context: str = "",
        recent_content_references: str = "",
        loop_events: list[dict[str, Any]] | None = None,
        execution_plan: dict[str, Any] | None = None,
        current_step: dict[str, Any] | None = None,
        completed_steps: list[dict[str, Any]] | None = None,
    ) -> DecisionOutput:
        output_text = self._generate_text(
            messages=[
                {"role": "system", "content": build_decision_system_prompt(workspace_docs, tool_prompt_docs, tool_names)},
                {
                    "role": "user",
                    "content": (
                        "请根据以下对话和会话摘要输出决策 JSON。\n\n"
                        f"Execution Plan:\n{json.dumps(execution_plan or {}, ensure_ascii=False)}\n\n"
                        f"Current Step:\n{json.dumps(current_step or {}, ensure_ascii=False)}\n\n"
                        f"Completed Steps:\n{json.dumps(completed_steps or [], ensure_ascii=False)}\n\n"
                        f"Session Summary:\n{session_summary or '(empty)'}\n\n"
                        f"Compacted Session Context:\n{compacted_session_context or '(empty)'}\n\n"
                        f"Recent Content References:\n{recent_content_references or '(empty)'}\n\n"
                        f"Messages:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                        f"Loop Events:\n{json.dumps(loop_events or [], ensure_ascii=False)}"
                    ),
                },
            ],
            max_tokens=self.config.decision_output_max_new_tokens,
            temperature=self.config.decision_temperature,
            top_p=self.config.decision_top_p,
            top_k=self.config.decision_top_k,
            enable_thinking=self.config.decision_enable_thinking,
        )
        self.last_decision_thinking = extract_think_text(output_text)
        self.last_decision_output_text = output_text
        return DecisionOutput.model_validate(extract_json_object(extract_answer_text(output_text)))

    def review_decision(
        self,
        *,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        compacted_session_context: str = "",
        recent_content_references: str = "",
        loop_events: list[dict[str, Any]] | None = None,
        execution_plan: dict[str, Any] | None = None,
        current_step: dict[str, Any] | None = None,
        completed_steps: list[dict[str, Any]] | None = None,
        decision: dict[str, Any],
        user_text: str,
    ) -> DecisionReviewOutput:
        output_text = self._generate_text(
            messages=[
                {"role": "system", "content": build_decision_review_system_prompt(workspace_docs)},
                {
                    "role": "user",
                    "content": (
                        "请判断下面这条 direct_reply 是否已经有足够证据支撑。\n\n"
                        f"User Message:\n{user_text}\n\n"
                        f"Execution Plan:\n{json.dumps(execution_plan or {}, ensure_ascii=False)}\n\n"
                        f"Current Step:\n{json.dumps(current_step or {}, ensure_ascii=False)}\n\n"
                        f"Completed Steps:\n{json.dumps(completed_steps or [], ensure_ascii=False)}\n\n"
                        f"Session Summary:\n{session_summary or '(empty)'}\n\n"
                        f"Compacted Session Context:\n{compacted_session_context or '(empty)'}\n\n"
                        f"Recent Content References:\n{recent_content_references or '(empty)'}\n\n"
                        f"Messages:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                        f"Loop Events:\n{json.dumps(loop_events or [], ensure_ascii=False)}\n\n"
                        f"Candidate Decision:\n{json.dumps(decision, ensure_ascii=False)}"
                    ),
                },
            ],
            max_tokens=self.config.decision_output_max_new_tokens,
            temperature=self.config.decision_temperature,
            top_p=self.config.decision_top_p,
            top_k=self.config.decision_top_k,
            enable_thinking=self.config.decision_enable_thinking,
        )
        self.last_review_thinking = extract_think_text(output_text)
        self.last_review_output_text = output_text
        return DecisionReviewOutput.model_validate(extract_json_object(extract_answer_text(output_text)))

    def estimate_prompt_tokens(
        self,
        *,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        tool_prompt_docs: str = "",
        tool_names: tuple[str, ...] = (),
        compacted_session_context: str = "",
        recent_content_references: str = "",
        loop_events: list[dict[str, Any]] | None = None,
        execution_plan: dict[str, Any] | None = None,
        current_step: dict[str, Any] | None = None,
        completed_steps: list[dict[str, Any]] | None = None,
    ) -> int:
        tokenizer = self._shared_model.ensure_tokenizer()
        prompt = _apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": build_decision_system_prompt(workspace_docs, tool_prompt_docs, tool_names)},
                {
                    "role": "user",
                    "content": (
                        "请根据以下对话和会话摘要输出决策 JSON。\n\n"
                        f"Execution Plan:\n{json.dumps(execution_plan or {}, ensure_ascii=False)}\n\n"
                        f"Current Step:\n{json.dumps(current_step or {}, ensure_ascii=False)}\n\n"
                        f"Completed Steps:\n{json.dumps(completed_steps or [], ensure_ascii=False)}\n\n"
                        f"Session Summary:\n{session_summary or '(empty)'}\n\n"
                        f"Compacted Session Context:\n{compacted_session_context or '(empty)'}\n\n"
                        f"Recent Content References:\n{recent_content_references or '(empty)'}\n\n"
                        f"Messages:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                        f"Loop Events:\n{json.dumps(loop_events or [], ensure_ascii=False)}"
                    ),
                },
            ],
            enable_thinking=self.config.decision_enable_thinking,
        )
        try:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return max(len(token_ids), 1)
        except Exception:
            return _estimate_text_tokens(
                messages,
                workspace_docs,
                session_summary,
                tool_prompt_docs,
                compacted_session_context,
                recent_content_references,
                loop_events or [],
                execution_plan or {},
                current_step or {},
                completed_steps or [],
            )


class VLLMCompressionBackend(_BaseVLLMBackend):
    name = "compression_backend"

    def __init__(self, config: AgentConfig, shared_model: _VLLMSharedModel | None = None) -> None:
        super().__init__(
            config,
            name=self.name,
            model_path=Path(config.compression_model_path),
            shared_model=shared_model,
        )

    def compress(self, request: CompressionRequest) -> CompressionResult:
        output_text = self._generate_text(
            messages=[
                {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": request.raw_text},
            ],
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            enable_thinking=request.enable_thinking,
            seed=resolve_seed(request.seed),
        )
        return CompressionResult(
            compressed_text=extract_answer_text(output_text),
            thinking=extract_think_text(output_text),
        )


class VLLMSummaryBackend(_BaseVLLMBackend):
    name = "summary_backend"

    def __init__(self, config: AgentConfig, shared_model: _VLLMSharedModel | None = None) -> None:
        super().__init__(
            config,
            name=self.name,
            model_path=Path(config.summary_model_path),
            shared_model=shared_model,
        )

    def summarize(
        self,
        *,
        session_id: str,
        messages: list[dict[str, str]],
        turn_records: list[dict[str, Any]],
        session_summary: str,
    ) -> CompactSummaryArtifact:
        output_text = self._generate_text(
            messages=[
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
            ],
            max_tokens=self.config.summary_max_new_tokens,
            temperature=self.config.summary_temperature,
            top_p=self.config.summary_top_p,
            top_k=self.config.summary_top_k,
            enable_thinking=self.config.summary_enable_thinking,
        )
        payload = extract_json_object(extract_answer_text(output_text))
        payload.setdefault("session_id", session_id)
        payload.setdefault("source", "llm")
        return CompactSummaryArtifact.model_validate(payload)
