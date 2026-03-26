from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..prompts import build_decision_review_system_prompt, build_decision_system_prompt
from ..schemas import BackendHealth, DecisionOutput, DecisionReviewOutput
from ..utils import extract_json_object, split_think_and_answer
from .base import BaseBackend


def _import_llm_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required") from exc
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        try:
            from modelscope import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers or modelscope is required") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
    except Exception:
        return False
    return True


def _preferred_cuda_dtype(torch_module: Any):
    try:
        if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
    except Exception:
        pass
    return torch_module.float16


def _select_model_runtime(torch_module: Any) -> dict[str, Any]:
    try:
        if torch_module.cuda.is_available():
            return {
                "dtype": _preferred_cuda_dtype(torch_module),
                "device_map": "auto" if _has_accelerate() else None,
                "device": "cuda:0",
            }
    except Exception:
        pass
    return {"dtype": torch_module.float32, "device_map": None, "device": "cpu"}


def _load_model_with_runtime(AutoModelForCausalLM: Any, model_path: Path, torch_module: Any):
    runtime = _select_model_runtime(torch_module)
    common_kwargs = {"trust_remote_code": True}

    if runtime["device_map"] is not None:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=runtime["dtype"],
                device_map=runtime["device_map"],
                **common_kwargs,
            )
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=runtime["dtype"],
                device_map=runtime["device_map"],
                **common_kwargs,
            )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=runtime["dtype"],
            **common_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=runtime["dtype"],
            **common_kwargs,
        )

    try:
        return model.to(runtime["device"])
    except Exception:
        return model


def _model_device(model: Any, torch_module: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return torch_module.device("cpu")


class LocalDecisionBackend(BaseBackend):
    name = "decision_backend"

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.model_path = Path(config.decision_model_path)
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

    def decide(
        self,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        compacted_session_context: str = "",
        recent_content_references: str = "",
        loop_events: list[dict[str, Any]] | None = None,
    ) -> DecisionOutput:
        payload = self._generate_json_payload(
            system_prompt=build_decision_system_prompt(workspace_docs),
            user_content=(
                "请根据以下对话和会话摘要输出决策 JSON。\n\n"
                f"Session Summary:\n{session_summary or '(empty)'}\n\n"
                f"Compacted Session Context:\n{compacted_session_context or '(empty)'}\n\n"
                f"Recent Content References:\n{recent_content_references or '(empty)'}\n\n"
                f"Messages:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                f"Loop Events:\n{json.dumps(loop_events or [], ensure_ascii=False)}"
            ),
        )
        return DecisionOutput.model_validate(payload)

    def review_decision(
        self,
        *,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        compacted_session_context: str = "",
        recent_content_references: str = "",
        loop_events: list[dict[str, Any]] | None = None,
        decision: dict[str, Any],
        user_text: str,
    ) -> DecisionReviewOutput:
        payload = self._generate_json_payload(
            system_prompt=build_decision_review_system_prompt(workspace_docs),
            user_content=(
                "请判断下面这条 direct_reply 是否已经有足够证据支撑。\n\n"
                f"User Message:\n{user_text}\n\n"
                f"Session Summary:\n{session_summary or '(empty)'}\n\n"
                f"Compacted Session Context:\n{compacted_session_context or '(empty)'}\n\n"
                f"Recent Content References:\n{recent_content_references or '(empty)'}\n\n"
                f"Messages:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                f"Loop Events:\n{json.dumps(loop_events or [], ensure_ascii=False)}\n\n"
                f"Candidate Decision:\n{json.dumps(decision, ensure_ascii=False)}"
            ),
        )
        return DecisionReviewOutput.model_validate(payload)

    def estimate_prompt_tokens(
        self,
        *,
        messages: list[dict[str, str]],
        workspace_docs: str,
        session_summary: str,
        compacted_session_context: str = "",
        recent_content_references: str = "",
        loop_events: list[dict[str, Any]] | None = None,
    ) -> int:
        self._ensure_loaded(load_model=False)
        assert self._tokenizer is not None

        system_prompt = build_decision_system_prompt(workspace_docs)
        request_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "请根据以下对话和会话摘要输出决策 JSON。\n\n"
                    f"Session Summary:\n{session_summary or '(empty)'}\n\n"
                    f"Compacted Session Context:\n{compacted_session_context or '(empty)'}\n\n"
                    f"Recent Content References:\n{recent_content_references or '(empty)'}\n\n"
                    f"Messages:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                    f"Loop Events:\n{json.dumps(loop_events or [], ensure_ascii=False)}"
                ),
            },
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                request_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.decision_enable_thinking,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                request_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        tokenized = self._tokenizer([prompt], return_tensors="pt")
        return int(tokenized.input_ids.shape[-1])

    def _generate_json_payload(self, *, system_prompt: str, user_content: str) -> dict[str, Any]:
        self._ensure_loaded(load_model=True)
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                request_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.decision_enable_thinking,
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
            max_new_tokens=self.config.decision_output_max_new_tokens,
            temperature=self.config.decision_temperature,
            top_p=self.config.decision_top_p,
            top_k=self.config.decision_top_k,
            do_sample=bool(self.config.decision_temperature > 0),
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        _, answer = split_think_and_answer(self._tokenizer, output_ids)
        return extract_json_object(answer)
