from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..schemas import BackendHealth, CompressionRequest, CompressionResult
from ..utils import get_think_end_token_id, resolve_seed, split_think_and_answer
from .base import BaseBackend
from .decision import _import_llm_dependencies, _load_model_with_runtime, _model_device


COMPRESSION_SYSTEM_PROMPT = """
你是一个小说章节压缩助手。目标是明显缩短篇幅，但必须严格保持原文事实、人物关系、发言归属和叙事顺序。

【硬性要求】
1. 不得编造、补全或篡改原文没有的信息；不得把否定说成肯定，或把肯定说成否定。
2. 人物关系必须严格保持：谁介绍谁、谁对谁说话、谁是谁的配偶/亲属/师徒/上下级、谁属于哪个阵营或身份，均不能反转、错配或改成第三人。
3. 时间顺序、因果关系、事件先后、身份揭示、设定规则必须与原文一致，不得偷换主语或改变动作归属。
4. 遇到多人对话、代词、省略主语、关系复杂的句子时，必须先辨清谁对谁说话、谁与谁是什么关系；若无法100%确认，就按原文的人物指向和关系归属保守改写成叙述句，禁止猜测、补全或改写成其他人物发言。
5. 可以压缩环境描写、重复动作、修辞和无关枝节，但不能删除推动剧情、确认关系、解释设定的关键信息。
6. 输出只做压缩总结：中文通顺、精炼、连贯，不扩写，不灌水，不加入评价。
""".strip()


class LocalCompressionBackend(BaseBackend):
    name = "compression_backend"

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.model_path = Path(config.compression_model_path)
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

    def compress(self, request: CompressionRequest) -> CompressionResult:
        self._ensure_loaded(load_model=True)
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        messages = [
            {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": request.raw_text},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=request.enable_thinking,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        model_inputs = self._tokenizer([prompt], return_tensors="pt").to(_model_device(self._model, self._torch))
        seed = resolve_seed(request.seed)
        generation_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": bool(request.temperature > 0),
        }
        if request.enable_thinking:
            generated_ids = _generate_with_answer_reserve(
                self._torch,
                self._model,
                self._tokenizer,
                model_inputs,
                generation_kwargs,
                seed,
                reserved_answer_tokens=self.config.compression_answer_reserved_tokens,
            )
        else:
            generated_ids = _generate_with_sampling_seed(
                self._torch,
                self._model,
                model_inputs,
                generation_kwargs,
                seed,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        thinking, answer = split_think_and_answer(self._tokenizer, output_ids)
        return CompressionResult(compressed_text=answer, thinking=thinking)


def _build_generator(torch_module: Any, model: Any, seed: int):
    device = _model_device(model, torch_module)
    for candidate in (device, str(device)):
        try:
            generator = torch_module.Generator(device=candidate)
            generator.manual_seed(seed)
            return generator
        except Exception:
            continue

    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return generator


def _generate_with_sampling_seed(
    torch_module: Any,
    model: Any,
    model_inputs: Any,
    generation_kwargs: dict[str, Any],
    seed: int,
):
    if not generation_kwargs.get("do_sample"):
        return model.generate(**model_inputs, **generation_kwargs)

    generator_kwargs = dict(generation_kwargs)
    generator_kwargs["generator"] = _build_generator(torch_module, model, seed)
    try:
        return model.generate(**model_inputs, **generator_kwargs)
    except (TypeError, ValueError) as exc:
        if not _is_unsupported_generator_error(exc):
            raise

    fallback_kwargs = dict(generation_kwargs)
    with _seeded_generation_context(torch_module, model, seed):
        return model.generate(**model_inputs, **fallback_kwargs)


def _generate_with_answer_reserve(
    torch_module: Any,
    model: Any,
    tokenizer: Any,
    model_inputs: Any,
    generation_kwargs: dict[str, Any],
    seed: int,
    *,
    reserved_answer_tokens: int,
):
    total_budget = max(int(generation_kwargs.get("max_new_tokens") or 0), 1)
    answer_budget = min(max(int(reserved_answer_tokens), 1), total_budget)
    thinking_budget = total_budget - answer_budget
    if thinking_budget <= 0:
        return _generate_with_sampling_seed(
            torch_module,
            model,
            model_inputs,
            generation_kwargs,
            seed,
        )

    think_end_id = get_think_end_token_id(tokenizer)
    thinking_kwargs = dict(generation_kwargs)
    thinking_kwargs["max_new_tokens"] = thinking_budget
    thinking_kwargs["eos_token_id"] = think_end_id
    thinking_stage = _generate_with_sampling_seed(
        torch_module,
        model,
        model_inputs,
        thinking_kwargs,
        seed,
    )

    prompt_length = int(model_inputs.input_ids.shape[1])
    thinking_output_ids = thinking_stage[0][prompt_length:].tolist()
    if think_end_id not in thinking_output_ids:
        thinking_stage = _append_forced_think_close(
            torch_module,
            tokenizer,
            thinking_stage,
            think_end_id,
        )

    answer_inputs = {
        "input_ids": thinking_stage,
        "attention_mask": torch_module.ones_like(thinking_stage),
    }
    answer_kwargs = dict(generation_kwargs)
    answer_kwargs["max_new_tokens"] = answer_budget
    answer_kwargs.pop("eos_token_id", None)
    return _generate_with_sampling_seed(
        torch_module,
        model,
        answer_inputs,
        answer_kwargs,
        seed,
    )


def _append_forced_think_close(torch_module: Any, tokenizer: Any, token_ids: Any, think_end_id: int):
    closing_ids = _forced_think_close_ids(tokenizer, think_end_id)
    closing_tensor = torch_module.tensor([closing_ids], dtype=token_ids.dtype, device=token_ids.device)
    return torch_module.cat([token_ids, closing_tensor], dim=1)


def _forced_think_close_ids(tokenizer: Any, think_end_id: int) -> list[int]:
    try:
        encoded = tokenizer.encode("</think>\n\n", add_special_tokens=False)
    except Exception:
        encoded = []
    cleaned = [int(token_id) for token_id in encoded if token_id is not None]
    if not cleaned or think_end_id not in cleaned:
        return [int(think_end_id)]
    return cleaned


def _is_unsupported_generator_error(exc: Exception) -> bool:
    message = str(exc)
    return "generator" in message and "model_kwargs" in message and "not used by the model" in message


@contextmanager
def _seeded_generation_context(torch_module: Any, model: Any, seed: int):
    random_module = getattr(torch_module, "random", None)
    fork_rng = getattr(random_module, "fork_rng", None)
    if callable(fork_rng):
        with fork_rng(devices=_model_cuda_devices(torch_module, model)):
            torch_module.manual_seed(seed)
            yield
        return

    torch_module.manual_seed(seed)
    yield


def _model_cuda_devices(torch_module: Any, model: Any) -> list[int]:
    hf_device_map = getattr(model, "hf_device_map", None)
    devices: list[int] = []
    if isinstance(hf_device_map, dict):
        for value in hf_device_map.values():
            index = _parse_cuda_device_index(value)
            if index is not None:
                devices.append(index)
    if devices:
        return sorted(set(devices))

    index = _parse_cuda_device_index(_model_device(model, torch_module))
    if index is not None:
        return [index]
    return []


def _parse_cuda_device_index(device: Any) -> int | None:
    if isinstance(device, int):
        return device if device >= 0 else None

    device_type = getattr(device, "type", None)
    if device_type == "cuda":
        index = getattr(device, "index", None)
        return 0 if index is None else index

    text = str(device)
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return None
    return None
