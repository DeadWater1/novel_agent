from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..schemas import BackendHealth, CompressionRequest, CompressionResult
from ..utils import resolve_seed, split_think_and_answer
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
        with self._torch.random.fork_rng():
            self._torch.manual_seed(seed)
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=bool(request.temperature > 0),
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        thinking, answer = split_think_and_answer(self._tokenizer, output_ids)
        return CompressionResult(compressed_text=answer, thinking=thinking)
