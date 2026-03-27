from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..schemas import BackendHealth
from .base import BaseBackend
from .decision import _import_llm_dependencies, _model_device, _preferred_cuda_dtype


def _import_embedding_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required") from exc
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception:
        try:
            from modelscope import AutoModel, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers or modelscope is required") from exc
    return torch, AutoModel, AutoTokenizer


def _embedding_runtime(torch_module: Any) -> dict[str, Any]:
    try:
        if torch_module.cuda.is_available():
            return {"dtype": _preferred_cuda_dtype(torch_module), "device": "cuda:0"}
    except Exception:
        pass
    return {"dtype": torch_module.float32, "device": "cpu"}


def _load_embedding_model(AutoModel: Any, model_path: Path, torch_module: Any):
    runtime = _embedding_runtime(torch_module)
    common_kwargs = {"trust_remote_code": True}
    try:
        model = AutoModel.from_pretrained(model_path, dtype=runtime["dtype"], **common_kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(model_path, torch_dtype=runtime["dtype"], **common_kwargs)
    try:
        model = model.to(runtime["device"])
    except Exception:
        pass
    return model


def _load_prompt_templates(model_path: Path) -> tuple[str, str]:
    config_path = model_path / "config_sentence_transformers.json"
    if not config_path.exists():
        return "", ""
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    prompts = payload.get("prompts") or {}
    query_prompt = str(prompts.get("query") or "")
    document_prompt = str(prompts.get("document") or "")
    return query_prompt, document_prompt


class LocalEmbeddingBackend(BaseBackend):
    name = "embedding_backend"

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.model_path = Path(config.embedding_model_path)
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._load_error: str | None = None
        self._query_prompt, self._document_prompt = _load_prompt_templates(self.model_path)

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

        torch, AutoModel, AutoTokenizer = _import_embedding_dependencies()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
        if load_model:
            self._model = _load_embedding_model(AutoModel, self.model_path, torch)

    def similarity(self, query: str, text: str) -> float:
        scores = self.similarity_batch(query, [text])
        return scores[0] if scores else 0.0

    def similarity_batch(self, query: str, texts: list[str]) -> list[float]:
        clean_query = query.strip()
        clean_texts = [text.strip() for text in texts]
        if not clean_query or not clean_texts:
            return [0.0 for _ in texts]

        query_embedding = self.embed_query(clean_query)
        text_embeddings = self.embed_texts(clean_texts, prompt_type="document")
        scores = text_embeddings @ query_embedding.T
        return [float(item) for item in scores.squeeze(1).tolist()]

    def embed_query(self, query: str) -> Any:
        clean_query = query.strip()
        if not clean_query:
            self._ensure_loaded(load_model=True)
            assert self._torch is not None
            return self._torch.empty((0, 0), dtype=self._torch.float32)
        return self.embed_texts([clean_query], prompt_type="query")

    def embed_texts(self, texts: list[str], *, prompt_type: str = "document") -> Any:
        clean_texts = [text.strip() for text in texts]
        if not any(clean_texts):
            self._ensure_loaded(load_model=True)
            assert self._torch is not None
            return self._torch.empty((0, 0), dtype=self._torch.float32)
        return self._encode_texts(clean_texts, prompt_type=prompt_type)

    def _encode_texts(self, texts: list[str], *, prompt_type: str) -> Any:
        self._ensure_loaded(load_model=True)
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        if not texts:
            return self._torch.empty((0, 0), dtype=self._torch.float32)

        prepared = [self._prepare_text(text, prompt_type=prompt_type) for text in texts]
        outputs = []
        device = _model_device(self._model, self._torch)
        max_length = self._effective_max_length()
        batch_size = max(int(self.config.embedding_batch_size), 1)

        with self._torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {name: value.to(device) for name, value in encoded.items()}
                model_output = self._model(**encoded)
                pooled = _last_token_pool(
                    self._torch,
                    model_output.last_hidden_state,
                    encoded["attention_mask"],
                )
                normalized = self._torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                outputs.append(normalized.cpu())

        return self._torch.cat(outputs, dim=0)

    def _effective_max_length(self) -> int:
        assert self._tokenizer is not None
        tokenizer_limit = int(getattr(self._tokenizer, "model_max_length", self.config.embedding_max_length) or 0)
        if tokenizer_limit <= 0 or tokenizer_limit > 1_000_000:
            tokenizer_limit = self.config.embedding_max_length
        return max(1, min(self.config.embedding_max_length, tokenizer_limit))

    def _prepare_text(self, text: str, *, prompt_type: str) -> str:
        prompt = self._query_prompt if prompt_type == "query" else self._document_prompt
        if not prompt:
            return text
        return f"{prompt}{text}"


def _last_token_pool(torch_module: Any, hidden_states: Any, attention_mask: Any) -> Any:
    lengths = attention_mask.long().sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    batch_indexes = torch_module.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indexes, lengths]
