from typing import Any

from .base import BaseBackend
from .compression import LocalCompressionBackend
from .decision import LocalDecisionBackend
from .embedding import LocalEmbeddingBackend
from .summary import LocalSummaryBackend
from .vllm_backend import VLLMCompressionBackend, VLLMDecisionBackend, VLLMSummaryBackend, _VLLMSharedModel


def build_generation_backends(config: Any):
    if getattr(config, "generation_backend", "local") == "vllm":
        shared_models: dict[str, _VLLMSharedModel] = {}

        def _shared(path_like: Any) -> _VLLMSharedModel:
            path_text = str(path_like)
            shared = shared_models.get(path_text)
            if shared is None:
                shared = _VLLMSharedModel(config, path_like)
                shared_models[path_text] = shared
            return shared

        return (
            VLLMDecisionBackend(config, shared_model=_shared(config.decision_model_path)),
            VLLMCompressionBackend(config, shared_model=_shared(config.compression_model_path)),
            VLLMSummaryBackend(config, shared_model=_shared(config.summary_model_path)),
        )
    decision = LocalDecisionBackend(config)
    compression = LocalCompressionBackend(config)
    summary = LocalSummaryBackend(config, shared_backend=decision)
    return decision, compression, summary


__all__ = [
    "BaseBackend",
    "LocalCompressionBackend",
    "LocalDecisionBackend",
    "LocalEmbeddingBackend",
    "LocalSummaryBackend",
    "VLLMCompressionBackend",
    "VLLMDecisionBackend",
    "VLLMSummaryBackend",
    "build_generation_backends",
]
