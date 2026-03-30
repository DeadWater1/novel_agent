from pathlib import Path

from novel_agent.backends import (
    LocalCompressionBackend,
    LocalDecisionBackend,
    LocalSummaryBackend,
    VLLMDecisionBackend,
    VLLMSummaryBackend,
    build_generation_backends,
)
from novel_agent.config import AgentConfig


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(item["content"] for item in messages)

    def encode(self, prompt, add_special_tokens=False):
        return list(range(max(len(prompt) // 3, 1)))


def test_build_generation_backends_uses_local_by_default():
    decision, compression, summary = build_generation_backends(AgentConfig())
    assert isinstance(decision, LocalDecisionBackend)
    assert isinstance(compression, LocalCompressionBackend)
    assert isinstance(summary, LocalSummaryBackend)


def test_build_generation_backends_can_select_vllm():
    config = AgentConfig(generation_backend="vllm")
    decision, compression, summary = build_generation_backends(config)
    assert isinstance(decision, VLLMDecisionBackend)
    assert decision.model_path == Path(config.decision_model_path)
    assert compression.model_path == Path(config.compression_model_path)
    assert summary.model_path == Path(config.summary_model_path)


def test_build_generation_backends_reuses_shared_model_for_summary_when_paths_match():
    base = AgentConfig()
    config = AgentConfig(generation_backend="vllm", summary_model_path=Path(base.decision_model_path))
    decision, _, summary = build_generation_backends(config)
    assert isinstance(decision, VLLMDecisionBackend)
    assert isinstance(summary, VLLMSummaryBackend)
    assert decision._shared_model is summary._shared_model


def test_vllm_healthcheck_reports_missing_dependency(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    backend = VLLMDecisionBackend(AgentConfig(generation_backend="vllm", decision_model_path=model_dir))
    monkeypatch.setattr("novel_agent.backends.vllm_backend._import_llm_dependencies", lambda: (None, None, type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: StubTokenizer())})))
    monkeypatch.setattr(
        "novel_agent.backends.vllm_backend._import_vllm_dependencies",
        lambda: (_ for _ in ()).throw(RuntimeError("vllm is required")),
    )
    health = backend.healthcheck()
    assert health.ok is False
    assert "vllm is required" in health.detail


def test_vllm_healthcheck_reports_missing_model_path(tmp_path: Path):
    backend = VLLMDecisionBackend(AgentConfig(generation_backend="vllm", decision_model_path=tmp_path / "missing"))
    health = backend.healthcheck()
    assert health.ok is False
    assert "model_path_not_found" in health.detail


def test_vllm_healthcheck_reports_ok_when_dependency_and_tokenizer_exist(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    backend = VLLMDecisionBackend(AgentConfig(generation_backend="vllm", decision_model_path=model_dir))
    monkeypatch.setattr("novel_agent.backends.vllm_backend._import_llm_dependencies", lambda: (None, None, type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: StubTokenizer())})))
    monkeypatch.setattr("novel_agent.backends.vllm_backend._import_vllm_dependencies", lambda: (object, object))
    health = backend.healthcheck()
    assert health.ok is True
    assert health.name == "decision_backend"


def test_from_env_reads_vllm_configuration(monkeypatch):
    monkeypatch.setenv("NOVEL_AGENT_GENERATION_BACKEND", "vllm")
    monkeypatch.setenv("NOVEL_AGENT_VLLM_TENSOR_PARALLEL_SIZE", "2")
    monkeypatch.setenv("NOVEL_AGENT_VLLM_GPU_MEMORY_UTILIZATION", "0.85")
    monkeypatch.setenv("NOVEL_AGENT_VLLM_MAX_MODEL_LEN", "16384")
    monkeypatch.setenv("NOVEL_AGENT_VLLM_DTYPE", "bfloat16")
    monkeypatch.setenv("NOVEL_AGENT_VLLM_TRUST_REMOTE_CODE", "false")
    monkeypatch.setenv("NOVEL_AGENT_VLLM_ENFORCE_EAGER", "true")
    config = AgentConfig.from_env()
    assert config.generation_backend == "vllm"
    assert config.vllm_tensor_parallel_size == 2
    assert config.vllm_gpu_memory_utilization == 0.85
    assert config.vllm_max_model_len == 16384
    assert config.vllm_dtype == "bfloat16"
    assert config.vllm_trust_remote_code is False
    assert config.vllm_enforce_eager is True
