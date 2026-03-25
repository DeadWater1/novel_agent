from novel_agent.backends.decision import _select_model_runtime


class StubCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self):
        return self._available


class StubTorch:
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self, cuda_available: bool):
        self.cuda = StubCuda(cuda_available)


def test_runtime_falls_back_to_cpu_without_accelerate(monkeypatch):
    monkeypatch.setattr("novel_agent.backends.decision._has_accelerate", lambda: False)
    runtime = _select_model_runtime(StubTorch(cuda_available=True))
    assert runtime["device"] == "cpu"
    assert runtime["device_map"] is None
    assert runtime["dtype"] == "float32"


def test_runtime_uses_device_map_when_accelerate_and_cuda_available(monkeypatch):
    monkeypatch.setattr("novel_agent.backends.decision._has_accelerate", lambda: True)
    runtime = _select_model_runtime(StubTorch(cuda_available=True))
    assert runtime["device"] == "cuda"
    assert runtime["device_map"] == "auto"
    assert runtime["dtype"] == "bfloat16"
