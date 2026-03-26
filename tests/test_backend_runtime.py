from novel_agent.backends.compression import _build_generator, _generate_with_sampling_seed
from novel_agent.backends.decision import _select_model_runtime


class StubCuda:
    def __init__(self, available: bool, bf16_supported: bool = True):
        self._available = available
        self._bf16_supported = bf16_supported

    def is_available(self):
        return self._available

    def is_bf16_supported(self):
        return self._bf16_supported


class StubTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, cuda_available: bool, bf16_supported: bool = True):
        self.cuda = StubCuda(cuda_available, bf16_supported=bf16_supported)


class StubGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class StubTorchWithGenerator:
    def __init__(self):
        self.calls = []
        self.manual_seed_calls = []
        self.fork_rng_devices = []
        self.random = StubRandom(self)

    def Generator(self, device=None):
        self.calls.append(device)
        return StubGenerator(device=device)

    def manual_seed(self, seed):
        self.manual_seed_calls.append(seed)


class StubForkRngContext:
    def __init__(self, owner, devices):
        self.owner = owner
        self.devices = devices

    def __enter__(self):
        self.owner.fork_rng_devices.append(self.devices)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class StubRandom:
    def __init__(self, owner):
        self.owner = owner

    def fork_rng(self, devices=None):
        return StubForkRngContext(self.owner, devices)


class RejectsGeneratorModel:
    device = "cuda:0"

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if "generator" in kwargs:
            raise ValueError("The following `model_kwargs` are not used by the model: ['generator']")
        return "ok"


def test_runtime_uses_single_gpu_without_accelerate(monkeypatch):
    monkeypatch.setattr("novel_agent.backends.decision._has_accelerate", lambda: False)
    runtime = _select_model_runtime(StubTorch(cuda_available=True))
    assert runtime["device"] == "cuda:0"
    assert runtime["device_map"] is None
    assert runtime["dtype"] == "bfloat16"


def test_runtime_uses_device_map_when_accelerate_and_cuda_available(monkeypatch):
    monkeypatch.setattr("novel_agent.backends.decision._has_accelerate", lambda: True)
    runtime = _select_model_runtime(StubTorch(cuda_available=True))
    assert runtime["device"] == "cuda:0"
    assert runtime["device_map"] == "auto"
    assert runtime["dtype"] == "bfloat16"


def test_runtime_uses_fp16_when_bf16_is_not_supported(monkeypatch):
    monkeypatch.setattr("novel_agent.backends.decision._has_accelerate", lambda: False)
    runtime = _select_model_runtime(StubTorch(cuda_available=True, bf16_supported=False))
    assert runtime["device"] == "cuda:0"
    assert runtime["device_map"] is None
    assert runtime["dtype"] == "float16"


def test_build_generator_uses_model_device_and_seed():
    torch = StubTorchWithGenerator()
    model = type("StubModel", (), {"device": "cuda:0"})()
    generator = _build_generator(torch, model, 123)
    assert torch.calls == ["cuda:0"]
    assert generator.device == "cuda:0"
    assert generator.seed == 123


def test_generate_with_sampling_seed_retries_without_generator_when_model_rejects_it():
    torch = StubTorchWithGenerator()
    model = RejectsGeneratorModel()
    result = _generate_with_sampling_seed(
        torch,
        model,
        {"input_ids": [[1, 2, 3]]},
        {"do_sample": True, "temperature": 0.7},
        123,
    )
    assert result == "ok"
    assert len(model.calls) == 2
    assert "generator" in model.calls[0]
    assert "generator" not in model.calls[1]
    assert torch.fork_rng_devices == [[0]]
    assert torch.manual_seed_calls == [123]
