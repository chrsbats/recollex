import pytest
from typing import List, Tuple
from pathlib import Path
import warnings

# Silence noisy PyTorch ONNX exporter re-registration warning without importing torch
warnings.filterwarnings(
    "ignore",
    message=r"Symbolic function 'aten::scaled_dot_product_attention' already registered.*",
)
warnings.filterwarnings(
    "ignore",
    module=r"torch\.onnx\._internal\.registration",
)

from recollex.engine import Recollex
from recollex.encoder.splade import SpladeEncoder


class FakeEncoder:
    def __init__(self, model: str = "fake", backend: str = "onnx") -> None:
        self._dims = 8

    @property
    def dims(self) -> int:
        return 8

    def _encode_text(self, text: str) -> Tuple[List[int], List[float]]:
        # Two deterministic term ids from salted hash; ensure distinct within dims
        h = abs(hash(text))
        i1 = h % self._dims
        i2 = (h // 7 + 1) % self._dims
        if i2 == i1:
            i2 = (i2 + 1) % self._dims
        return [int(i1), int(i2)], [1.0, 0.5]

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        return self._encode_text(text)

    def encode_many(self, texts):
        return [self._encode_text(t) for t in texts]


@pytest.fixture(autouse=True)
def _patch_encoder(monkeypatch, request):
    # Skip patching when explicitly requested (for real Splade e2e)
    if request.node.get_closest_marker("real_splade"):
        # Do not patch; still yield to satisfy generator fixture contract
        yield
        return
    monkeypatch.setattr("recollex.engine.SpladeEncoder", FakeEncoder)
    yield


@pytest.fixture
def index(tmp_path):
    return Recollex.open(tmp_path / "idx")


@pytest.fixture
def now():
    counter = {"t": 10_000}
    def _next() -> int:
        counter["t"] += 1
        return counter["t"]
    return _next


MODEL_DIR = Path("./models/seerware__Splade_PP_en_v2")

@pytest.fixture(scope="session")
def splade_enc():
    precisions = ("int8", "fp16", "fp32")
    has_any = any((MODEL_DIR / "onnx" / q / "model.onnx").exists() for q in precisions)
    if not has_any:
        pytest.skip("Prefetched model not found; run `recollex-prefetch --model seerware/Splade_PP_en_v2`")
    # Ensure we force CPU to avoid using GPU in tests
    return SpladeEncoder(model="seerware/Splade_PP_en_v2", backend="onnx", device="cpu")
