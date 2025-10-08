import os
import types
import builtins
import numpy as np
import scipy.sparse as sp
import pytest

import recollex.encoder.splade as spl
from importlib import metadata as importlib_metadata


def test_env_precision_valid_and_invalid(monkeypatch):
    monkeypatch.setenv("RECOLLEX_ONNX_PRECISION", "fp16")
    assert spl._get_env_precision() == "fp16"
    monkeypatch.setenv("RECOLLEX_ONNX_PRECISION", "weird")
    assert spl._get_env_precision() is None
    monkeypatch.delenv("RECOLLEX_ONNX_PRECISION", raising=False)
    assert spl._get_env_precision() is None


def test_installed_ort_flavor_detects_and_none(monkeypatch):
    calls = {"n": 0}
    def fake_version(name):
        calls["n"] += 1
        if name == "onnxruntime-rocm":
            return "1.17.0"
        raise importlib_metadata.PackageNotFoundError  # type: ignore[name-defined]
    # Bind name used in module (imported earlier)
    import importlib.metadata as im  # noqa
    monkeypatch.setattr(spl.importlib_metadata, "version", fake_version)
    assert spl._installed_ort_flavor() == "onnxruntime-rocm"
    # None found
    def fake_version_none(name):
        raise importlib_metadata.PackageNotFoundError  # type: ignore[name-defined]
    monkeypatch.setattr(spl.importlib_metadata, "version", fake_version_none)
    assert spl._installed_ort_flavor() is None

def test_installed_ort_flavor_handles_generic_exception(monkeypatch):
    # Generic exception should be swallowed and continue scanning; final result None
    def boom_version(name):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(spl.importlib_metadata, "version", boom_version)
    assert spl._installed_ort_flavor() is None


def test_has_accel_provider_true_and_false(monkeypatch):
    class ORT:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider", "DmlExecutionProvider"]
    monkeypatch.setattr(spl, "ort", ORT)
    assert spl._has_accel_provider() is True
    class ORT2:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]
    monkeypatch.setattr(spl, "ort", ORT2)
    assert spl._has_accel_provider() is False
    # broken ORT
    class ORT3:
        @staticmethod
        def get_available_providers():
            raise RuntimeError("boom")
    monkeypatch.setattr(spl, "ort", ORT3)
    assert spl._has_accel_provider() is False


def test_default_providers_order_and_fallback(monkeypatch):
    class ORT:
        @staticmethod
        def get_available_providers():
            return ["ROCMExecutionProvider", "CPUExecutionProvider", "CUDAExecutionProvider"]
    monkeypatch.setattr(spl, "ort", ORT)
    assert spl._default_providers() == ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
    class ORTNone:
        @staticmethod
        def get_available_providers():
            raise RuntimeError("boom")
    monkeypatch.setattr(spl, "ort", ORTNone)
    assert spl._default_providers() == ["CPUExecutionProvider"]
    monkeypatch.setattr(spl, "ort", None)
    assert spl._default_providers() == ["CPUExecutionProvider"]


def test_choose_precision_env_flavor_providers_and_fallback(monkeypatch):
    monkeypatch.setenv("RECOLLEX_ONNX_PRECISION", "fp32")
    assert spl._choose_onnx_precision() == "fp32"
    monkeypatch.delenv("RECOLLEX_ONNX_PRECISION", raising=False)

    monkeypatch.setattr(spl, "_installed_ort_flavor", lambda: "onnxruntime-gpu")
    assert spl._choose_onnx_precision() == "fp16"

    monkeypatch.setattr(spl, "_installed_ort_flavor", lambda: None)
    monkeypatch.setattr(spl, "_has_accel_provider", lambda: True)
    assert spl._choose_onnx_precision() == "fp16"

    monkeypatch.setattr(spl, "_has_accel_provider", lambda: False)
    assert spl._choose_onnx_precision() == "int8"


def test_SpladeEncoder_init_uses_cache_and_tokenizer_dims(monkeypatch, tmp_path):
    # Work in a temp CWD so ./models is under tmp
    monkeypatch.chdir(tmp_path)
    # Ensure required ONNX path exists so prefetch is not called
    models_dir = tmp_path / "models" / "seerware__Splade_PP_en_v2" / "onnx" / "int8"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "model.onnx").write_bytes(b"fake")
    # Fake backend encoder
    class FakeONNX:
        def __init__(self, *a, **k):
            self.tokenizer = types.SimpleNamespace(vocab_size=1234)
            self.top_k = 350
            self.threshold = None
        def to(self, device): return self
        def encode(self, *a, **k):
            # CSR 1xD
            return sp.csr_matrix(([1.0], ([0], [0])), shape=(1, 2048), dtype=np.float32)
    # Avoid network prefetch regardless
    monkeypatch.setattr(spl, "prefetch", lambda *a, **k: str(tmp_path / "models"))
    # Inject fake SparseEncoderONNX
    import recollex.encoder.stonnx as st
    monkeypatch.setattr(st, "SparseEncoderONNX", FakeONNX)
    # Clear cache to control behavior
    spl._ENC_CACHE.clear()
    enc1 = spl.SpladeEncoder(model="seerware/Splade_PP_en_v2", backend="onnx", device="cpu")
    assert enc1.dims == 1234
    # Second init hits cache (same object returned)
    enc2 = spl.SpladeEncoder(model="seerware/Splade_PP_en_v2", backend="onnx", device="cpu")
    assert enc1._se is enc2._se
    # Properties read different aliases
    assert enc1.sparsify_topk == 350
    assert enc1.sparsify_threshold is None


def test_row_to_indices_values_tuple_and_dense(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Minimal setup to get an instance
    target = tmp_path / "models" / "m__x" / "onnx" / "int8"
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.onnx").write_bytes(b"ok")
    class FakeONNX:
        def __init__(self, *a, **k):
            self.tokenizer = None
        def to(self, device): return self
        def encode(self, texts, **kw):
            return sp.csr_matrix(([1.0], ([0], [1])), shape=(1, 4))
    import recollex.encoder.stonnx as st
    monkeypatch.setattr(st, "SparseEncoderONNX", FakeONNX)
    enc = spl.SpladeEncoder(model="m/x", backend="onnx", device="cpu")
    idx, vals = enc._row_to_indices_values(([3, 7], [0.1, 0.2]))
    assert idx == [3, 7] and vals == [0.1, 0.2]
    dense = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
    di, dv = enc._row_to_indices_values(dense)
    assert di == [1, 3] and dv == [1.0, 2.0]


def test_encode_and_encode_many_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Prepare fake model file
    target = tmp_path / "models" / "m__y" / "onnx" / "int8"
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.onnx").write_bytes(b"x")
    # Fake returns CSR for single, and iterable rows for many
    class FakeONNX:
        def __init__(self, *a, **k): pass
        def to(self, device): return self
        def encode(self, texts, **kw):
            if isinstance(texts, list) and len(texts) == 1:
                return sp.csr_matrix(([1.0, 2.0], ([0, 0], [1, 3])), shape=(1, 5), dtype=np.float32)
            # "torch-like" iterable (no getrow), returns list of (idx, vals)
            class TorchLike(list):
                def coalesce(self): return self
            rows = TorchLike()
            rows.append(([2], [0.5]))
            rows.append(([1, 3], [1.0, 2.0]))
            return rows
    import recollex.encoder.stonnx as st
    monkeypatch.setattr(st, "SparseEncoderONNX", FakeONNX)
    enc = spl.SpladeEncoder(model="m/y", backend="onnx", device="cpu")
    i, v = enc.encode("one")
    assert i == [1, 3] and v == [1.0, 2.0]
    many = enc.encode_many(["a", "b"])
    assert many == [([2], [0.5]), ([1, 3], [1.0, 2.0])]


def test_prefetch_onnx_skips_when_required_exists_and_when_preexisting(monkeypatch, tmp_path):
    # Set up base dirs
    base = tmp_path / "models"
    model = "owner/model"
    target = base / model.replace("/", "__")
    sub = "onnx/fp16"
    req = target / sub / "model.onnx"
    # Case 1: required exists (early return)
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_bytes(b"x")
    called = {"n": 0}
    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("should not be called")
    monkeypatch.setattr(spl, "snapshot_download", boom)
    out = spl.prefetch(model=model, models_dir=str(base), backend="onnx", model_kwargs={"subfolder": sub, "file_name": "model.onnx"})
    assert out == str(target) and called["n"] == 0
    # Case 2: preexisting target but required missing → still early return
    # Reset target without required
    for p in sorted(target.rglob("*"), reverse=True):
        try: p.unlink()
        except Exception: pass
    for p in sorted(target.rglob("*"), reverse=True):
        try: p.rmdir()
        except Exception: pass
    target.mkdir(parents=True, exist_ok=True)
    out2 = spl.prefetch(model=model, models_dir=str(base), backend="onnx", model_kwargs={"subfolder": sub, "file_name": "model.onnx"})
    assert out2 == str(target)


def test_prefetch_snapshot_fallback_on_typeerror(monkeypatch, tmp_path):
    base = tmp_path / "models"
    model = "owner/model"
    sub = "onnx/int8"
    called = {"calls": []}
    def fake_snap(**kw):
        called["calls"].append(("first", kw))
        raise TypeError("older hub")
    def fake_snap2(**kw):
        called["calls"].append(("second", kw))
    # Patch the first call to raise, then replace with second impl
    def side_effect(*a, **kw):
        return fake_snap(**kw)
    monkeypatch.setattr(spl, "snapshot_download", side_effect)
    # After first failure, patch to second implementation
    def switch(*a, **kw):
        monkeypatch.setattr(spl, "snapshot_download", lambda **kw2: fake_snap2(**kw2))
        return fake_snap(**kw)
    monkeypatch.setattr(spl, "snapshot_download", switch)
    out = spl.prefetch(model=model, models_dir=str(base), backend="onnx", model_kwargs={"subfolder": sub, "file_name": "model.onnx"})
    assert out == str((base / model.replace("/", "__")))
    # We expect at least one recorded call; exact args differ by fallback
    assert len(called["calls"]) >= 1


def test_prefetch_main_invokes_prefetch_and_prints(monkeypatch, capsys):
    def fake_prefetch(**kw):
        return "/tmp/models/owner__m"
    monkeypatch.setattr(spl, "prefetch", lambda model, models_dir, backend, model_kwargs=None: fake_prefetch())
    rc = spl.prefetch_main(["--model", "owner/m", "--models-dir", "/tmp/models", "--backend", "onnx", "--quant", "fp16"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "/tmp/models/owner__m" in out


def test_clean_main_dry_and_real(tmp_path, capsys):
    base = tmp_path / "models"
    tgt = base / "owner__m"
    (tgt / "onnx" / "fp16").mkdir(parents=True, exist_ok=True)
    # dry run quant removal
    rc = spl.clean_main(["--model", "owner/m", "--models-dir", str(base), "--backend", "onnx", "--quant", "fp16", "--dry"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would remove" in out
    assert (tgt / "onnx" / "fp16").exists()
    # real remove all (no quant)
    (base / ".hf_cache").mkdir(parents=True, exist_ok=True)
    rc2 = spl.clean_main(["--model", "owner/m", "--models-dir", str(base)])
    assert rc2 == 0
    # target removed
    assert not tgt.exists()

def test_clean_main_refuses_root(capsys):
    # Refuses to operate on filesystem root
    rc = spl.clean_main(["--models-dir", "/"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Refusing to operate on root" in err

def test_splade_onnx_missing_ort_raises(monkeypatch, tmp_path):
    # Ensure required ONNX path exists so prefetch is not called
    base = tmp_path / "models" / "m__z" / "onnx" / "int8"
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.onnx").write_bytes(b"fake")
    # Simulate missing onnxruntime
    monkeypatch.setattr(spl, "ort", None)
    with pytest.raises(RuntimeError):
        spl.SpladeEncoder(model="m/z", backend="onnx", device="cpu")

def test_row_to_indices_values_torch_coo(monkeypatch, tmp_path):
    # Minimal setup to instantiate
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models" / "m__coo" / "onnx" / "int8"
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.onnx").write_bytes(b"ok")
    class FakeONNX:
        def __init__(self, *a, **k):
            self.tokenizer = None
        def to(self, device): return self
        def encode(self, texts, **kw):
            return sp.csr_matrix(([1.0], ([0], [1])), shape=(1, 4))
    import recollex.encoder.stonnx as st
    monkeypatch.setattr(st, "SparseEncoderONNX", FakeONNX)
    enc = spl.SpladeEncoder(model="m/coo", backend="onnx", device="cpu")
    # Fake torch COO row
    class TorchCOORow:
        def coalesce(self): return self
        def indices(self):
            # emulate 2 x nnz; first row would be row ids, we only read [0]
            return np.asarray([[2, 4]], dtype=np.int64)
        def values(self):
            return np.asarray([0.3, 0.7], dtype=np.float32)
    idx, vals = enc._row_to_indices_values(TorchCOORow())
    assert idx == [2, 4]
    assert np.allclose(vals, [0.3, 0.7], rtol=1e-6, atol=1e-6)

def test_splade_typeerror_fallbacks_for_encode_and_dims(monkeypatch, tmp_path):
    # Work in a temp CWD so ./models is under tmp
    monkeypatch.chdir(tmp_path)
    # Ensure required ONNX path exists so prefetch is not called
    target = tmp_path / "models" / "m__type" / "onnx" / "int8"
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.onnx").write_bytes(b"x")
    # Fake ONNX encoder whose encode() does NOT accept 'device' kwarg -> triggers TypeError fallbacks
    class FakeONNXNoDevice:
        def __init__(self, *a, **k): self.tokenizer = None
        def to(self, device): return self
        def encode(self, texts):
            # Return CSR for single or multi to exercise both branches
            import scipy.sparse as sp
            if isinstance(texts, list) and len(texts) == 1:
                return sp.csr_matrix(([1.0, 2.0], ([0, 0], [0, 2])), shape=(1, 5))
            # 2 rows, 5 dims
            data = [1.0, 2.0, 3.0]
            rows = [0, 1, 1]
            cols = [1, 0, 3]
            return sp.csr_matrix((data, (rows, cols)), shape=(2, 5))
    # Avoid any download
    monkeypatch.setattr(spl, "prefetch", lambda *a, **k: str(tmp_path / "models"))
    import recollex.encoder.stonnx as st
    monkeypatch.setattr(st, "SparseEncoderONNX", FakeONNXNoDevice)
    enc = spl.SpladeEncoder(model="m/type", backend="onnx", device="cpu")
    # Dims determined via CSR shape after TypeError path
    assert enc.dims == 5
    # encode() uses TypeError fallback internally
    si, sv = enc.encode("hello")
    assert si == [0, 2] and all(isinstance(x, float) for x in sv)
    # encode_many() uses TypeError fallback and iterates CSR rows
    rows = enc.encode_many(["a", "b"])
    assert rows[0][0] and rows[1][0]
