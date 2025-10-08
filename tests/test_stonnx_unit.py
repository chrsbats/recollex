import numpy as np
from pathlib import Path
import types
import pytest

import recollex.encoder.stonnx as st


class FakeInput:
    def __init__(self, name): self.name = name
class FakeOutput:
    def __init__(self, name): self.name = name


def _mk_tokenizer(vocab_size=10, with_am=True, with_tt=False):
    class Tok:
        def __init__(self): self.vocab_size = vocab_size
        def __call__(self, batch, padding=True, truncation=True, return_tensors="np", return_token_type_ids=True):
            B = len(batch)
            T = 4
            ids = np.arange(1, T + 1, dtype=np.int64)[None, :].repeat(B, axis=0)
            out = {"input_ids": ids}
            if with_am:
                am = np.ones_like(ids, dtype=np.int64)
                am[:, -1] = 0  # mask last token
                out["attention_mask"] = am
            if with_tt:
                out["token_type_ids"] = np.zeros_like(ids, dtype=np.int64)
            return out
    return Tok()


def test_stonnx_encode_3d_logits_topk_and_splade(monkeypatch, tmp_path):
    # Create fake model file
    mdir = tmp_path / "onnx"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.onnx").write_bytes(b"x")
    # Fake ORT session: inputs include ids/am/tt/pos; outputs produce (B,T,V) logits
    class Sess:
        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("attention_mask"), FakeInput("token_type_ids"), FakeInput("position_ids")]
        def get_outputs(self):
            return [FakeOutput("logits")]
        def run(self, _none, ort_inputs):
            B, T = ort_inputs["input_ids"].shape
            V = 6
            # simple pattern: increasing logits; last token masked by attention in tokenizer
            logits = np.tile(np.linspace(0.0, 1.0, T * V, dtype=np.float32), (B, 1)).reshape(B, T, V)
            return [logits]
    monkeypatch.setattr(st.ort, "InferenceSession", lambda path, providers=None: Sess())
    monkeypatch.setattr(st, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _mk_tokenizer(vocab_size=6, with_am=True, with_tt=False)))
    enc = st.SparseEncoderONNX(model_dir=str(mdir), providers=["CPUExecutionProvider"], tokenizer_id=str(mdir), max_active_dims=2)
    rows = enc.encode(["a", "b"])
    assert isinstance(rows, list) and len(rows) == 2
    for idx, vals in rows:
        assert len(idx) == len(vals) == 2
        assert idx == sorted(idx)
    # dims property and to()/eval() no-ops
    assert enc.dims == 6
    assert enc.to("cpu") is enc
    assert enc.eval() is enc

def test_stonnx_encode_2d_logits_fallback(monkeypatch, tmp_path):
    mdir = tmp_path / "onnx3"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.onnx").write_bytes(b"x")
    class Sess:
        def get_inputs(self):
            return [FakeInput("input_ids")]
        def get_outputs(self):
            return [FakeOutput("logits")]
        def run(self, _none, ort_inputs):
            B = ort_inputs["input_ids"].shape[0]
            V = 4
            # 2D logits, with negatives (will relu)
            logits = np.array([[0.0, -0.5, 0.2, 0.0]] * B, dtype=np.float32)
            return [logits]
    monkeypatch.setattr(st.ort, "InferenceSession", lambda path, providers=None: Sess())
    monkeypatch.setattr(st, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _mk_tokenizer(vocab_size=4)))
    enc = st.SparseEncoderONNX(model_dir=str(mdir), providers=["CPUExecutionProvider"], tokenizer_id=str(mdir), max_active_dims=None)
    rows = enc.encode(["y"])
    assert rows[0][0] == [2]
    # Compare with tolerance to account for float32 rounding
    assert np.allclose(rows[0][1], [0.2], rtol=1e-6, atol=1e-6)


def test_stonnx_ort_missing_raises(monkeypatch, tmp_path):
    mdir = tmp_path / "onnx_missing"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.onnx").write_bytes(b"x")
    # Simulate missing onnxruntime
    monkeypatch.setattr(st, "ort", None)
    with pytest.raises(RuntimeError):
        st.SparseEncoderONNX(model_dir=str(mdir), providers=["CPUExecutionProvider"], tokenizer_id=str(mdir))

def test_stonnx_topk_zero_and_threshold_edge_paths(monkeypatch, tmp_path):
    mdir = tmp_path / "onnx_more"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.onnx").write_bytes(b"x")
    class Sess:
        def __init__(self, logits):
            self._logits = logits
        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("attention_mask")]
        def get_outputs(self):
            return [FakeOutput("logits")]
        def run(self, _none, ort_inputs):
            return [self._logits]
    # Case A: (B,T,V) all negatives -> zero positives in top-k branch
    logits_neg = -np.ones((1, 4, 5), dtype=np.float32)
    monkeypatch.setattr(st.ort, "InferenceSession", lambda path, providers=None: Sess(logits_neg))
    monkeypatch.setattr(st, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _mk_tokenizer(vocab_size=5, with_am=True)))
    enc_a = st.SparseEncoderONNX(model_dir=str(mdir), providers=["CPUExecutionProvider"], tokenizer_id=str(mdir), max_active_dims=3)
    rows_a = enc_a.encode(["z"])
    assert rows_a[0] == ([], [])
    # Case B: two positives only, k larger than positives -> <= topk branch
    logits_two = np.zeros((1, 4, 6), dtype=np.float32)
    logits_two[:, :, 1] = 0.1
    logits_two[:, :, 4] = 0.3
    monkeypatch.setattr(st.ort, "InferenceSession", lambda path, providers=None: Sess(logits_two))
    monkeypatch.setattr(st, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _mk_tokenizer(vocab_size=6, with_am=True)))
    enc_b = st.SparseEncoderONNX(model_dir=str(mdir), providers=["CPUExecutionProvider"], tokenizer_id=str(mdir), max_active_dims=10)
    rows_b = enc_b.encode(["q"])
    b_idx, b_vals = rows_b[0]
    assert set(b_idx) == {1, 4}
    assert all(v > 0 for v in b_vals)
