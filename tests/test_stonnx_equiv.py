import pytest
import numpy as np
from pathlib import Path

from recollex.encoder.stonnx import SparseEncoderONNX
from tests.conftest import MODEL_DIR  # reuse the same model dir constant
from sentence_transformers import SparseEncoder as STSparseEncoder

def _find_quant():
    for q in ("int8", "fp16", "fp32"):
        if (MODEL_DIR / "onnx" / q / "model.onnx").exists():
            return q
    return None

def _rows_from_st_output(st_mat):
    import numpy as _np
    # SciPy CSR
    if hasattr(st_mat, "getrow") and hasattr(st_mat, "shape"):
        rows = []
        for i in range(st_mat.shape[0]):
            row = st_mat.getrow(i)
            rows.append((row.indices.tolist(), row.data.astype(_np.float32).tolist()))
        return rows
    # torch sparse COO (2D)
    if hasattr(st_mat, "coalesce") and hasattr(st_mat, "indices") and hasattr(st_mat, "values"):
        sp = st_mat.coalesce()
        idx = sp.indices()
        vals_t = sp.values()
        n_rows = int(sp.size(0)) if hasattr(sp, "size") else int(st_mat.shape[0])
        buckets = [[] for _ in range(n_rows)]
        r_ids = idx[0].tolist()
        c_ids = idx[1].tolist()
        try:
            v_list = vals_t.detach().cpu().numpy().astype(_np.float32).tolist()
        except Exception:
            try:
                v_list = vals_t.cpu().numpy().astype(_np.float32).tolist()
            except Exception:
                v_list = vals_t.tolist()
        for r, c, v in zip(r_ids, c_ids, v_list):
            buckets[int(r)].append((int(c), float(v)))
        out = []
        for pairs in buckets:
            if pairs:
                pairs.sort(key=lambda x: x[0])
                out.append(([c for c, _ in pairs], [v for _, v in pairs]))
            else:
                out.append(([], []))
        return out
    # list/tuple per row ([(indices, values, ...), ...] or dict-like)
    if isinstance(st_mat, (list, tuple)):
        out = []
        for item in st_mat:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                idx, vals = item[0], item[1]
                idx = idx.tolist() if hasattr(idx, "tolist") else list(idx)
                vals = vals.tolist() if hasattr(vals, "tolist") else list(vals)
                order = _np.argsort(_np.asarray(idx, dtype=_np.int64))
                idx = [int(_np.asarray(idx)[j]) for j in order]
                vals = [float(_np.asarray(vals, dtype=_np.float32)[j]) for j in order]
                out.append((idx, vals))
                continue
            if isinstance(item, dict) and "indices" in item and "values" in item:
                idx = item["indices"]
                vals = item["values"]
                idx = idx.tolist() if hasattr(idx, "tolist") else list(idx)
                vals = vals.tolist() if hasattr(vals, "tolist") else list(vals)
                order = _np.argsort(_np.asarray(idx, dtype=_np.int64))
                idx = [int(_np.asarray(idx)[j]) for j in order]
                vals = [float(_np.asarray(vals, dtype=_np.float32)[j]) for j in order]
                out.append((idx, vals))
                continue
            # Dense row fallback
            arr = _np.asarray(item)
            nz = _np.nonzero(arr > 0)[0]
            out.append((nz.astype(_np.int32).tolist(), arr[nz].astype(_np.float32).tolist()))
        return out
    # Dense array fallback
    try:
        arr = _np.asarray(st_mat)
        if arr.ndim == 2:
            rows = []
            for r in arr:
                nz = _np.nonzero(r > 0)[0]
                rows.append((nz.astype(_np.int32).tolist(), r[nz].astype(_np.float32).tolist()))
            return rows
    except Exception:
        pass
    raise TypeError("Unsupported reference encoder output")

def test_stonnx_equivalence_many():
    quant = _find_quant()
    if quant is None:
        pytest.skip("Prefetched model not found; run recollex-prefetch")

    texts = [
        "Redis quickstart",
        "Postgres tips and tricks",
        "database best practices",
        "postgres connection pool",
        "kubernetes deployment guide",
        "vector indexers for rag",
        "python typing generics",
        "machine learning sparse embeddings",
        "how to bake bread",
        "system design interview",
    ]

    # Reference encoder from sentence_transformers (ONNX backend for parity)
    st_enc = STSparseEncoder(
        str(MODEL_DIR),
        backend="onnx",
        device="cpu",
        model_kwargs={"subfolder": f"onnx/{quant}", "file_name": "model.onnx"},
    )
    mad = getattr(st_enc, "max_active_dims", getattr(st_enc, "logits_to_sparse_topk", None))

    # Our ONNX encoder
    onnx_enc = SparseEncoderONNX(
        model_dir=str(MODEL_DIR / "onnx" / quant),
        tokenizer_id=str(MODEL_DIR),
        providers=["CPUExecutionProvider"],
        max_active_dims=mad,
    )

    st_mat = st_enc.encode(texts)
    ref_rows = _rows_from_st_output(st_mat)

    test_rows = onnx_enc.encode(texts)

    # Compare each row strictly on indices and closely on values
    for (ref_idx, ref_val), (t_idx, t_val) in zip(ref_rows, test_rows):
        assert list(ref_idx) == list(t_idx)
        assert np.allclose(np.asarray(ref_val, dtype=np.float32),
                           np.asarray(t_val, dtype=np.float32),
                           rtol=1e-5, atol=1e-6)

def test_stonnx_equivalence_single():
    quant = _find_quant()
    if quant is None:
        pytest.skip("Prefetched model not found; run recollex-prefetch")

    text = "postgres connection pool tuning"

    st_enc = STSparseEncoder(
        str(MODEL_DIR),
        backend="onnx",
        device="cpu",
        model_kwargs={"subfolder": f"onnx/{quant}", "file_name": "model.onnx"},
    )
    mad = getattr(st_enc, "max_active_dims", getattr(st_enc, "logits_to_sparse_topk", None))

    onnx_enc = SparseEncoderONNX(
        model_dir=str(MODEL_DIR / "onnx" / quant),
        tokenizer_id=str(MODEL_DIR),
        providers=["CPUExecutionProvider"],
        max_active_dims=mad,
    )

    st_mat = st_enc.encode([text])
    ref_rows = _rows_from_st_output(st_mat)
    ref_idx, ref_val = ref_rows[0]

    test_rows = onnx_enc.encode([text])
    t_idx, t_val = test_rows[0]

    assert list(ref_idx) == list(t_idx)
    assert np.allclose(np.asarray(ref_val, dtype=np.float32),
                       np.asarray(t_val, dtype=np.float32),
                       rtol=1e-5, atol=1e-6)

def test_sparseencoder_onnx_output_types_and_shapes():
    quant = _find_quant()
    if quant is None:
        pytest.skip("Prefetched model not found; run recollex-prefetch")

    enc = SparseEncoderONNX(
        model_dir=str(MODEL_DIR / "onnx" / quant),
        tokenizer_id=str(MODEL_DIR),
        providers=["CPUExecutionProvider"],
        max_active_dims=350,
    )

    texts = ["one small step", "two docs", "three samples"]
    rows = enc.encode(texts)

    assert isinstance(rows, list)
    assert len(rows) == len(texts)
    for idx, vals in rows:
        assert isinstance(idx, list)
        assert isinstance(vals, list)
        assert len(idx) == len(vals)
        assert all(isinstance(i, int) for i in idx)
        assert all(isinstance(v, float) for v in vals)
        assert idx == sorted(idx)

    single = enc.encode("fourth example")
    assert isinstance(single, list) and len(single) == 1
    s_idx, s_vals = single[0]
    assert isinstance(s_idx, list) and isinstance(s_vals, list)




