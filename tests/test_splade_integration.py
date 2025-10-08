from __future__ import annotations

import pytest

from recollex.encoder.splade import SpladeEncoder
import pytest




def test_splade_encode_real_model_single(splade_enc):
    enc = splade_enc  # defaults to prithivida/Splade_PP_en_v2, backend=onnx
    idxs, vals = enc.encode("hello world")

    assert isinstance(enc.dims, int) and enc.dims > 0
    assert isinstance(idxs, list) and isinstance(vals, list)
    assert len(idxs) == len(vals) and len(idxs) > 0
    assert all(isinstance(i, int) and 0 <= i < enc.dims for i in idxs)
    assert all(isinstance(v, float) for v in vals)


def test_splade_encode_real_model_many(splade_enc):
    enc = splade_enc
    texts = ["a test query", "another query", "and one more"]
    outs = enc.encode_many(texts)

    assert len(outs) == len(texts)
    for idxs, vals in outs:
        assert len(idxs) == len(vals) and len(idxs) > 0
        assert all(isinstance(i, int) and 0 <= i < enc.dims for i in idxs)
        assert all(isinstance(v, float) for v in vals)
