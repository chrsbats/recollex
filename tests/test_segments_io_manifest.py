import numpy as np
import pytest

from recollex.engine import Recollex
from recollex.io.segments import write_segments, open_segment


def test_segments_written_files_exist(index: Recollex, now):
    # First add creates seg_000
    index.add("hello", tags=["t:x"], timestamp=now())

    seg_name = index._manifest["segments"][0]["name"]
    seg_path = index._segments_dir / seg_name

    # Files exist
    for fname in ("indptr.npy", "indices.npy", "data.npy", "row_ids.npy"):
        assert (seg_path / fname).exists()

    # Shapes consistent
    indptr = np.load(seg_path / "indptr.npy")
    indices = np.load(seg_path / "indices.npy")
    data = np.load(seg_path / "data.npy")
    row_ids = np.load(seg_path / "row_ids.npy")

    assert indptr.dtype == np.int64
    assert indices.dtype == np.int32
    assert data.dtype == np.float32

    assert int(indptr[-1]) == len(indices) == len(data)
    n_rows = int(indptr.shape[0] - 1)
    assert len(row_ids) == n_rows


def test_add_many_dims_mismatch_raises(index: Recollex, now):
    # Establish manifest dims via first add (FakeEncoder.dims == 8)
    index.add("seed", tags=["t"], timestamp=now())

    # Next add_many with a different dims than manifest should raise
    docs = [
        {"doc_id": 1, "indices": [0], "data": [1.0], "text": "x", "tags": None, "seq": now()},
    ]
    with pytest.raises(ValueError):
        index.add_many(docs, dims=9)  # manifest dims=8 -> mismatch


def test_open_segment_dims_validation(index: Recollex, now):
    index.add("q", tags=["t"], timestamp=now())
    seg_name = index._manifest["segments"][0]["name"]
    seg_path = index._segments_dir / seg_name

    # Works when dims >= needed and matches FakeEncoder dims
    reader = open_segment(seg_path, dims=8)
    needed = int(reader.indices.max() + 1) if reader.indices.size else 0
    assert reader.dims == 8
    reader.close()

    # dims smaller than needed must fail
    if needed > 0:
        with pytest.raises(ValueError):
            _ = open_segment(seg_path, dims=needed - 1)


def test_row_ids_decoding_variants(tmp_path):
    # ints row_ids
    seg_int = tmp_path / "seg_int"
    indptr = np.asarray([0, 1, 2], dtype=np.int64)
    indices = np.asarray([0, 1], dtype=np.int32)
    data = np.asarray([1.0, 2.0], dtype=np.float32)
    row_ids_int = [100, 101]
    _ = write_segments(seg_int, indptr, indices, data, row_ids_int)

    r1 = open_segment(seg_int, dims=4)
    assert r1.doc_id_for_row(0) == "100"
    assert r1.doc_ids_for_rows([0, 1]) == ["100", "101"]
    r1.close()

    # bytes row_ids
    seg_bytes = tmp_path / "seg_bytes"
    row_ids_bytes = [b"200", b"201"]
    _ = write_segments(seg_bytes, indptr, indices, data, row_ids_bytes)

    r2 = open_segment(seg_bytes, dims=4)
    assert r2.doc_id_for_row(0) == "200"
    assert r2.doc_ids_for_rows([0, 1]) == ["200", "201"]
    r2.close()
