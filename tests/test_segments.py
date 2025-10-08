from pathlib import Path

import numpy as np
import pytest

from recollex.io.segments import write_segments, open_segment, NpySegmentReader, group_by_segment
from recollex.io.sqlite_store import SQLiteMetadataStore
from recollex.abcs import DocRecord


def test_write_open_segment_with_bytes_row_ids(tmp_path: Path):
    seg = tmp_path / "seg_001"
    indptr = np.asarray([0, 2, 4], dtype=np.int64)
    indices = np.asarray([1, 3, 0, 3], dtype=np.int32)
    data = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    row_ids = [b"10", b"11"]

    meta = write_segments(seg, indptr, indices, data, row_ids)
    assert meta["n_rows"] == 2
    assert meta["nnz"] == 4
    assert meta["dims"] == 4

    reader = open_segment(seg, dims=5)
    assert isinstance(reader, NpySegmentReader)
    assert reader.n_rows == 2
    assert reader.dims == 5  # overridden by open_segment
    assert reader.indptr.dtype == np.int64
    assert reader.indices.dtype == np.int32
    assert reader.data.dtype == np.float32

    # row id decoding
    assert reader.doc_id_for_row(0) == "10"
    assert reader.doc_ids_for_rows([0, 1]) == ["10", "11"]

    reader.close()


def test_write_open_segment_with_int_row_ids_and_dim_check(tmp_path: Path):
    seg = tmp_path / "seg_002"
    indptr = np.asarray([0, 1], dtype=np.int64)
    indices = np.asarray([7], dtype=np.int32)
    data = np.asarray([1.0], dtype=np.float32)
    row_ids = [100]  # int -> will be saved as int64

    _ = write_segments(seg, indptr, indices, data, row_ids)
    # dims smaller than needed (max index 7 => need 8) should fail
    with pytest.raises(ValueError):
        _ = open_segment(seg, dims=7)
    # OK when dims >= needed
    reader = open_segment(seg, dims=8)
    assert reader.dims == 8
    assert reader.doc_id_for_row(0) == "100"
    reader.close()


def test_group_by_segment(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "meta.sqlite")
    with store.transaction():
        store.upsert_doc(DocRecord(doc_id="1", segment_id="a", row_offset=0, seq=1))
        store.upsert_doc(DocRecord(doc_id="2", segment_id="a", row_offset=1, seq=2))
        store.upsert_doc(DocRecord(doc_id="3", segment_id="b", row_offset=0, seq=3))
    grouped = group_by_segment(store, ["2", "1", "3", "missing"])
    assert grouped == {"a": [1, 0], "b": [0]}
    store.close()
