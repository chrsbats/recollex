from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from recollex.abcs import MetadataStore, SegmentReader


class NpySegmentReader(SegmentReader):
    """
    SegmentReader backed by .npy CSR arrays on disk (mmap'ed).
    Files:
      - indptr.npy   (int64, len=N+1)
      - indices.npy  (int32, len=nnz)
      - data.npy     (float32, len=nnz)
      - row_ids.npy  (bytes utf-8, fixed-width 'S' dtype) OR int64
    """

    def __init__(self) -> None:
        self._seg_path: Optional[Path] = None
        self._indptr: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._data: Optional[np.ndarray] = None
        self._row_ids: Optional[np.ndarray] = None
        self._n_rows: int = 0
        self._dims: int = 0

    def open(self, segment_path: Union[str, Path]) -> None:
        p = Path(segment_path)
        self._seg_path = p
        self._indptr = np.load(p / "indptr.npy", mmap_mode="r")
        self._indices = np.load(p / "indices.npy", mmap_mode="r")
        self._data = np.load(p / "data.npy", mmap_mode="r")
        self._row_ids = np.load(p / "row_ids.npy", mmap_mode="r")

        if self._indptr.dtype != np.int64:
            raise TypeError("indptr.npy must be int64")
        if self._indices.dtype != np.int32:
            raise TypeError("indices.npy must be int32")
        if self._data.dtype != np.float32:
            raise TypeError("data.npy must be float32")
        if self._indices.shape[0] != self._data.shape[0]:
            raise ValueError("indices/data length mismatch")

        self._n_rows = int(self._indptr.shape[0] - 1)
        if self._row_ids.shape[0] != self._n_rows:
            raise ValueError("row_ids length must equal N rows")

        # Minimal required dims from indices content
        self._dims = int(self._indices.max() + 1) if self._indices.size else 0

    @property
    def indptr(self) -> "np.ndarray":
        assert self._indptr is not None
        return self._indptr

    @property
    def indices(self) -> "np.ndarray":
        assert self._indices is not None
        return self._indices

    @property
    def data(self) -> "np.ndarray":
        assert self._data is not None
        return self._data

    @property
    def row_ids(self) -> "np.ndarray":
        assert self._row_ids is not None
        return self._row_ids

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def dims(self) -> int:
        return self._dims

    def _decode_doc_id(self, x: Union[np.generic, bytes]) -> str:
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8")
        dt = getattr(self._row_ids, "dtype", None)
        if dt is not None and np.issubdtype(dt, np.integer):
            return str(int(x))
        try:
            return x.decode("utf-8")  # type: ignore[attr-defined]
        except Exception:
            return str(x)

    def doc_id_for_row(self, row_offset: int) -> str:
        rid = self.row_ids[row_offset]
        return self._decode_doc_id(rid)

    def doc_ids_for_rows(self, row_offsets: Sequence[int]) -> List[str]:
        rids = self.row_ids[list(row_offsets)]
        return [self._decode_doc_id(r) for r in rids.tolist()]

    def close(self) -> None:
        self._indptr = None
        self._indices = None
        self._data = None
        self._row_ids = None
        self._seg_path = None
        self._n_rows = 0
        self._dims = 0


def _coerce_row_ids_array(row_ids: Sequence[Union[str, int, bytes]]) -> np.ndarray:
    """
    Convert row_ids to a numpy array suitable for mmap:
      - If all ints -> int64
      - Else -> fixed-width bytes ('S{maxlen}') with utf-8 encoding
    """
    if not row_ids:
        return np.asarray([], dtype=np.int64)
    all_int = all(isinstance(x, int) for x in row_ids)
    if all_int:
        return np.asarray(row_ids, dtype=np.int64)
    b = [x if isinstance(x, (bytes, np.bytes_)) else str(x).encode("utf-8") for x in row_ids]
    return np.asarray(b, dtype=np.bytes_)


def write_segments(
    segment_path: Union[str, Path],
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_ids: Sequence[Union[str, int, bytes]],
) -> Dict[str, int]:
    """
    Write CSR arrays and row_ids to segment_path as .npy files.

    Returns metadata: {'n_rows': N, 'nnz': nnz, 'dims': D_min}
    """
    p = Path(segment_path)
    p.mkdir(parents=True, exist_ok=True)

    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int32)
    data = np.asarray(data, dtype=np.float32)
    row_ids_arr = _coerce_row_ids_array(row_ids)

    if indices.shape[0] != data.shape[0]:
        raise ValueError("indices and data must be same length")
    if indptr.shape[0] == 0 or indptr.dtype != np.int64:
        raise ValueError("indptr must be int64 and non-empty")
    n_rows = int(indptr.shape[0] - 1)
    if row_ids_arr.shape[0] != n_rows:
        raise ValueError("row_ids length must equal N rows")
    if indptr[-1] != indices.shape[0]:
        raise ValueError("indptr[-1] must equal nnz (len(indices))")

    np.save(p / "indptr.npy", indptr, allow_pickle=False)
    np.save(p / "indices.npy", indices, allow_pickle=False)
    np.save(p / "data.npy", data, allow_pickle=False)
    np.save(p / "row_ids.npy", row_ids_arr, allow_pickle=False)

    dims_min = int(indices.max() + 1) if indices.size else 0
    return {"n_rows": n_rows, "nnz": int(indices.shape[0]), "dims": dims_min}


def open_segment(segment_path: Union[str, Path], dims: Optional[int] = None) -> NpySegmentReader:
    """
    Open a segment with mmap. If dims is provided, validate dims >= max_index+1
    and set reader.dims to the provided value (to align with manifest/encoder).
    """
    reader = NpySegmentReader()
    reader.open(segment_path)

    needed = int(reader.indices.max() + 1) if reader.indices.size else 0
    if dims is not None:
        if dims < needed:
            raise ValueError(f"dims ({dims}) smaller than required ({needed}) for {segment_path}")
        reader._dims = int(dims)
    return reader


def group_by_segment(store: MetadataStore, doc_ids: Iterable[str]) -> Dict[str, List[int]]:
    """
    Map doc_ids -> {segment_id: [row_offsets]} using the MetadataStore.
    Missing doc_ids are ignored.
    """
    grouped: Dict[str, List[int]] = {}
    for did in doc_ids:
        rec = store.get_doc(did)
        if rec is None:
            continue
        grouped.setdefault(rec.segment_id, []).append(int(rec.row_offset))
    return grouped
