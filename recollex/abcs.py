from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Dict, Iterator, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # for type hints only

# Serialized Roaring bitmap form (e.g., BMFilter.serialize() latin-1 string or bytes)
BitmapBlob = Union[str, bytes]


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    segment_id: str
    row_offset: int
    seq: int
    text: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None


class MetadataStore(ABC):
    """
    Stateful metadata backend (docs/bitmaps/stats/kv; transactions, caches).
    Implementations must ensure `transaction()` provides atomicity.
    """

    # --- Transactions ---
    @abstractmethod
    def transaction(self) -> ContextManager[None]:
        """with store.transaction(): ...  (commit on success, rollback on exception)"""
        raise NotImplementedError  # pragma: no cover

    # --- Documents ---
    @abstractmethod
    def upsert_doc(self, doc: DocRecord) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_doc(self, doc_id: str) -> Optional[DocRecord]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_docs_many(self, doc_ids: Sequence[str]) -> List[DocRecord]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def iter_docs_by_segment(self, segment_id: str) -> Iterator[DocRecord]:
        raise NotImplementedError  # pragma: no cover

    # --- Bitmaps ---
    @abstractmethod
    def get_bitmap(self, name: str) -> Optional[BitmapBlob]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def put_bitmap(self, name: str, blob: BitmapBlob) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def delete_bitmap(self, name: str) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def list_bitmaps(self, prefix: Optional[str] = None) -> List[str]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def iter_recent_doc_ids(self, limit: int) -> Iterator[str]:
        """Yield most-recent doc_ids by seq desc (LIMIT)."""
        raise NotImplementedError  # pragma: no cover

    # --- Stats (e.g., term DF) ---
    @abstractmethod
    def get_stat(self, key: str) -> Optional[int]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def put_stat(self, key: str, value: int) -> None:
        raise NotImplementedError  # pragma: no cover

    # --- Generic KV (TEXT/JSON) ---
    @abstractmethod
    def get_kv(self, key: str) -> Optional[str]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def put_kv(self, key: str, value: str) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def delete_kv(self, key: str) -> None:
        raise NotImplementedError  # pragma: no cover

    # --- Lifecycle ---
    def close(self) -> None:
        return None  # pragma: no cover


class SegmentReader(ABC):
    """
    Stateful reader for a segment's CSR arrays and row_id mapping.

    Expected files:
      - indptr.npy   (int64, len=N+1)
      - indices.npy  (int32, len=nnz)
      - data.npy     (float32, len=nnz)
      - row_ids.npy  (utf8 bytes or int64, len=N) mapping row_offset -> doc_id
    """

    @abstractmethod
    def open(self, segment_path: Union[str, Path]) -> None:
        """Open segment (prefer np.load(..., mmap_mode='r'))."""
        raise NotImplementedError  # pragma: no cover

    # --- CSR views ---
    @property
    @abstractmethod
    def indptr(self) -> "np.ndarray":
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def indices(self) -> "np.ndarray":
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def data(self) -> "np.ndarray":
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def row_ids(self) -> "np.ndarray":
        """Array mapping row_offset -> doc_id (utf8 bytes or object dtype)."""
        raise NotImplementedError  # pragma: no cover

    # --- Shapes/meta ---
    @property
    @abstractmethod
    def n_rows(self) -> int:
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def dims(self) -> int:
        """Vector dimensionality D (from manifest; validated vs encoder if present)."""
        raise NotImplementedError  # pragma: no cover

    # --- Lookups ---
    @abstractmethod
    def doc_id_for_row(self, row_offset: int) -> str:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def doc_ids_for_rows(self, row_offsets: Sequence[int]) -> List[str]:
        raise NotImplementedError  # pragma: no cover

    # --- Lifecycle ---
    def close(self) -> None:
        return None  # pragma: no cover
