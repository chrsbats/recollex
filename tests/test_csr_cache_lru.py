from pathlib import Path

import pytest

from recollex.engine import Recollex


def _mk_seg(rx: Recollex, seg_name: str, tid: int, did: int, seq: int, tag: str):
    # One-doc segment with a single term tid and a tag to scope searches
    docs = [{
        "doc_id": did,
        "indices": [tid],
        "data": [1.0],
        "text": f"doc-{did}",
        "tags": [tag],
        "seq": seq,
    }]
    rx.add_many(docs, segment_id=seg_name, dims=8)


def test_seg_reader_cache_lru_eviction(tmp_path: Path):
    # seg_cache_max=2 so the third distinct segment should evict the oldest
    rx = Recollex.open(tmp_path / "idx", seg_cache_max=2, csr_cache_max=16, csr_ram_limit_bytes=1 << 30)

    # Create three segments; all share term=1 to be scored in one query
    _mk_seg(rx, "seg_000", tid=1, did=1, seq=10, tag="scope:a")
    _mk_seg(rx, "seg_001", tid=1, did=2, seq=11, tag="scope:a")
    _mk_seg(rx, "seg_002", tid=1, did=3, seq=12, tag="scope:a")

    q_terms = [(1, 1.0)]

    # Search all (no tag scope) to touch all three readers
    _ = rx.search_terms(q_terms, k=5, profile="rag")

    # Only two most recently used readers should remain
    keys = set(rx._seg_cache.keys())
    assert len(keys) == 2
    # The oldest "seg_000" should have been evicted
    assert "seg_000" not in keys
    assert {"seg_001", "seg_002"}.issubset(keys)


def test_csr_cache_lru_count_and_touch(tmp_path: Path):
    # csr_cache_max=1 to force single-entry cache; large RAM limit to avoid RAM-based eviction here
    rx = Recollex.open(tmp_path / "idx2", seg_cache_max=8, csr_cache_max=1, csr_ram_limit_bytes=1 << 30)

    # Two separate segments, both with the same single term=2
    _mk_seg(rx, "seg_000", tid=2, did=10, seq=100, tag="seg=A")
    _mk_seg(rx, "seg_001", tid=2, did=11, seq=200, tag="seg=B")

    q_terms = [(2, 1.0)]

    # Scope to seg=A to populate CSR for seg_000
    _ = rx.search_terms(q_terms, k=5, profile="rag", tags_all_of=["seg=A"])
    entries = rx._csr_cache["entries"]
    keys = set(entries.keys())
    assert keys == {("seg_000", 8)}

    # Now scope to seg=B; LRU should evict seg_000 and keep only seg_001
    _ = rx.search_terms(q_terms, k=5, profile="rag", tags_all_of=["seg=B"])
    entries = rx._csr_cache["entries"]
    keys = set(entries.keys())
    assert keys == {("seg_001", 8)}

    # Touch seg_000 again; seg_001 should be evicted and seg_000 re-cached
    _ = rx.search_terms(q_terms, k=5, profile="rag", tags_all_of=["seg=A"])
    entries = rx._csr_cache["entries"]
    keys = set(entries.keys())
    assert keys == {("seg_000", 8)}

    # total_size should be non-negative and reflect the single entry
    assert int(rx._csr_cache["total_size"]) >= 0
    assert len(rx._csr_cache["entries"]) <= 1


def test_csr_cache_ram_limit_eviction(tmp_path: Path):
    # Set an extremely small RAM limit so any insertion triggers eviction
    rx = Recollex.open(tmp_path / "idx3", seg_cache_max=8, csr_cache_max=16, csr_ram_limit_bytes=1)

    _mk_seg(rx, "seg_000", tid=3, did=20, seq=1000, tag="tiny")

    q_terms = [(3, 1.0)]
    _ = rx.search_terms(q_terms, k=5, profile="rag", tags_all_of=["tiny"])

    # After insertion, RAM-based eviction should clear entries
    assert len(rx._csr_cache["entries"]) == 0
    assert int(rx._csr_cache["total_size"]) >= 0
