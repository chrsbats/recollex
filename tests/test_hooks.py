import numpy as np
from scipy.sparse import csr_matrix

from recollex.hooks import score_csr_slice, evict_lru, rank_merge_heap, rank_merge_recent


def test_score_csr_slice_from_arrays():
    # X: 3x5
    X = csr_matrix(
        (
            np.array([1.0, 0.5, 2.0], dtype=np.float32),
            np.array([0, 2, 4], dtype=np.int32),
            np.array([0, 1, 2, 3], dtype=np.int64),
        ),
        shape=(3, 5),
        dtype=np.float32,
    )
    q = csr_matrix(([1.0, 1.0], ([0, 0], [0, 4])), shape=(1, 5), dtype=np.float32)
    # Provide raw arrays in segment_ctx
    ctx = {
        "indptr": X.indptr.astype(np.int64),
        "indices": X.indices.astype(np.int32),
        "data": X.data.astype(np.float32),
        "dims": 5,
    }
    # rows 0 and 2
    scores = score_csr_slice(q, ctx, [0, 2])
    # row0 has col0=1 -> score 1; row2 has col4=2 -> score 2
    assert scores == [(0, 1.0), (2, 2.0)]


def test_rank_merges_and_evict_lru():
    per_seg = {"a": [(0, 0.9), (2, 0.1)], "b": [(1, 0.8)]}
    merged = rank_merge_heap(per_seg, k=2)
    assert merged[0][0] in ("a", "b")
    assert merged[0][2] >= merged[1][2]

    per_seg_recent = {"a": [(0, 100), (2, 10)], "b": [(1, 80)]}
    merged_recent = rank_merge_recent(per_seg_recent, k=2)
    assert [m[2] for m in merged_recent] == [100, 80]

    # evict_lru: count-based
    cache = {
        "x": {"last_used": 1, "size": 10},
        "y": {"last_used": 2, "size": 10},
        "z": {"last_used": 3, "size": 10},
    }
    evicted = evict_lru(cache, max_items=2)
    assert evicted == ["x"]

    # evict_lru: ram_limit_bytes (oldest-first; may evict all)
    cache2 = {
        "entries": {
            "a": {"last_used": 1, "size": 10},
            "b": {"last_used": 2, "size": 10},
            "c": {"last_used": 3, "size": 50},
        },
        "total_size": 70,
        "ram_limit_bytes": 40,
    }
    evicted2 = evict_lru(cache2, max_items=999)
    assert set(evicted2) == {"a", "b", "c"}

from pyroaring import BitMap as Roaring
from recollex.hooks import (
    filter_policy_default,
    candidate_supplier_default,
    score_accumulator,
)

def test_filter_policy_default_no_terms_returns_empty():
    must, should = filter_policy_default([])
    assert must == [] and should == []

def test_filter_policy_default_min_must_without_bitmaps():
    # Without get_bitmap/base_bitmap, falls back to top min_must by weight*idf
    q_terms = [(1, 1.0), (2, 0.5), (3, 0.1)]
    must, should = filter_policy_default(q_terms, knobs={"min_must": 2, "should_cap": 1})
    assert len(must) == 2
    assert len(should) == 1
    assert set(must).issubset({1, 2, 3})

def test_candidate_supplier_default_handles_missing_postings_and_should_union():
    base = Roaring([1, 2, 3, 4, 5])
    # must term 10 has postings {2,3}, should terms: 20 has {3,4}, 30 missing (None)
    def get_bitmap(name: str):
        if name == "term:10":
            return Roaring([2, 3])
        if name == "term:20":
            return Roaring([3, 4])
        if name == "term:30":
            return None
        return Roaring()
    out = candidate_supplier_default(
        must_terms=[10],
        should_terms=[20, 30],
        base_bitmap=base,
        get_bitmap=get_bitmap,
        budget=None,
    )
    # base ∩ MUST -> {2,3}; SHOULD union -> {3,4}; intersect final -> {3}
    assert set(out) == {3}

def test_score_accumulator_combines_weights_and_filters_rows():
    q = [(1, 2.0), (2, 1.0)]
    postings = {
        1: [(0, 0.5), (1, 1.0)],
        2: [(1, 2.0), (2, 3.0)],
    }
    # Only rows {0,2}
    out = score_accumulator(q, postings, row_offsets=[0, 2])
    dd = dict(out)
    # row0: only term1 contributes 2.0*0.5 = 1.0
    # row2: only term2 contributes 1.0*3.0 = 3.0
    assert dd[0] == 1.0 and dd[2] == 3.0
