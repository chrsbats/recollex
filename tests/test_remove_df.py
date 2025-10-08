from typing import List
import pytest

from recollex.engine import Recollex


def test_df_and_term_bitmap_decrement_single(index: Recollex, now):
    # Prepare two docs; term 1 appears in doc1 only; term 2 appears in both
    docs = [
        {"doc_id": 1, "indices": [1, 2], "data": [1.0, 0.5], "text": "d1", "tags": None, "seq": now()},
        {"doc_id": 2, "indices": [2],     "data": [1.0],      "text": "d2", "tags": None, "seq": now() + 1},
    ]
    index.add_many(docs, dims=8)

    # Pre-conditions after add_many
    bm_t1 = index._get_bitmap("term:1")
    bm_t2 = index._get_bitmap("term:2")
    assert set(bm_t1) == {1}
    assert set(bm_t2) == {1, 2}
    assert index._store.get_stat("term_df:1") == 1
    assert index._store.get_stat("term_df:2") == 2

    # Remove doc 1 (single-item branch)
    index.remove(1)

    # Post-conditions: doc 1 removed from both term bitmaps; DF decremented
    bm_t1_after = index._get_bitmap("term:1")
    bm_t2_after = index._get_bitmap("term:2")
    assert list(bm_t1_after) == []                  # term:1 no longer has doc 1
    assert set(bm_t2_after) == {2}                  # term:2 keeps only doc 2

    assert index._store.get_stat("term_df:1") == 0  # decremented from 1 -> 0
    assert index._store.get_stat("term_df:2") == 1  # decremented from 2 -> 1


def test_df_and_term_bitmap_decrement_batch_and_floor(index: Recollex, now):
    # Prepare two docs, both on term 2
    docs = [
        {"doc_id": 10, "indices": [2], "data": [1.0], "text": "a", "tags": None, "seq": now()},
        {"doc_id": 11, "indices": [2], "data": [1.0], "text": "b", "tags": None, "seq": now() + 1},
    ]
    index.add_many(docs, dims=8)

    # Pre-conditions
    assert set(index._get_bitmap("term:2")) == {10, 11}
    assert index._store.get_stat("term_df:2") == 2

    # Batch remove both docs
    index.remove([10, 11])

    # Bitmaps emptied; DF goes to 0
    assert list(index._get_bitmap("term:2")) == []
    assert index._store.get_stat("term_df:2") == 0

    # Removing again should not decrement below 0 (membership check prevents double-decrement)
    index.remove([10, 11])
    assert index._store.get_stat("term_df:2") == 0
