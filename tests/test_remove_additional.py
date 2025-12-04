import multiprocessing as mp
from pathlib import Path

import pytest

from recollex.engine import Recollex
from recollex.io.segments import group_by_segment


def test_remove_by_string_and_mixed_ids(index: Recollex, now):
    # Create three distinct docs
    d1 = index.add("s1", tags=["topic:x"], timestamp=now())
    d2 = index.add("s2", tags=["topic:x"], timestamp=now() + 1)
    d3 = index.add("s3", tags=["topic:x"], timestamp=now() + 2)

    # Remove by string doc_id
    index.remove(str(d1))
    res1 = index.search("s1", k=5)
    assert all(h["doc_id"] != str(d1) for h in res1)

    # Mixed str/int batch
    index.remove([str(d2), d3])
    res2 = index.search("s2", k=5)
    res3 = index.search("s3", k=5)
    assert all(h["doc_id"] != str(d2) for h in res2)
    assert all(h["doc_id"] != str(d3) for h in res3)

    # All removed; live_docs_count should be 0
    cnt = int(index._store.get_stat("live_docs_count") or 0)
    assert cnt == 0


def test_remove_nonexistent_ids_is_noop(index: Recollex, now):
    d = index.add("keep", tags=["t:z"], timestamp=now())
    before_cnt = int(index._store.get_stat("live_docs_count") or 0)

    # Remove int non-existent
    index.remove(999999)
    # Remove str non-existent (numeric-looking)
    index.remove("123456789")
    # Remove non-numeric string -> should also be a no-op
    index.remove("not-a-number")

    after_cnt = int(index._store.get_stat("live_docs_count") or 0)
    assert after_cnt == before_cnt

    # Doc still present
    hits = index.search("keep", k=5)
    assert any(h["doc_id"] == str(d) for h in hits)


def test_tag_bitmaps_updated_on_remove(index: Recollex, now):
    # Two docs sharing the same tag; one with an extra tag
    t = "topic:ml"
    d1 = index.add("ml1", tags=[t, "group:a"], timestamp=now())
    d2 = index.add("ml2", tags=[t], timestamp=now() + 1)

    # Pre-condition: tag bitmap contains both numeric doc ids
    bm_before = index._get_bitmap(f"tag:{t}")
    assert set(bm_before) == {int(d1), int(d2)}

    # Remove first doc -> tag bitmap drops it
    index.remove(d1)
    bm_mid = index._get_bitmap(f"tag:{t}")
    assert set(bm_mid) == {int(d2)}

    # Remove second -> tag bitmap empty
    index.remove(d2)
    bm_after = index._get_bitmap(f"tag:{t}")
    assert list(bm_after) == []


def test_remove_across_segments_and_grouping(tmp_path: Path, now):
    rx = Recollex.open(tmp_path / "idx_segments")

    # Create two segments explicitly
    docs_a = [
        {"doc_id": 1, "indices": [1], "data": [1.0], "text": "a1", "tags": ["scope:A"], "seq": now()},
        {"doc_id": 2, "indices": [1], "data": [1.0], "text": "a2", "tags": ["scope:A"], "seq": now() + 1},
    ]
    docs_b = [
        {"doc_id": 3, "indices": [1], "data": [1.0], "text": "b1", "tags": ["scope:B"], "seq": now() + 2},
    ]
    rx.add_many(docs_a, segment_id="seg_A", dims=8)
    rx.add_many(docs_b, segment_id="seg_B", dims=8)

    # All three match term 1
    q_terms = [(1, 1.0)]
    hits_all = rx.search_terms(q_terms, k=10, profile="rag")
    assert {h["doc_id"] for h in hits_all} == {"1", "2", "3"}

    # Remove one doc from seg_A
    rx.remove(2)

    # Remaining hits are 1 and 3
    hits_after = rx.search_terms(q_terms, k=10, profile="rag")
    ids_after = [h["doc_id"] for h in hits_after]
    assert set(ids_after) == {"1", "3"}

    # group_by_segment over survivors maps to correct segments
    grouped = group_by_segment(rx._store, ["1", "3"])
    # seg_A contains doc_id 1 at row_offset 0; seg_B contains doc_id 3 at row_offset 0
    assert set(grouped.keys()) == {"seg_A", "seg_B"}
    assert grouped["seg_A"] == [0]
    assert grouped["seg_B"] == [0]


def _remover(index_dir: str, ids):
    rx = Recollex.open(Path(index_dir))
    rx.remove(ids)


def test_remove_concurrency(tmp_path: Path, now):
    idx_dir = tmp_path / "idx_remove_concurrent"
    rx = Recollex.open(idx_dir)

    # Create three docs
    d1 = rx.add("c1", tags=["t"], timestamp=now())
    d2 = rx.add("c2", tags=["t"], timestamp=now() + 1)
    d3 = rx.add("c3", tags=["t"], timestamp=now() + 2)

    # Two processes remove disjoint/mixed sets
    ctx = mp.get_context("spawn")
    p1 = ctx.Process(target=_remover, args=(str(idx_dir), [str(d1)]))
    p2 = ctx.Process(target=_remover, args=(str(idx_dir), [d2, d3]))
    p1.start(); p2.start()
    p1.join(10); p2.join(10)
    assert p1.exitcode == 0 and p2.exitcode == 0

    # All removed; live_docs_count zero and searches empty
    rx2 = Recollex.open(idx_dir)
    cnt = int(rx2._store.get_stat("live_docs_count") or 0)
    assert cnt == 0
    for t in ["c1", "c2", "c3"]:
        assert rx2.search(t, k=5) == []
