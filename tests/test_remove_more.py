from pathlib import Path
import multiprocessing as mp

from recollex.engine import Recollex
from recollex.bitmaps import LIVE_DOCS


def test_tag_dict_bitmaps_updated_on_remove(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_dict_tags")
    # Create two docs with dict tags directly via add_many
    docs = [
        {"doc_id": 1, "indices": [0], "data": [1.0], "text": "x", "tags": {"lang": "en"}, "seq": 1},
        {"doc_id": 2, "indices": [0], "data": [1.0], "text": "y", "tags": {"lang": "en"}, "seq": 2},
    ]
    rx.add_many(docs, dims=8)
    bm = rx._get_bitmap("tag:lang=en")
    assert set(bm) == {1, 2}

    rx.remove(1)
    assert set(rx._get_bitmap("tag:lang=en")) == {2}

    rx.remove(2)
    assert list(rx._get_bitmap("tag:lang=en")) == []


def test_remove_mixed_list_with_non_numeric_is_partial_noop(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_mixed_remove")
    a = rx.add("a", tags=["t"], timestamp=1)
    b = rx.add("b", tags=["t"], timestamp=2)
    rx.remove([a, "not-a-number"])
    assert all(h["doc_id"] != str(a) for h in rx.search("a", k=5))
    assert any(h["doc_id"] == str(b) for h in rx.search("b", k=5))


def _remover(index_dir: str, ids):
    rx = Recollex.open(Path(index_dir))
    rx.remove(ids)


def test_remove_concurrency_overlapping_ids(tmp_path: Path):
    idx_dir = tmp_path / "idx_remove_overlap_same"
    rx = Recollex.open(idx_dir)
    d = rx.add("c", tags=["t"], timestamp=1)

    ctx = mp.get_context("spawn")
    p1 = ctx.Process(target=_remover, args=(str(idx_dir), [d]))
    p2 = ctx.Process(target=_remover, args=(str(idx_dir), [d]))
    p1.start(); p2.start()
    p1.join(10); p2.join(10)
    assert p1.exitcode == 0 and p2.exitcode == 0

    rx2 = Recollex.open(idx_dir)
    assert int(rx2._store.get_stat("live_docs_count") or 0) == 0
    assert rx2.search("c", k=5) == []


def test_all_docs_bitmap_fallback_and_tombstones(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_all_docs_fallback")
    a = rx.add("x1", tags=["g"], timestamp=1)
    b = rx.add("x2", tags=["g"], timestamp=2)

    # Force fallback (no live_docs)
    rx._store.delete_bitmap(LIVE_DOCS)
    bm = rx._all_docs_bitmap()
    assert set(bm) == {int(a), int(b)}

    # Remove one; ensure fallback reflects survivor
    rx.remove(a)
    rx._store.delete_bitmap(LIVE_DOCS)
    bm2 = rx._all_docs_bitmap()
    assert set(bm2) == {int(b)}


def test_remove_by_dry_run_returns_count_without_deleting(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_remove_dry_run")
    d1 = rx.add("x", tags={"project": "p1"}, timestamp=1)
    d2 = rx.add("x", tags={"project": "p1"}, timestamp=2)

    # Dry run should report 2 but not delete anything
    n = rx.remove_by(all_of_tags=[{"project": "p1"}], dry_run=True)
    assert n == 2

    ids = {h["doc_id"] for h in rx.search("x", k=10)}
    assert {str(d1), str(d2)}.issubset(ids)
