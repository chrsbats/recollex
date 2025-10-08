import os
from pathlib import Path

import pytest
from pyroaring import BitMap as Roaring

from recollex.io.sqlite_store import SQLiteMetadataStore
from recollex.abcs import DocRecord


def test_sqlite_store_docs_kv_stats_bitmaps(tmp_path: Path):
    db = tmp_path / "meta.sqlite"
    store = SQLiteMetadataStore(db)

    # Docs CRUD + ordering for get_docs_many
    with store.transaction():
        store.upsert_doc(DocRecord(doc_id="1", segment_id="seg_a", row_offset=0, seq=10, text="a", tags={"k":"v"}))
        store.upsert_doc(DocRecord(doc_id="2", segment_id="seg_a", row_offset=1, seq=20, text="b", tags=["t"]))
        store.upsert_doc(DocRecord(doc_id="3", segment_id="seg_b", row_offset=0, seq=30, text=None, tags=None))

    d1 = store.get_doc("1")
    assert d1 is not None and d1.segment_id == "seg_a" and d1.row_offset == 0 and d1.text == "a" and d1.tags == {"k":"v"}

    got = store.get_docs_many(["3", "1", "2", "2"])  # dedup, preserve order
    assert [r.doc_id for r in got] == ["3", "1", "2"]

    # Iter by segment
    rows = list(store.iter_docs_by_segment("seg_a"))
    assert [r.doc_id for r in rows] == ["1", "2"]

    # KV
    store.put_kv("foo", "bar")
    assert store.get_kv("foo") == "bar"
    store.delete_kv("foo")
    assert store.get_kv("foo") is None

    # Stats
    assert store.get_stat("term_df:1") is None
    store.put_stat("term_df:1", 7)
    assert store.get_stat("term_df:1") == 7

    # Bitmaps: put/get/list/delete, plus cache path
    bm = Roaring([1, 2, 3])
    store.put_bitmap("tag:x", bm.serialize())
    blob = store.get_bitmap("tag:x")
    assert blob is not None
    # Deserialize from TEXT
    if isinstance(blob, str):
        got_bm = Roaring.deserialize(blob.encode("latin-1"))
    else:
        got_bm = Roaring.deserialize(blob)
    assert list(got_bm) == [1, 2, 3]

    names = store.list_bitmaps(prefix="tag:")
    assert "tag:x" in names

    store.delete_bitmap("tag:x")
    assert store.get_bitmap("tag:x") is None

    # transaction rollback
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.put_kv("will_rollback", "1")
            raise RuntimeError("boom")
    assert store.get_kv("will_rollback") is None

    store.close()
    assert os.path.exists(db)


def test_kv_roundtrip(tmp_path):
    db = tmp_path / "meta.sqlite"
    store = SQLiteMetadataStore(db)

    assert store.get_kv("missing") is None
    store.put_kv("a", "1")
    assert store.get_kv("a") == "1"
    store.delete_kv("a")
    assert store.get_kv("a") is None

    store.close()


def test_bitmap_roundtrip_and_listing(tmp_path):
    db = tmp_path / "meta.sqlite"
    store = SQLiteMetadataStore(db)

    bm = Roaring([1, 3, 5])
    blob_text = bm.serialize().decode("latin-1")  # store as TEXT

    name = "tag:test"
    store.put_bitmap(name, blob_text)

    got = store.get_bitmap(name)
    assert isinstance(got, str)
    assert got == blob_text

    # Deserialize and compare contents
    bm2 = Roaring.deserialize(got.encode("latin-1"))
    assert list(bm2) == [1, 3, 5]

    # Listing by prefix
    names = store.list_bitmaps(prefix="tag:")
    assert name in names

    store.close()


def test_iter_recent_doc_ids_order(tmp_path):
    db = tmp_path / "meta.sqlite"
    store = SQLiteMetadataStore(db)

    # Insert docs with increasing seq
    with store.transaction():
        store.upsert_doc(DocRecord(doc_id="1", segment_id="seg", row_offset=0, seq=100, text=None, tags=None))
        store.upsert_doc(DocRecord(doc_id="2", segment_id="seg", row_offset=1, seq=200, text=None, tags=None))
        store.upsert_doc(DocRecord(doc_id="3", segment_id="seg", row_offset=2, seq=300, text=None, tags=None))

    # Expect descending by seq; limit smaller than total
    recent2 = list(store.iter_recent_doc_ids(limit=2))
    assert recent2 == ["3", "2"]

    # Full list matches table order by seq desc
    recent_all = list(store.iter_recent_doc_ids(limit=10))
    seqs = [store.get_doc(did).seq for did in recent_all]
    assert seqs == sorted(seqs, reverse=True)

    store.close()
