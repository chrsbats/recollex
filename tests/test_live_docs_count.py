from pyroaring import BitMap as Roaring


def test_live_docs_count_tracks_adds_and_removes(index, now):
    d1 = index.add("x1", tags=["t"], timestamp=now())
    d2 = index.add("x2", tags=["t"], timestamp=now())
    d3 = index.add("x3", tags=["t"], timestamp=now())

    cnt = index._store.get_stat("live_docs_count")
    assert int(cnt or 0) == 3

    blob = index._store.get_bitmap("live_docs")
    bm = Roaring.deserialize(blob.encode("latin-1")) if isinstance(blob, str) else Roaring.deserialize(blob)
    assert len(bm) == 3

    index.remove(d2)
    cnt2 = index._store.get_stat("live_docs_count")
    assert int(cnt2 or 0) == 2
    blob2 = index._store.get_bitmap("live_docs")
    bm2 = Roaring.deserialize(blob2.encode("latin-1")) if isinstance(blob2, str) else Roaring.deserialize(blob2)
    assert len(bm2) == 2
    assert int(d2) not in bm2

    index.remove([d1, d3])
    cnt3 = index._store.get_stat("live_docs_count")
    assert int(cnt3 or 0) == 0
    blob3 = index._store.get_bitmap("live_docs")
    bm3 = Roaring.deserialize(blob3.encode("latin-1")) if isinstance(blob3, str) else Roaring.deserialize(blob3)
    assert len(bm3) == 0
