from recollex.engine import Recollex


def test_segment_reader_reopen_on_higher_dims(index: Recollex, now):
    # Create a segment; manifest dims=8 (FakeEncoder)
    _ = index.add("hello", tags=["t"], timestamp=now())
    seg = index._manifest["segments"][0]["name"]

    # First open with dims=8
    r1 = index._open_segment(seg, dims=8)
    assert r1.dims == 8

    # Request higher dims: cached reader should be replaced with a new one at larger dims
    r2 = index._open_segment(seg, dims=16)
    assert r2.dims == 16
    # Subsequent calls reuse the enlarged reader
    r3 = index._open_segment(seg, dims=12)
    assert r3.dims == 16
