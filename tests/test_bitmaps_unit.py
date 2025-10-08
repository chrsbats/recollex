try:
    # Some wheels export Roaring; others only BitMap
    from pyroaring import Roaring  # type: ignore
except ImportError:  # pragma: no cover
    from pyroaring import BitMap as Roaring  # type: ignore
from recollex.bitmaps import deserialize_bitmap_blob

def test_deserialize_bitmap_blob_str_and_bytes_roundtrip():
    bm = Roaring([1, 3, 5, 7])
    blob_bytes = bm.serialize()
    blob_str = blob_bytes.decode("latin-1")
    r1 = deserialize_bitmap_blob(blob_bytes)
    r2 = deserialize_bitmap_blob(blob_str)
    assert list(r1) == [1, 3, 5, 7]
    assert list(r2) == [1, 3, 5, 7]
