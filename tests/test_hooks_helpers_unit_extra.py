from pyroaring import BitMap as Roaring
from recollex.hooks import _or_bitmaps, _and_bitmaps, _first_n, _len_bitmap

def test_bitmap_helpers_empty_and_first_n_zero():
    empty_or = _or_bitmaps([])
    empty_and = _and_bitmaps([])
    assert _len_bitmap(empty_or) == 0
    assert _len_bitmap(empty_and) == 0

    bm = Roaring([1, 2, 3])
    assert _len_bitmap(_first_n(bm, 0)) == 0
    assert list(_first_n(bm, 2)) == [1, 2]
