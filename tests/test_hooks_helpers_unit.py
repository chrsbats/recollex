import pytest
from pyroaring import BitMap as Roaring
from scipy.sparse import csr_matrix

from recollex.hooks import _or_bitmaps, _and_bitmaps, _first_n, _len_bitmap, score_csr_slice


def test_bitmap_helpers_or_and_first_n_len():
    b1 = Roaring([1, 3])
    b2 = Roaring([3, 4])
    assert list(_or_bitmaps([b1, b2])) == [1, 3, 4]
    assert list(_and_bitmaps([b1, b2])) == [3]
    assert list(_first_n(Roaring([10, 11, 12]), 2)) == [10, 11]
    assert _len_bitmap(Roaring([5, 6])) == 2


def test_score_csr_slice_dims_mismatch_raises():
    X = csr_matrix(([1.0], ([0], [0])), shape=(1, 5))
    q = csr_matrix(([1.0], ([0], [0])), shape=(1, 3))
    with pytest.raises(ValueError):
        score_csr_slice(q, {"csr": X}, [0])
