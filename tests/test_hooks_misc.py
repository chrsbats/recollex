import pytest
from scipy.sparse import csr_matrix
from recollex.hooks import score_hook_noop, make_profile

def test_score_hook_noop_returns_zeros():
    q = csr_matrix(([1.0], ([0], [0])), shape=(1, 3))
    scores = score_hook_noop(q, {}, [0, 2])
    assert scores == [(0, 0.0), (2, 0.0)]

def test_make_profile_unknown_profile_raises():
    with pytest.raises(Exception):
        make_profile("unknown-profile")
