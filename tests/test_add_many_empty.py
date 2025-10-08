from recollex.engine import Recollex

def test_add_many_empty_returns_zero(tmp_path):
    rx = Recollex.open(tmp_path / "idx")
    out = rx.add_many([], dims=8)
    assert out == {"n_docs": 0, "nnz": 0}
