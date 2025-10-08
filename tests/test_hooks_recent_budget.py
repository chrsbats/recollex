from pyroaring import BitMap as Roaring
from recollex.hooks import candidate_supplier_recent


def test_candidate_supplier_recent_budget_caps_to_first_n():
    base = Roaring(list(range(10)))
    out = candidate_supplier_recent(base, budget=3)
    assert len(out) == 3
    assert list(out) == [0, 1, 2]
