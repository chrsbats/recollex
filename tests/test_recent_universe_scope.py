import pytest
from recollex.engine import Recollex

def test_recent_universe_scope_uses_store_and_budget(tmp_path, now):
    idx = Recollex.open(tmp_path / "idx")
    # Add several docs with increasing seq
    ids = idx.add([
        ("a", ["t"], now()),
        ("b", ["t"], now() + 1),
        ("c", ["t"], now() + 2),
        ("d", ["t"], now() + 3),
    ])
    # Universe scope: no tag filters, no text, so engine should use store.iter_recent_doc_ids
    out = idx.search("", profile="recent", k=10, exclude_doc_ids=None, override_knobs={"budget": 2})
    # Budget trims to 2 most recent
    assert len(out) == 2
    # Ordered by seq desc (most recent added last)
    assert [h["doc_id"] for h in out] == [str(ids[-1]), str(ids[-2])]

    # Excluding the most recent doc id reflows to next ones
    out2 = idx.search("", profile="recent", k=10, exclude_doc_ids=[str(ids[-1])], override_knobs={"budget": 2})
    assert len(out2) == 2
    assert str(ids[-1]) not in [h["doc_id"] for h in out2]
