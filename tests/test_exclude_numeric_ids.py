from recollex.engine import Recollex

def test_exclude_numeric_doc_ids_removes_hit(index: Recollex, now):
    a = index.add("alpha", tags=["t"], timestamp=now())
    _ = index.add("alpha", tags=["t"], timestamp=now() + 1)

    hits = index.search("alpha", k=10, exclude_doc_ids=[str(a)])
    ids = [h["doc_id"] for h in hits]
    assert str(a) not in ids
