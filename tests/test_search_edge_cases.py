import pytest
from recollex.engine import Recollex


def test_exclude_non_numeric_doc_ids_is_noop(index: Recollex, now):
    a = index.add("alpha", tags=["t"], timestamp=now())
    _ = index.add("beta", tags=["t"], timestamp=now() + 1)

    # Excluding a non-numeric doc_id should be ignored (no crash, no exclusion)
    hits = index.search("alpha", k=5, exclude_doc_ids=["not-a-number"])
    ids = [h["doc_id"] for h in hits]
    assert str(a) in ids


def test_q_terms_with_tid_out_of_manifest_dims_raises(index: Recollex, now):
    # Establish manifest dims via first add (FakeEncoder dims == 8)
    index.add("seed", tags=["t"], timestamp=now())
    # Craft a query with a term id outside manifest dims
    with pytest.raises(ValueError):
        _ = index.search_terms(q_terms=[(9999, 1.0)], k=5, profile="rag")


def test_q_to_csr_raises_when_term_id_exceeds_target_dims(index: Recollex):
    # target_dims=8 (FakeEncoder dims), but term id 10 >= 8
    with pytest.raises(ValueError):
        index._q_to_csr([(10, 1.0)], target_dims=8)


def test_recent_profile_with_scope_and_empty_query_returns_docs(index: Recollex, now):
    a = index.add("x", tags=["scoped"], timestamp=now())
    b = index.add("y", tags=["scoped"], timestamp=now() + 1)
    out = index.search("", profile="recent", all_of_tags=["scoped"], k=5)
    assert [h["doc_id"] for h in out] == [str(b), str(a)]  # ordered by seq desc


def test_recent_profile_with_min_score_gates_results(index: Recollex, now):
    # Two docs with same text so they share terms; seq differs
    d1 = index.add("alpha beta", tags=["t"], timestamp=now())
    d2 = index.add("alpha alpha alpha", tags=["t"], timestamp=now() + 1)

    q_terms = index._q_terms_from_text("alpha")

    # recent + min_score should execute the gating path; we just assert it returns something
    hits = index.search_terms(
        q_terms=q_terms,
        k=10,
        profile="recent",
        min_score=0.0,  # low threshold to avoid filtering everything
    )
    ids = {h["doc_id"] for h in hits}
    assert str(d1) in ids or str(d2) in ids
