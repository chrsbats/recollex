from typing import Set

from recollex.engine import Recollex


def test_tag_filters_all_one_none(index: Recollex, now):
    d1 = index.add("a", tags=["tenant:acme", "topic:db"], timestamp=now())
    d2 = index.add("b", tags=["tenant:acme", "topic:ml"], timestamp=now())
    d3 = index.add("c", tags=["tenant:beta", "topic:db"], timestamp=now())

    # all_of => intersection
    res_all = index.search("", all_of_tags=["tenant:acme", "topic:db"], k=10)
    assert {h["doc_id"] for h in res_all} == {str(d1)}

    # one_of => union (base becomes union if no other filters)
    res_one = index.search("", one_of_tags=["tenant:beta", "topic:ml"], k=10)
    assert {h["doc_id"] for h in res_one} == {str(d2), str(d3)}

    # none_of => exclude from base; with empty text, base -> union(all previous tag filters)
    res_none = index.search("", none_of_tags=["topic:ml"], k=10)
    ids_none: Set[str] = {h["doc_id"] for h in res_none}
    assert str(d2) not in ids_none
    assert {str(d1), str(d3)}.issubset(ids_none)


def test_exclude_doc_ids(index: Recollex, now):
    d1 = index.add("q", tags=["t"], timestamp=now())
    d2 = index.add("q", tags=["t"], timestamp=now())
    d3 = index.add("q", tags=["t"], timestamp=now())

    res = index.search("q", k=10, exclude_doc_ids=[str(d2)])
    ids = [h["doc_id"] for h in res]
    assert str(d2) not in ids
    assert str(d1) in ids and str(d3) in ids


def test_empty_query_with_tag_scope_in_score_profile(index: Recollex, now):
    a = index.add("x", tags=["tenant:acme"], timestamp=now())
    b = index.add("y", tags=["tenant:acme"], timestamp=now() + 1)

    # With tag scope: returns docs, scores are 0.0 (no terms)
    scoped = index.search("", profile="rag", all_of_tags=["tenant:acme"], k=5)
    assert {h["doc_id"] for h in scoped} == {str(a), str(b)}
    assert all(h["score"] == 0.0 for h in scoped)

    # Without any scope and empty query: returns []
    empty = index.search("", profile="rag", k=5)
    assert empty == []
