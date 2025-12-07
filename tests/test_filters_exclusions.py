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


def test_tags_views_dict_and_list_and_none(index: Recollex, now):
    # Dict tags
    d1 = index.add("a", tags={"project": "p1", "doc_key": "d1"}, timestamp=now())
    # List tags
    d2 = index.add("b", tags=["project:p2", "foo"], timestamp=now())
    # No tags
    d3 = index.add("c", tags=None, timestamp=now())

    # Dict tags doc
    hits1 = index.search("a", k=5)
    h1 = next(h for h in hits1 if h["doc_id"] == str(d1))
    assert isinstance(h1["tags"], list)
    assert isinstance(h1["tags_list"], list)
    assert isinstance(h1["tags_dict"], dict)
    assert set(h1["tags"]) == set(h1["tags_list"])
    # Order of dict items is not guaranteed; just check membership
    assert "project:p1" in h1["tags_list"]
    assert "doc_key:d1" in h1["tags_list"]
    assert h1["tags_dict"]["project"] == "p1"
    assert h1["tags_dict"]["doc_key"] == "d1"

    # List tags doc
    hits2 = index.search("b", k=5)
    h2 = next(h for h in hits2 if h["doc_id"] == str(d2))
    assert isinstance(h2["tags"], list)
    assert isinstance(h2["tags_list"], list)
    assert isinstance(h2["tags_dict"], dict)
    assert set(h2["tags"]) == set(h2["tags_list"])
    assert set(h2["tags_list"]) == {"project:p2", "foo"}
    assert h2["tags_dict"]["project"] == "p2"
    # "foo" has no "k:v" form, so it should not appear in tags_dict
    assert "foo" not in h2["tags_dict"]

    # No tags doc
    hits3 = index.search("c", k=5)
    h3 = next(h for h in hits3 if h["doc_id"] == str(d3))
    assert h3["tags"] == []
    assert h3["tags_list"] == []
    assert h3["tags_dict"] == {}


def test_tags_dict_last_wins_for_duplicate_keys(index: Recollex, now):
    d = index.add("dup", tags=["k:v1", "k:v2"], timestamp=now())
    hits = index.search("dup", k=5)
    h = next(h for h in hits if h["doc_id"] == str(d))
    assert h["tags_dict"]["k"] == "v2"


def test_project_scopes_search_and_remove(index: Recollex, now):
    # Dict tags with project for both docs so project=... scopes via filters
    d1 = index.add("a", tags={"project": "p1", "topic": "db"}, timestamp=now())
    d2 = index.add("a", tags={"project": "p2", "topic": "db"}, timestamp=now() + 1)

    # search with project="p1" should only see d1
    hits_p1 = index.search("a", project="p1", k=10)
    ids_p1 = {h["doc_id"] for h in hits_p1}
    assert str(d1) in ids_p1
    assert str(d2) not in ids_p1

    # search with project="p2" should only see d2
    hits_p2 = index.search("a", project="p2", k=10)
    ids_p2 = {h["doc_id"] for h in hits_p2}
    assert str(d2) in ids_p2
    assert str(d1) not in ids_p2

    # remove_by with project="p1" removes only d1
    n = index.remove_by(project="p1")
    assert n == 1
    remaining_ids = {h["doc_id"] for h in index.search("a", k=10)}
    assert str(d1) not in remaining_ids
    assert str(d2) in remaining_ids


def test_project_does_not_override_explicit_filter(index: Recollex, now):
    d1 = index.add("x", tags={"project": "p1"}, timestamp=now())
    d2 = index.add("x", tags={"project": "p2"}, timestamp=now() + 1)

    # Baseline: without project or filters, both docs are visible
    hits = index.search("x", k=10)
    assert {h["doc_id"] for h in hits} == {str(d1), str(d2)}

    # filters["project"] = "p2" should win over project="p1"
    hits_filtered = index.search_terms(
        q_terms=index._q_terms_from_text("x"),
        k=10,
        filters={"project": "p2"},
        project="p1",
    )
    ids = {h["doc_id"] for h in hits_filtered}
    assert str(d2) in ids
    assert str(d1) not in ids


def test_last_with_project_scopes_results(index: Recollex, now):
    d1 = index.add("x", tags={"project": "p1"}, timestamp=now())
    d2 = index.add("x", tags={"project": "p2"}, timestamp=now() + 1)

    hits_p1 = index.last(project="p1", k=10)
    ids_p1 = {h["doc_id"] for h in hits_p1}
    assert str(d1) in ids_p1
    assert str(d2) not in ids_p1
