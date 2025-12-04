import pytest
from pathlib import Path
from recollex import Recollex


def _seed_simple_corpus(rx: Recollex):
    # Create 4 docs across two terms: 10 and 20; seq increasing
    docs = [
        {"doc_id": 1, "indices": [10], "data": [1.0], "text": "A", "tags": None, "seq": 100},
        {"doc_id": 2, "indices": [10], "data": [0.1], "text": "B", "tags": None, "seq": 101},
        {"doc_id": 3, "indices": [20], "data": [5.0], "text": "C", "tags": None, "seq": 102},
        {"doc_id": 4, "indices": [10], "data": [0.6], "text": "D", "tags": None, "seq": 103},
    ]
    rx.add_many(docs, dims=32)


def test_min_score_filters_in_score_profile(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx")
    _seed_simple_corpus(rx)
    # Query on term 10 with unit weight -> scores equal row values on term 10
    res = rx.search_terms(q_terms=[(10, 1.0)], k=10, profile="rag", min_score=0.5)
    got = [int(r["doc_id"]) for r in res]
    # Expect doc1 (1.0) and doc4 (0.6); score ordering desc
    assert got == [1, 4]
    # Threshold above max score -> empty
    res2 = rx.search_terms(q_terms=[(10, 1.0)], k=10, profile="rag", min_score=1.1)
    assert res2 == []


def test_min_score_recent_orders_by_seq_and_gates_scores(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx2")
    _seed_simple_corpus(rx)
    # With recent profile and non-empty query, min_score gates but ordering is by seq
    res = rx.search_terms(q_terms=[(10, 1.0)], k=10, profile="recent", min_score=0.5)
    got = [int(r["doc_id"]) for r in res]
    # Docs {1 (seq=100, score=1.0), 4 (seq=103, score=0.6)} pass; order by seq desc => [4, 1]
    assert got == [4, 1]


def test_min_score_ignored_for_recent_empty_query(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx3")
    _seed_simple_corpus(rx)
    # Empty query under 'recent': min_score is ignored
    res_no_min = rx.search_terms(q_terms=[], k=3, profile="recent", min_score=None)
    res_min = rx.search_terms(q_terms=[], k=3, profile="recent", min_score=0.9)
    got_no_min = [int(r["doc_id"]) for r in res_no_min]
    got_min = [int(r["doc_id"]) for r in res_min]
    # Top-3 by seq: [4,3,2]
    assert got_no_min == [4, 3, 2]
    assert got_min == got_no_min


def test_min_score_recent_applies_dims_guard(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx4")
    _seed_simple_corpus(rx)  # sets manifest dims=32
    # Using min_score with recent must still validate q_terms against manifest dims
    with pytest.raises(ValueError):
        rx.search_terms(q_terms=[(1000, 1.0)], k=5, profile="recent", min_score=0.1)
