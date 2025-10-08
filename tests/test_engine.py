from pathlib import Path

from recollex.engine import Recollex


def test_add_many_and_search_terms_and_recency(tmp_path: Path):
    idx = tmp_path / "index"
    rx = Recollex.open(idx)

    docs = [
        {
            "doc_id": 1,
            "indices": [1],
            "data": [1.0],
            "text": "doc1",
            "tags": {"lang": "en"},
            "seq": 10,
        },
        {
            "doc_id": 2,
            "indices": [1, 2],
            "data": [0.5, 1.2],
            "text": "doc2",
            "tags": ["python"],
            "seq": 20,
        },
        {
            "doc_id": 3,
            "indices": [2],
            "data": [0.2],
            "text": "doc3",
            "tags": {"lang": "fr"},
            "seq": 30,
        },
    ]
    res = rx.add_many(docs, dims=8)
    assert res["n_docs"] == 3
    assert res["nnz"] == 4

    # Single-term query: MUST=[1], SHOULD=[]
    q1 = [(1, 1.0)]
    out1 = rx.search_terms(q1, k=5, profile="rag")
    assert [o["doc_id"] for o in out1] == ["1", "2"]
    assert out1[0]["score"] > out1[1]["score"]  # doc1(1.0) > doc2(0.5)

    # Two-term query: policy yields MUST=[1], SHOULD=[2] -> only doc2 survives C = base ∩ AND(MUST) ∩ OR(SHOULD)
    q2 = [(1, 1.0), (2, 1.0)]
    out2 = rx.search_terms(q2, k=5, profile="rag")
    assert [o["doc_id"] for o in out2] == ["2"]

    # Tag filters with single-term query
    out_lang_en = rx.search_terms(q1, k=5, profile="rag", tags_all_of=["lang=en"])
    assert [o["doc_id"] for o in out_lang_en] == ["1"]  # only doc1 has lang=en

    out_python = rx.search_terms(q1, k=5, profile="rag", tags_one_of=["python"])
    assert [o["doc_id"] for o in out_python] == ["2"]

    out_not_fr = rx.search_terms(q1, k=5, profile="rag", tags_none_of=["lang=fr"])
    assert [o["doc_id"] for o in out_not_fr] == ["1", "2"]

    # Exclude specific doc ids
    out_excl = rx.search_terms(q1, k=5, profile="rag", exclude_doc_ids=["2"])
    assert [o["doc_id"] for o in out_excl] == ["1"]

    # Old filters dict (k=v) with single-term query
    out_old = rx.search_terms(q1, k=5, profile="rag", filters={"lang": "en"})
    assert [o["doc_id"] for o in out_old] == ["1"]

    # Recency-first: last() returns by seq desc
    last = rx.last(k=2)
    assert [o["doc_id"] for o in last] == ["3", "2"]
