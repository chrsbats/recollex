import numpy as np

from recollex.engine import Recollex


def test_csr_dot_matches_bruteforce(index: Recollex, now):
    # Add docs with identical text so they share the same indices/weights
    ids = index.add([
        ("q", ["tenant:acme"], now()),
        ("q", ["tenant:acme"], now() + 1),
        ("q", ["tenant:acme"], now() + 2),
        ("q", ["tenant:acme"], now() + 3),
    ])
    q_terms = index._q_terms_from_text("q")

    hits = index.search_terms(
        q_terms,
        k=10,
        profile="rag",
        tags_all_of=["tenant:acme"],
    )
    assert len(hits) >= 1

    # Manifest dims
    D = int(index._manifest.get("dims", 0) or 0)
    assert D > 0

    # For each hit: brute-force dot(q, row)
    for h in hits:
        seg = h["segment_id"]
        row = int(h["row_offset"])

        # Open reader via IO layer
        from recollex.io.segments import open_segment
        reader = open_segment(index._segments_dir / seg, dims=D)

        indptr = reader.indptr
        indices = reader.indices
        data = reader.data

        start = int(indptr[row])
        end = int(indptr[row + 1])
        cols = indices[start:end]
        vals = data[start:end]

        q = {int(t): float(w) for t, w in q_terms}
        brute = float(np.sum([q.get(int(c), 0.0) * float(v) for c, v in zip(cols.tolist(), vals.tolist())]))
        assert np.isclose(brute, float(h["score"]), rtol=1e-6, atol=1e-6)


def test_bitmap_gating_must_should_vs_rag(index: Recollex, now):
    # Construct postings: term 1 is "high DF" (present in many), term 2 is low DF
    docs = [
        {"doc_id": 1, "indices": [1], "data": [1.0], "text": "a", "tags": None, "seq": now()},
        {"doc_id": 2, "indices": [1], "data": [1.0], "text": "b", "tags": None, "seq": now() + 1},
        {"doc_id": 3, "indices": [1], "data": [1.0], "text": "c", "tags": None, "seq": now() + 2},
        {"doc_id": 4, "indices": [2], "data": [1.0], "text": "d", "tags": None, "seq": now() + 3},
        {"doc_id": 5, "indices": [1, 2], "data": [1.0, 1.0], "text": "e", "tags": None, "seq": now() + 4},
    ]
    index.add_many(docs, dims=8)

    q_terms = [(1, 1.0), (2, 1.0)]

    rag = index.search_terms(q_terms, k=10, profile="rag")
    hp = index.search_terms(q_terms, k=10, profile="paraphrase_hp")

    rag_ids = {h["doc_id"] for h in rag}
    hp_ids = {h["doc_id"] for h in hp}

    assert hp_ids.issubset(rag_ids)
    assert len(hp_ids) <= len(rag_ids)
