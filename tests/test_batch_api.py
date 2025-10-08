from typing import List

from recollex.engine import Recollex


def test_add_single_and_batch_variants(index: Recollex, now):
    # Single add returns int; tags persisted
    did_single = index.add("alpha", tags=["t:a"], timestamp=now())
    assert isinstance(did_single, int)
    hits = index.search("alpha", k=5)
    assert any(h["doc_id"] == str(did_single) for h in hits)
    hit = next(h for h in hits if h["doc_id"] == str(did_single))
    assert "t:a" in (hit["tags"] or [])

    # Batch add with list of tuples returns list[int]; tags persisted
    ids_tuple: List[int] = index.add([
        ("beta", ["t:b"], now()),
        ("gamma", ["t:c"], now() + 1),
    ])
    assert isinstance(ids_tuple, list) and len(ids_tuple) == 2 and all(isinstance(x, int) for x in ids_tuple)

    hits_beta = index.search("beta", k=5)
    assert any(h["doc_id"] == str(ids_tuple[0]) for h in hits_beta)
    hit_beta = next(h for h in hits_beta if h["doc_id"] == str(ids_tuple[0]))
    assert "t:b" in (hit_beta["tags"] or [])

    # Batch add with list of dicts works
    ids_dict: List[int] = index.add([
        {"text": "delta", "tags": ["t:d"], "timestamp": now()},
        {"text": "epsilon", "tags": ["t:e"], "timestamp": now() + 1},
    ])
    assert isinstance(ids_dict, list) and len(ids_dict) == 2 and all(isinstance(x, int) for x in ids_dict)

    hits_delta = index.search("delta", k=5)
    assert any(h["doc_id"] == str(ids_dict[0]) for h in hits_delta)
    hit_delta = next(h for h in hits_delta if h["doc_id"] == str(ids_dict[0]))
    assert "t:d" in (hit_delta["tags"] or [])


def test_add_tuple_is_not_batch(index: Recollex, now):
    # "list means batch" only; tuple should be treated as single (no error), returning int
    did = index.add(("zeta", ["t:z"], now()))
    assert isinstance(did, int)


def test_search_batch_and_empty_batch(index: Recollex, now):
    index.add([
        ("q1 text", ["t:x"], now()),
        ("q2 text", ["t:y"], now() + 1),
    ])
    res = index.search(["q1 text", "q2 text"], k=3)
    assert isinstance(res, list) and len(res) == 2 and all(isinstance(r, list) for r in res)

    # Empty batch returns empty list
    assert index.search([]) == []


def test_remove_single_and_batch(index: Recollex, now):
    d1, d2, d3 = index.add([
        ("r1", ["t:r"], now()),
        ("r2", ["t:r"], now() + 1),
        ("r3", ["t:r"], now() + 2),
    ])

    # Single remove hides it
    index.remove(d1)
    hits1 = index.search("r1", k=5)
    assert all(h["doc_id"] != str(d1) for h in hits1)

    # Batch remove hides both; empty list is no-op
    index.remove([d2, d3])
    hits2 = index.search("r2", k=5)
    hits3 = index.search("r3", k=5)
    assert all(h["doc_id"] != str(d2) for h in hits2)
    assert all(h["doc_id"] != str(d3) for h in hits3)

    index.remove([])  # should not error
