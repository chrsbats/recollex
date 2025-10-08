def test_smoke_add_search_single(index, now):
    # Add two docs with increasing seq; query equals first doc text to ensure a hit
    index.add("q", tags=["tenant:acme"], timestamp=now())
    index.add("qq", tags=["tenant:acme"], timestamp=now() + 1)

    res = index.search("q", k=5)
    assert isinstance(res, list) and len(res) >= 1

    must_keys = {"doc_id", "score", "segment_id", "row_offset", "seq", "text", "tags"}
    for hit in res:
        assert must_keys.issubset(hit.keys())

    # Scores non-increasing
    scores = [h["score"] for h in res]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_smoke_batch_add_and_batch_search(index, now):
    ids = index.add([
        ("a", ["tenant:acme"], now()),
        ("b", ["tenant:acme"], now() + 1),
    ])
    assert isinstance(ids, list) and len(ids) == 2 and all(isinstance(x, int) for x in ids)

    res = index.search(["a", "b"], k=3)
    assert isinstance(res, list) and len(res) == 2
    assert all(isinstance(r, list) for r in res)


def test_smoke_recent_profile(index, now):
    N = 5
    for i in range(N):
        index.add(f"t{i}", tags=["tenant:acme"], timestamp=now())

    res = index.search("", profile="recent", k=N)
    assert isinstance(res, list)
    # Ensure ordered by seq desc
    seqs = [h["seq"] for h in res]
    assert seqs == sorted(seqs, reverse=True)
