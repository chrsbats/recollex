from pathlib import Path
from recollex import Recollex


def test_remove_by_all_of_tags(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx")
    ids = rx.add([
        ("A", ["tenant:acme", "doc:a"], 1),
        ("B", ["tenant:acme", "doc:b"], 2),
        ("C", ["tenant:globex", "doc:c"], 3),
        ("D", ["tenant:acme", "doc:d"], 4),
    ])
    n = rx.remove_by(all_of_tags=["tenant:acme"])
    assert n == 3
    # Only the globex doc remains
    res = rx.search("", profile="recent", k=10)
    got = [int(r["doc_id"]) for r in res]
    assert set(got) == {ids[2]}


def test_remove_by_filters_and_dry_run(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx2")
    # Mix dict-style tags (filters=...) and list strings
    rx.add_many([
        {"doc_id": 101, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme", "doc": "a"}, "seq": 1},
        {"doc_id": 102, "indices": [], "data": [], "text": "y", "tags": {"tenant": "acme", "doc": "b"}, "seq": 2},
        {"doc_id": 103, "indices": [], "data": [], "text": "z", "tags": {"tenant": "acme", "doc": "c"}, "seq": 3},
        {"doc_id": 104, "indices": [], "data": [], "text": "w", "tags": {"tenant": "globex", "doc": "d"}, "seq": 4},
    ], dims=0)
    # Dry run should not delete
    cnt = rx.remove_by(all_of_tags=[("tenant", "acme")], dry_run=True)
    assert cnt == 3
    pre = rx.search("", profile="recent", k=10)
    assert len(pre) == 4
    # Real delete
    n = rx.remove_by(all_of_tags=[("tenant", "acme")])
    assert n == 3
    post = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in post] == [104]


def test_remove_by_one_of_tags(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_oneof")
    rx.add_many([
        {"doc_id": 1, "indices": [], "data": [], "text": "x", "tags": ["t:a"], "seq": 1},
        {"doc_id": 2, "indices": [], "data": [], "text": "y", "tags": ["t:b"], "seq": 2},
        {"doc_id": 3, "indices": [], "data": [], "text": "z", "tags": ["t:c"], "seq": 3},
    ], dims=0)
    n = rx.remove_by(one_of_tags=["t:a", "t:b"])
    assert n == 2
    res = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res] == [3]


def test_remove_by_none_of_tags(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_noneof")
    rx.add_many([
        {"doc_id": 10, "indices": [], "data": [], "text": "x", "tags": ["keep"], "seq": 1},
        {"doc_id": 11, "indices": [], "data": [], "text": "y", "tags": ["drop"], "seq": 2},
        {"doc_id": 12, "indices": [], "data": [], "text": "z", "tags": ["drop"], "seq": 3},
    ], dims=0)
    # Remove all docs not tagged 'keep'
    n = rx.remove_by(none_of_tags=["keep"])
    assert n == 2
    res = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res] == [10]


def test_remove_by_no_scope_is_noop(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_no_scope")
    rx.add_many([
        {"doc_id": 1, "indices": [], "data": [], "text": "x", "tags": ["t"], "seq": 1},
    ], dims=0)
    assert rx.remove_by() == 0
    res = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res] == [1]


def test_remove_by_all_of_tags_kv_tuple_and_dict(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_kv_allof")
    # Index with dict tags so we get tag:tenant=acme, tag:tenant=globex
    rx.add_many([
        {"doc_id": 1, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 2, "indices": [], "data": [], "text": "y", "tags": {"tenant": "acme"}, "seq": 2},
        {"doc_id": 3, "indices": [], "data": [], "text": "z", "tags": {"tenant": "globex"}, "seq": 3},
    ], dims=0)
    # Tuple form ("tenant","acme") should match tag:tenant=acme bitmap
    n1 = rx.remove_by(all_of_tags=[("tenant", "acme")])
    assert n1 == 2
    res1 = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res1] == [3]

    # Re-seed and use dict form {"tenant":"acme"}
    rx = Recollex.open(tmp_path / "idx_kv_allof2")
    rx.add_many([
        {"doc_id": 10, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 11, "indices": [], "data": [], "text": "y", "tags": {"tenant": "acme"}, "seq": 2},
        {"doc_id": 12, "indices": [], "data": [], "text": "z", "tags": {"tenant": "globex"}, "seq": 3},
    ], dims=0)
    n2 = rx.remove_by(all_of_tags=[{"tenant": "acme"}])
    assert n2 == 2
    res2 = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res2] == [12]


def test_remove_by_all_of_tags_single_dict(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_kv_allof_single_dict")
    rx.add_many([
        {"doc_id": 10, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 11, "indices": [], "data": [], "text": "y", "tags": {"tenant": "acme"}, "seq": 2},
        {"doc_id": 12, "indices": [], "data": [], "text": "z", "tags": {"tenant": "globex"}, "seq": 3},
    ], dims=0)
    n = rx.remove_by(all_of_tags={"tenant": "acme"})
    assert n == 2
    res = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res] == [12]


def test_remove_by_one_of_tags_kv_tuple_and_dict(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_kv_oneof")
    rx.add_many([
        {"doc_id": 1, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 2, "indices": [], "data": [], "text": "y", "tags": {"tenant": "globex"}, "seq": 2},
        {"doc_id": 3, "indices": [], "data": [], "text": "z", "tags": {"tenant": "initech"}, "seq": 3},
    ], dims=0)
    # Tuple form: remove tenant in {acme, globex}
    n1 = rx.remove_by(one_of_tags=[("tenant", "acme"), ("tenant", "globex")])
    assert n1 == 2
    res1 = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res1] == [3]

    # Re-seed and use dict form
    rx = Recollex.open(tmp_path / "idx_kv_oneof2")
    rx.add_many([
        {"doc_id": 10, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 11, "indices": [], "data": [], "text": "y", "tags": {"tenant": "globex"}, "seq": 2},
        {"doc_id": 12, "indices": [], "data": [], "text": "z", "tags": {"tenant": "initech"}, "seq": 3},
    ], dims=0)
    n2 = rx.remove_by(one_of_tags=[{"tenant": "acme"}, {"tenant": "globex"}])
    assert n2 == 2
    res2 = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res2] == [12]


def test_remove_by_one_of_tags_single_dict(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_kv_oneof_single_dict")
    rx.add_many([
        {"doc_id": 10, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 11, "indices": [], "data": [], "text": "y", "tags": {"tenant": "globex"}, "seq": 2},
        {"doc_id": 12, "indices": [], "data": [], "text": "z", "tags": {"tenant": "initech"}, "seq": 3},
    ], dims=0)
    n = rx.remove_by(one_of_tags={"tenant": "acme"})
    assert n == 1
    res = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res] == [12, 11]


def test_remove_by_none_of_tags_kv_tuple_and_dict(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_kv_noneof")
    rx.add_many([
        {"doc_id": 1, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 2, "indices": [], "data": [], "text": "y", "tags": {"tenant": "globex"}, "seq": 2},
        {"doc_id": 3, "indices": [], "data": [], "text": "z", "tags": {"tenant": "initech"}, "seq": 3},
    ], dims=0)
    # none_of: remove all docs whose tenant is NOT acme (i.e. keep only acme)
    n1 = rx.remove_by(none_of_tags=[("tenant", "acme")])
    assert n1 == 2
    res1 = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res1] == [1]

    # Re-seed and use dict form
    rx = Recollex.open(tmp_path / "idx_kv_noneof2")
    rx.add_many([
        {"doc_id": 10, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 11, "indices": [], "data": [], "text": "y", "tags": {"tenant": "globex"}, "seq": 2},
        {"doc_id": 12, "indices": [], "data": [], "text": "z", "tags": {"tenant": "initech"}, "seq": 3},
    ], dims=0)
    n2 = rx.remove_by(none_of_tags=[{"tenant": "acme"}])
    assert n2 == 2
    res2 = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res2] == [10]


def test_remove_by_none_of_tags_single_dict(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx_kv_noneof_single_dict")
    rx.add_many([
        {"doc_id": 10, "indices": [], "data": [], "text": "x", "tags": {"tenant": "acme"}, "seq": 1},
        {"doc_id": 11, "indices": [], "data": [], "text": "y", "tags": {"tenant": "globex"}, "seq": 2},
        {"doc_id": 12, "indices": [], "data": [], "text": "z", "tags": {"tenant": "initech"}, "seq": 3},
    ], dims=0)
    n = rx.remove_by(none_of_tags={"tenant": "acme"})
    assert n == 2
    res = rx.search("", profile="recent", k=10)
    assert [int(r["doc_id"]) for r in res] == [10]
