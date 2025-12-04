import time
from pathlib import Path
from recollex import Recollex

def test_add_single_without_timestamp_assigns_seq(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx")
    did = rx.add("no ts", tags=["tenant:acme"])
    res = rx.search("", profile="recent", k=1)
    assert res, "recent should return the doc"
    assert res[0]["doc_id"] == str(did)
    assert isinstance(res[0]["seq"], int)
    assert res[0]["seq"] >= 1

def test_batch_add_dicts_without_timestamp_assigns_seq(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx")
    ids = rx.add([
        {"text": "a", "tags": []},
        {"text": "b", "tags": []},
    ])
    res = rx.search("", profile="recent", k=10)
    got = {r["doc_id"]: r["seq"] for r in res}
    assert all(str(d) in got for d in ids)
    seqs = [got[str(d)] for d in ids]
    assert all(isinstance(s, int) for s in seqs)
    assert seqs[0] != seqs[1], "seq should be distinct and monotonic"

def test_batch_add_mixed_seq_and_missing_timestamp(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx")
    base = 100
    ids = rx.add([
        {"text": "with-seq", "tags": [], "seq": base},
        {"text": "no-ts", "tags": []},
    ])
    res = rx.search("", profile="recent", k=10)
    got = {r["doc_id"]: r["seq"] for r in res}
    assert set(got[str(i)] for i in ids) == {base, base + 1}, "explicit seq should be respected; next_seq should advance"
