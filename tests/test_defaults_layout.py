import json
from pathlib import Path

from recollex.engine import Recollex


def test_open_creates_layout(tmp_path: Path, now):
    p = tmp_path / "idx"
    rx = Recollex.open(p)

    # Smart defaults: base dir + segments/ + manifest.json exist on first use
    assert p.exists() and p.is_dir()
    assert (p / "segments").exists() and (p / "segments").is_dir()
    assert (p / "manifest.json").exists() and (p / "manifest.json").is_file()

    # After first add, SQLite DB exists
    rx.add("hello", tags=["tenant:acme"], timestamp=now())
    assert (p / "meta.sqlite").exists()


def test_reopen_existing(tmp_path: Path, now):
    p = tmp_path / "idx"
    rx1 = Recollex.open(p)
    did = rx1.add("hello world", tags=["tenant:acme"], timestamp=now())

    # Reopen existing index
    rx2 = Recollex.open(p)
    hits = rx2.search("hello world", k=5)
    assert isinstance(hits, list) and len(hits) >= 1
    # The added doc is present
    doc_ids = [h["doc_id"] for h in hits]
    assert str(did) in doc_ids


def test_manifest_initialization_and_dims(tmp_path: Path, index, now):
    p = tmp_path / "idx"

    # Before first add: manifest exists with version=1, dims=0
    man_path = p / "manifest.json"
    assert man_path.exists()
    with man_path.open("r", encoding="utf-8") as f:
        man = json.load(f)
    assert man.get("version") == 1
    assert int(man.get("dims", 0)) == 0

    # After first add: dims == FakeEncoder.dims (8 from conftest FakeEncoder)
    index.add("first", tags=["tenant:acme"], timestamp=now())
    with man_path.open("r", encoding="utf-8") as f:
        man2 = json.load(f)
    assert int(man2.get("dims", 0)) == 8
