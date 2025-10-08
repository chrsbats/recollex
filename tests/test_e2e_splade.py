import pytest
from pathlib import Path
from recollex.engine import Recollex

pytestmark = pytest.mark.real_splade

def test_e2e_add_search_real_splade(tmp_path: Path, splade_enc):
    idx = Recollex.open(tmp_path)
    # Reuse preloaded encoder so model loads only once
    idx._encoder = splade_enc

    ids = idx.add([
        ("hello world", ["tag1"], 1),
        ("world test", ["tag2"], 2),
        ("another doc about hello", ["tag1"], 3),
    ])
    assert len(ids) == 3

    res = idx.search("hello", k=5)
    assert len(res) > 0
    texts = [r.get("text") or "" for r in res]
    assert any("hello" in t for t in texts)

def test_e2e_batch_search_real_splade(tmp_path: Path, splade_enc):
    idx = Recollex.open(tmp_path)
    idx._encoder = splade_enc

    idx.add([
        ("a small cat", ["pet"], 10),
        ("a big dog", ["pet"], 11),
        ("quantum physics text", ["science"], 12),
    ])

    res_batch = idx.search(["cat", "quantum"], k=3)
    assert isinstance(res_batch, list) and len(res_batch) == 2
    assert any("cat" in (r.get("text") or "") for r in res_batch[0])
    assert any("quantum" in (r.get("text") or "") for r in res_batch[1])

def test_encoder_runs_on_cpu(splade_enc):
    # Sanity check that encode works on CPU
    idxs, vals = splade_enc.encode("hello world")
    assert isinstance(idxs, list) and isinstance(vals, list)
    assert len(idxs) == len(vals) and len(idxs) > 0
