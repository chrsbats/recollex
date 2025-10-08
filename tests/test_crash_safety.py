import json
from pathlib import Path

import numpy as np

from recollex.engine import Recollex
from recollex.io.segments import write_segments


def test_crash_safety_manifest_tmp_ignored(tmp_path: Path, now):
    p = tmp_path / "idx"
    rx = Recollex.open(p)
    did = rx.add("hello", tags=["tenant:acme"], timestamp=now())

    # manifest.json exists and is valid
    man_path = p / "manifest.json"
    assert man_path.exists()
    with man_path.open("r", encoding="utf-8") as f:
        man = json.load(f)
    assert int(man.get("dims", 0)) > 0

    # Simulate crash: write bogus manifest.tmp (engine writes tmp via with_suffix('.tmp'))
    tmp_path_file = man_path.with_suffix(".tmp")
    tmp_path_file.write_text("{not-json", encoding="utf-8")

    # Reopen; should ignore manifest.tmp and load manifest.json
    rx2 = Recollex.open(p)
    hits = rx2.search("hello", k=5)
    assert any(h["doc_id"] == str(did) for h in hits)

    # dims still valid (FakeEncoder dims=8 after first add)
    assert int(rx2._manifest.get("dims", 0)) == 8


def test_crash_safety_orphan_segment_ignored(tmp_path: Path, now):
    p = tmp_path / "idx2"
    rx = Recollex.open(p)
    _ = rx.add("a", tags=["t"], timestamp=now())

    orig_len = len(rx._manifest.get("segments", []))

    # Manually write an orphan segment not referenced in manifest
    seg_path = p / "segments" / "seg_999"
    indptr = np.asarray([0, 1], dtype=np.int64)
    indices = np.asarray([0], dtype=np.int32)
    data = np.asarray([1.0], dtype=np.float32)
    row_ids = ["999"]
    _ = write_segments(seg_path, indptr, indices, data, row_ids)

    # Reopen and add another doc; manifest should grow by exactly 1
    rx2 = Recollex.open(p)
    did2 = rx2.add("b", tags=["t"], timestamp=now())

    new_len = len(rx2._manifest.get("segments", []))
    assert new_len == orig_len + 1

    # Orphan seg_999 is not referenced in manifest
    seg_names = [s["name"] for s in rx2._manifest.get("segments", [])]
    assert "seg_999" not in seg_names

    # Search still works and finds the new doc
    hits = rx2.search("b", k=5)
    assert any(h["doc_id"] == str(did2) for h in hits)
