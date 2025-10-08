import multiprocessing as mp
import os
from pathlib import Path
import time
import pytest
from recollex.engine import Recollex


def _writer(index_dir: str, docs):
    rx = Recollex.open(Path(index_dir))
    rx.add_many(docs, dims=8)


def test_concurrent_adds_produce_distinct_segments(tmp_path, monkeypatch):
    idx_dir = tmp_path / "idx"
    # Force engine to use sidecar PID lock to make contention deterministic across processes
    monkeypatch.setenv("RECOLLEX_FORCE_PID_LOCK", "1")
    rx = Recollex.open(idx_dir)

    docs1 = [{"doc_id": 1, "indices": [0], "data": [1.0], "text": "a", "tags": ["t"], "seq": 1}]
    docs2 = [{"doc_id": 2, "indices": [0], "data": [1.0], "text": "b", "tags": ["t"], "seq": 2}]

    ctx = mp.get_context("spawn")
    p1 = ctx.Process(target=_writer, args=(str(idx_dir), docs1))
    p2 = ctx.Process(target=_writer, args=(str(idx_dir), docs2))
    p1.start(); p2.start()
    p1.join(10); p2.join(10)
    assert p1.exitcode == 0 and p2.exitcode == 0

    rx2 = Recollex.open(idx_dir)
    segs = rx2._manifest.get("segments", [])
    assert len(segs) == 2
    names = [s["name"] for s in segs]
    assert len(set(names)) == 2  # no duplicate segment names

    q = [(0, 1.0)]
    hits = rx2.search_terms(q, k=10, profile="rag")
    got_ids = {h["doc_id"] for h in hits}
    assert {"1", "2"}.issubset(got_ids)


def _hold_lock(index_dir: str, hold_s: float):
    # Ensure child uses the deterministic sidecar PID lock
    os.environ["RECOLLEX_FORCE_PID_LOCK"] = "1"
    rx = Recollex.open(Path(index_dir))
    ready = Path(index_dir) / ".manifest_lock_ready"
    with rx._manifest_lock(timeout=5.0):
        # Signal to the parent that the lock is held
        try:
            ready.write_text("ready", encoding="utf-8")
        except Exception:
            pass
        time.sleep(hold_s)


def test_manifest_lock_timeout(tmp_path, monkeypatch):
    # Force engine to use sidecar PID lock in both parent and child
    monkeypatch.setenv("RECOLLEX_FORCE_PID_LOCK", "1")

    idx_dir = tmp_path / "idx_timeout"
    rx = Recollex.open(idx_dir)

    # Start a process that acquires and holds the manifest lock
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_hold_lock, args=(str(idx_dir), 5.0))
    p.start()

    # Wait until the child signals it holds the lock (up to 10s)
    ready = idx_dir / ".manifest_lock_ready"
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if ready.exists():
            break
        time.sleep(0.1)

    if not ready.exists():
        try:
            contents = [str(pth.name) for pth in idx_dir.iterdir()]
        except Exception as e:
            contents = [f"<list failed: {e}>"]
        pytest.fail(f"child did not signal readiness within 10s; idx_dir={idx_dir}, contents={contents}")

    # Attempt to acquire with a very short timeout; should raise
    with pytest.raises(TimeoutError):
        with rx._manifest_lock(timeout=0.3):
            pass

    p.join(10)
    assert p.exitcode == 0
