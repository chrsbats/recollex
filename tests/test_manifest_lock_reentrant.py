import os
from pathlib import Path
from recollex.engine import Recollex

def test_manifest_lock_acquires_posix(tmp_path: Path, monkeypatch):
    # Ensure we do not force the sidecar lock so fcntl path is used on POSIX
    monkeypatch.delenv("RECOLLEX_FORCE_PID_LOCK", raising=False)
    rx = Recollex.open(tmp_path / "idx")
    # Acquire and release; should not raise
    with rx._manifest_lock(timeout=2.0):
        pass
    # Acquire again to ensure it’s re-usable
    with rx._manifest_lock(timeout=2.0):
        pass
