import pytest
from recollex.engine import Recollex

def test_all_docs_bitmap_handles_store_list_exception(tmp_path, monkeypatch):
    rx = Recollex.open(tmp_path / "idx")
    class BoomStore:
        def get_bitmap(self, name): return None
        def list_bitmaps(self, prefix=None): raise RuntimeError("boom")
    # Patch only the methods used by _all_docs_bitmap
    monkeypatch.setattr(rx, "_store", BoomStore(), raising=False)
    bm = rx._all_docs_bitmap()
    # With no bitmaps and exception, should return empty bitmap
    assert len(bm) == 0
