from pathlib import Path

import pytest

from recollex.engine import Recollex


def test_add_many_indices_data_length_mismatch_raises(tmp_path: Path):
    rx = Recollex.open(tmp_path / "idx")
    docs = [
        {"doc_id": 1, "indices": [1, 2], "data": [1.0], "text": "x", "tags": None, "seq": 1},
    ]
    with pytest.raises(ValueError):
        rx.add_many(docs, dims=8)
