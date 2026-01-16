import pytest
from pathlib import Path
from recollex import Recollex

class FakeEncoder:
    def __init__(self):
        self.dims = 16
        self.encode_calls = 0

    def encode(self, text: str):
        self.encode_calls += 1
        # return two term ids so add_many writes something non-empty
        return ([1, 2], [0.5, 0.25])

    def encode_many(self, texts):
        self.encode_calls += len(texts)
        return [([1, 2], [0.5, 0.25]) for _ in texts]

def test_open_accepts_encoder_instance(tmp_path: Path):
    idx = tmp_path / "idx"
    enc = FakeEncoder()

    rx = Recollex.open(str(idx), encoder=enc)
    try:
        # injected instance used and accessible
        assert rx._encoder is enc

        # add one doc (should call encode)
        new_id = rx.add("hello world")
        assert isinstance(new_id, int)
        assert enc.encode_calls >= 1

        # search (single item) should call encode() again
        out = rx.search("hello world", k=10)
        assert isinstance(out, list)
        assert enc.encode_calls >= 2

        # open a second Recollex instance on same path reusing same encoder
        rx2 = Recollex.open(str(idx), encoder=enc)
        try:
            assert rx2._encoder is enc
            # search again via the second instance
            out2 = rx2.search("hello world", k=5)
            assert isinstance(out2, list)
            assert enc.encode_calls >= 3
        finally:
            rx2.close()
    finally:
        rx.close()
