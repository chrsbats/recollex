from __future__ import annotations

from typing import Union

from pyroaring import BitMap as Roaring

# Common bitmap names/prefixes
TERM_PREFIX = "term:"
TAG_PREFIX = "tag:"
LIVE_DOCS = "live_docs"
TOMBSTONES = "tombstones"

# Serialized Roaring bitmap form (e.g., TEXT latin-1 safe or bytes)
BitmapBlob = Union[str, bytes]


def deserialize_bitmap_blob(blob: BitmapBlob) -> Roaring:
    """
    Deserialize a stored BitmapBlob (str as latin-1 or raw bytes) into a Roaring bitmap.
    """
    if isinstance(blob, str):
        return Roaring.deserialize(blob.encode("latin-1"))
    return Roaring.deserialize(blob)
