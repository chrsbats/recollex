from __future__ import annotations

import json
import sqlite3
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Iterator, List, Optional, Sequence, Union

from recollex.abcs import MetadataStore, DocRecord, BitmapBlob


class SQLiteMetadataStore(MetadataStore):
    """
    SQLite-backed MetadataStore.

    - Bitmaps are serialized Roaring blobs stored as TEXT (latin-1 safe).
    - Tiny in-memory LRU caches hot bitmap blobs (TEXT).
    - See docs/design.md for schema; created lazily if missing.
    """

    def __init__(self, db_path: Union[str, Path], bitmap_cache_size: int = 256) -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.row_factory = sqlite3.Row
        self._in_tx = False

        self._bm_cache_size = int(bitmap_cache_size)
        self._bm_cache: OrderedDict[str, str] = OrderedDict()

        self._ensure_schema()

    # ---------- Schema ----------
    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()

        # docs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS docs(
              doc_id TEXT PRIMARY KEY,
              segment_id TEXT NOT NULL,
              row_offset INTEGER NOT NULL,
              seq INTEGER NOT NULL,
              text TEXT,
              tags TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS docs_seg_off ON docs(segment_id, row_offset)")
        cur.execute("CREATE INDEX IF NOT EXISTS docs_seq ON docs(seq)")

        # bitmaps (TEXT storing latin-1 serialized blobs)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bitmaps(
              name TEXT PRIMARY KEY,
              data TEXT NOT NULL,
              last_used INTEGER
            )
            """
        )

        # stats
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stats(
              key TEXT PRIMARY KEY,
              value INTEGER
            )
            """
        )

        # kv
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kv(
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )

        self._conn.commit()

    # ---------- Transactions ----------
    @contextmanager
    def transaction(self) -> ContextManager[None]:
        if self._in_tx:
            # allow nested usage to be a no-op wrapper
            yield
            return
        self._in_tx = True
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._in_tx = False

    # ---------- Documents ----------
    def upsert_doc(self, doc: DocRecord) -> None:
        tags_text = None if doc.tags is None else json.dumps(doc.tags, separators=(",", ":"))
        self._conn.execute(
            """
            INSERT INTO docs(doc_id, segment_id, row_offset, seq, text, tags)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
              segment_id=excluded.segment_id,
              row_offset=excluded.row_offset,
              seq=excluded.seq,
              text=excluded.text,
              tags=excluded.tags
            """,
            (doc.doc_id, doc.segment_id, doc.row_offset, doc.seq, doc.text, tags_text),
        )
        if not self._in_tx:
            self._conn.commit()

    def get_doc(self, doc_id: str) -> Optional[DocRecord]:
        row = self._conn.execute(
            "SELECT doc_id, segment_id, row_offset, seq, text, tags FROM docs WHERE doc_id=?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        tags = json.loads(row["tags"]) if row["tags"] else None
        return DocRecord(
            doc_id=row["doc_id"],
            segment_id=row["segment_id"],
            row_offset=int(row["row_offset"]),
            seq=int(row["seq"]),
            text=row["text"],
            tags=tags,
        )

    def get_docs_many(self, doc_ids: Sequence[str]) -> List[DocRecord]:
        if not doc_ids:
            return []
        # Preserve input order while deduping
        ids = list(dict.fromkeys(str(x) for x in doc_ids))
        out_map = {}
        # Conservative chunk size to avoid exceeding max variable number
        CHUNK = 900
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            cur = self._conn.execute(
                f"SELECT doc_id, segment_id, row_offset, seq, text, tags FROM docs WHERE doc_id IN ({placeholders})",
                tuple(chunk),
            )
            for row in cur:
                tags = json.loads(row["tags"]) if row["tags"] else None
                out_map[row["doc_id"]] = DocRecord(
                    doc_id=row["doc_id"],
                    segment_id=row["segment_id"],
                    row_offset=int(row["row_offset"]),
                    seq=int(row["seq"]),
                    text=row["text"],
                    tags=tags,
                )
        return [out_map[x] for x in ids if x in out_map]

    def iter_docs_by_segment(self, segment_id: str) -> Iterator[DocRecord]:
        cur = self._conn.execute(
            "SELECT doc_id, segment_id, row_offset, seq, text, tags FROM docs WHERE segment_id=? ORDER BY row_offset",
            (segment_id,),
        )
        for row in cur:
            tags = json.loads(row["tags"]) if row["tags"] else None
            yield DocRecord(
                doc_id=row["doc_id"],
                segment_id=row["segment_id"],
                row_offset=int(row["row_offset"]),
                seq=int(row["seq"]),
                text=row["text"],
                tags=tags,
            )

    # ---------- Bitmaps (TEXT; latin-1 safe) ----------
    def _coerce_blob_in(self, blob: BitmapBlob) -> str:
        # Accept bytes or str; store TEXT (latin-1)
        if isinstance(blob, bytes):
            return blob.decode("latin-1")
        return blob

    def _cache_get(self, name: str) -> Optional[str]:
        blob = self._bm_cache.get(name)
        if blob is not None:
            # LRU touch
            self._bm_cache.move_to_end(name)
        return blob

    def _cache_put(self, name: str, blob_text: str) -> None:
        self._bm_cache[name] = blob_text
        self._bm_cache.move_to_end(name)
        while len(self._bm_cache) > self._bm_cache_size:
            self._bm_cache.popitem(last=False)

    def _touch_last_used(self, name: str) -> None:
        now = int(time.time())
        self._conn.execute("UPDATE bitmaps SET last_used=? WHERE name=?", (now, name))
        if not self._in_tx:
            self._conn.commit()

    def get_bitmap(self, name: str) -> Optional[BitmapBlob]:
        cached = self._cache_get(name)
        if cached is not None:
            # Update last_used even on cache hit
            self._touch_last_used(name)
            return cached

        row = self._conn.execute("SELECT data FROM bitmaps WHERE name=?", (name,)).fetchone()
        if row is None:
            return None

        blob_text: str = row["data"]
        self._cache_put(name, blob_text)
        self._touch_last_used(name)
        return blob_text

    def put_bitmap(self, name: str, blob: BitmapBlob) -> None:
        blob_text = self._coerce_blob_in(blob)
        now = int(time.time())
        self._conn.execute(
            """
            INSERT INTO bitmaps(name, data, last_used)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              data=excluded.data,
              last_used=excluded.last_used
            """,
            (name, blob_text, now),
        )
        self._cache_put(name, blob_text)
        if not self._in_tx:
            self._conn.commit()

    def delete_bitmap(self, name: str) -> None:
        self._conn.execute("DELETE FROM bitmaps WHERE name=?", (name,))
        if name in self._bm_cache:
            self._bm_cache.pop(name, None)
        if not self._in_tx:
            self._conn.commit()

    def list_bitmaps(self, prefix: Optional[str] = None) -> List[str]:
        if prefix:
            cur = self._conn.execute("SELECT name FROM bitmaps WHERE name LIKE ? ORDER BY name", (f"{prefix}%",))
        else:
            cur = self._conn.execute("SELECT name FROM bitmaps ORDER BY name")
        return [r["name"] for r in cur.fetchall()]

    def iter_recent_doc_ids(self, limit: int):
        cur = self._conn.execute(
            "SELECT doc_id FROM docs ORDER BY seq DESC LIMIT ?",
            (int(limit),),
        )
        for row in cur:
            yield row["doc_id"]

    # ---------- Stats ----------
    def get_stat(self, key: str) -> Optional[int]:
        row = self._conn.execute("SELECT value FROM stats WHERE key=?", (key,)).fetchone()
        return int(row["value"]) if row else None

    def put_stat(self, key: str, value: int) -> None:
        self._conn.execute(
            """
            INSERT INTO stats(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, int(value)),
        )
        if not self._in_tx:
            self._conn.commit()

    # ---------- KV ----------
    def get_kv(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def put_kv(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO kv(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )
        if not self._in_tx:
            self._conn.commit()

    def delete_kv(self, key: str) -> None:
        self._conn.execute("DELETE FROM kv WHERE key=?", (key,))
        if not self._in_tx:
            self._conn.commit()

    # ---------- Lifecycle ----------
    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
