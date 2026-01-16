from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import os
import json
import time
import contextlib
import numpy as np
from scipy.sparse import csr_matrix
from pyroaring import BitMap as Roaring

# Optional platform-specific locking modules
try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore
try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None  # type: ignore

from recollex.abcs import MetadataStore, DocRecord
from recollex.hooks import (
    make_profile,
    candidate_supplier_default,
    candidate_supplier_recent,
    rank_merge_heap,
    rank_merge_recent,
    score_csr_slice,
    score_hook_noop,
    evict_lru,
)
from recollex.io import open_segment, write_segments, SQLiteMetadataStore
from recollex.encoder.splade import SpladeEncoder
from recollex.bitmaps import deserialize_bitmap_blob, TERM_PREFIX, TAG_PREFIX, LIVE_DOCS, TOMBSTONES


def _has_only_everything(xs: Optional[Sequence[str]]) -> bool:
    if not xs:
        return False
    return all(str(x) == "everything" for x in xs)


def _apply_project_to_filters(
    project: Optional[str],
    filters: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """
    If project is provided, ensure filters['project'] is set (without overriding an explicit filter).
    """
    if project is None:
        return filters
    out = dict(filters or {})
    out.setdefault("project", str(project))
    return out


def _tag_bitmap_key(tag: Any) -> Optional[str]:
    """
    Normalize a tag spec into a TAG_PREFIX'ed bitmap name.

    Supported forms:
      - "tenant:acme"            -> "tag:tenant:acme"
      - ("tenant", "acme")       -> "tag:tenant=acme"
      - {"tenant": "acme"}       -> "tag:tenant=acme"
      - "everything"             -> None (treated as no restriction)
    """
    if isinstance(tag, tuple) and len(tag) == 2:
        k, v = tag
        return f"{TAG_PREFIX}{k}={v}"
    if isinstance(tag, dict) and len(tag) == 1:
        (k, v), = tag.items()
        return f"{TAG_PREFIX}{k}={v}"
    s = str(tag)
    if s == "everything":
        return None
    return f"{TAG_PREFIX}{s}"


def _tags_views(raw: Any) -> Tuple[List[str], Dict[str, str]]:
    """
    Normalize stored tags into:
      - tags_list: list[str] of tag strings
      - tags_dict: dict[str,str] parsed from 'k:v' entries in tags_list (last wins per key)
    """
    if raw is None:
        return [], {}

    if isinstance(raw, dict):
        # Preserve original dict semantics: keys/values as-is
        tags_list = [f"{k}:{v}" for k, v in raw.items()]
    elif isinstance(raw, (list, tuple, set)):
        # Already a collection of tag strings
        tags_list = [str(t) for t in raw]
    else:
        # Unsupported/opaque type: no parsed tags
        tags_list = []

    tags_dict: Dict[str, str] = {}
    for t in tags_list:
        if ":" in t:
            k, v = t.split(":", 1)
            tags_dict[k] = v
    return tags_list, tags_dict


class Recollex:
    """
    Minimal engine wiring:
      - Bitmaps in the DB store integer doc IDs (assumed numeric; we coerce via str(int)).
      - Segments live under index_path/segments/<segment_id> with npy arrays.
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        store: Optional[MetadataStore] = None,
        encoder_model: Optional[str] = None,
        encoder: Optional[Any] = None,
        *,
        seg_cache_max: int = 64,
        csr_cache_max: int = 128,
        csr_ram_limit_bytes: Optional[int] = 512 * 1024 * 1024,
    ) -> None:
        self._base = Path(index_path)
        # Ensure base directory and segments dir exist
        self._base.mkdir(parents=True, exist_ok=True)
        self._segments_dir = self._base / "segments"
        self._segments_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._base / "manifest.json"
        self._manifest_lock_path = self._base / "manifest.lock"

        # Default store if none provided
        self._store = store if store is not None else SQLiteMetadataStore(self._base / "meta.sqlite")

        # Load or initialize manifest
        self._manifest = self._load_manifest()
        if not self._manifest_path.exists():
            # Write a default manifest to mark a new index directory
            self._write_manifest_atomic(self._manifest)

        self._lru_clock: int = 0
        # segment reader cache (per segment_id)
        self._seg_cache_max = int(seg_cache_max)
        self._seg_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"reader": NpySegmentReader, "last_used": int}
        # CSR cache (per (segment_id, dims))
        self._csr_cache_max = int(csr_cache_max)
        self._csr_cache: Dict[str, Any] = {
            "entries": {},             # (segment_id, dims) -> {"csr": csr_matrix, "last_used": int, "size": int}
            "total_size": 0,
        }
        if csr_ram_limit_bytes is not None:
            self._csr_cache["ram_limit_bytes"] = int(csr_ram_limit_bytes)
        self._encoder_model: Optional[str] = encoder_model
        # Allow injecting a ready-made encoder instance (public API).
        # If encoder is None we keep lazy instantiation behavior.
        self._encoder: Optional[Any] = encoder

    @classmethod
    def open(
        cls,
        index_path: Union[str, Path],
        store: Optional[MetadataStore] = None,
        encoder_model: Optional[str] = None,
        encoder: Optional[Any] = None,
        *,
        seg_cache_max: int = 64,
        csr_cache_max: int = 128,
        csr_ram_limit_bytes: Optional[int] = 512 * 1024 * 1024,
    ) -> "Recollex":
        return cls(
            index_path,
            store,
            encoder_model,
            encoder,
            seg_cache_max=seg_cache_max,
            csr_cache_max=csr_cache_max,
            csr_ram_limit_bytes=csr_ram_limit_bytes,
        )

    # --------------- internal helpers ---------------

    def _deserialize_bitmap(self, name: str) -> Roaring:
        blob = self._store.get_bitmap(name)
        if blob is None:
            return Roaring()
        return deserialize_bitmap_blob(blob)

    def _get_bitmap(self, name: str) -> Roaring:
        return self._deserialize_bitmap(name)

    def _all_docs_bitmap(self) -> Roaring:
        """
        Build a bitmap of all live docs by unioning known term/tag postings,
        minus tombstones.
        """
        # Prefer explicit live_docs bitmap (minus tombstones) if present
        live = self._get_bitmap(LIVE_DOCS)
        if live:
            tomb = self._get_bitmap(TOMBSTONES)
            if tomb:
                live -= tomb
            return live
        bm = Roaring()
        # If store supports listing, union term and tag postings
        try:
            for prefix in (TERM_PREFIX, TAG_PREFIX):
                for name in self._store.list_bitmaps(prefix):
                    blob = self._store.get_bitmap(name)
                    if blob is not None:
                        bm |= deserialize_bitmap_blob(blob)
        except Exception:
            # If listing bitmaps is unsupported or fails, fall back to empty bm
            pass
        tomb = self._get_bitmap(TOMBSTONES)
        if tomb:
            bm -= tomb
        return bm

    def _ensure_encoder(self) -> SpladeEncoder:
        if self._encoder is None:
            self._encoder = SpladeEncoder(model=self._encoder_model) if self._encoder_model else SpladeEncoder()
        return self._encoder

    def _q_terms_from_text(self, text: str) -> List[Tuple[int, float]]:
        enc = self._ensure_encoder()
        idxs, vals = enc.encode(text)
        return [(int(t), float(w)) for t, w in zip(idxs, vals)]

    def _allocate_doc_id(self) -> int:
        cur_next = self._store.get_kv("next_doc_id")
        next_id = int(cur_next) if cur_next is not None else 1
        self._store.put_kv("next_doc_id", str(next_id + 1))
        return next_id

    def _build_base_bitmap(
        self,
        q_terms: Sequence[Tuple[int, float]],
        exclude_doc_ids: Optional[Iterable[str]],
        filters: Optional[Dict[str, str]] = None,
        tags_all_of: Optional[Sequence[Any]] = None,
        tags_one_of: Optional[Sequence[Any]] = None,
        tags_none_of: Optional[Sequence[Any]] = None,
    ) -> Roaring:
        if isinstance(tags_all_of, dict):
            tags_all_of = [tags_all_of]
        if isinstance(tags_one_of, dict):
            tags_one_of = [tags_one_of]
        if isinstance(tags_none_of, dict):
            tags_none_of = [tags_none_of]

        # Intersect/union tag bitmaps; support special 'none' (empty), and treat 'everything' as no restriction.
        base: Optional[Roaring] = None
        if filters:
            if "none" in filters:
                return Roaring()
            for k, v in filters.items():
                bm = self._get_bitmap(f"{TAG_PREFIX}{k}={v}")
                base = bm if base is None else (base & bm)

        # all_of_tags: intersection
        if tags_all_of and not _has_only_everything(tags_all_of):
            for t in tags_all_of:
                key = _tag_bitmap_key(t)
                if key is None:
                    continue
                bm = self._get_bitmap(key)
                base = bm if base is None else (base & bm)

        # one_of_tags: union then intersect with base (if any)
        if tags_one_of and not _has_only_everything(tags_one_of):
            union = Roaring()
            any_union = False
            for t in tags_one_of:
                key = _tag_bitmap_key(t)
                if key is None:
                    continue
                bm = self._get_bitmap(key)
                if bm:
                    union |= bm
                    any_union = True
            if any_union:
                base = union if base is None else (base & union)

        # Ensure we have some base before applying negative filters
        if base is None:
            base = Roaring()
            for tid, _wt in q_terms:
                bm = self._get_bitmap(f"{TERM_PREFIX}{int(tid)}")
                if bm:
                    base |= bm

        # If no positive scope and empty query, start from 'all docs' so none_of can subtract from universe
        if (not q_terms) and tags_none_of and len(base) == 0:
            base = self._all_docs_bitmap()

        # none_of_tags: subtract union (ignore 'everything')
        if tags_none_of:
            if not _has_only_everything(tags_none_of):
                excl = Roaring()
                for t in tags_none_of:
                    key = _tag_bitmap_key(t)
                    if key is None:
                        continue
                    bm = self._get_bitmap(key)
                    if bm:
                        excl |= bm
                if excl:
                    base -= excl

        # Subtract tombstones
        tomb = self._get_bitmap(TOMBSTONES)
        if tomb:
            base -= tomb

        # Subtract explicit exclusions (numeric doc_ids only)
        if exclude_doc_ids:
            ex = Roaring()
            for did in exclude_doc_ids:
                try:
                    ex.add(int(did))
                except Exception:
                    # Non-numeric doc_ids cannot be represented in the bitmap domain; skip.
                    continue
            if ex:
                base -= ex

        return base

    def _open_segment(self, segment_id: str, dims: Optional[int]) -> Any:
        reader_entry = self._seg_cache.get(segment_id)
        if reader_entry is None:
            seg_path = self._segments_dir / segment_id
            reader = open_segment(seg_path, dims=dims)
            self._lru_clock += 1
            self._seg_cache[segment_id] = {"reader": reader, "last_used": self._lru_clock}
            # Evict if over max items
            if len(self._seg_cache) > self._seg_cache_max:
                # Build a minimal state for evict_lru
                state = {k: {"last_used": v["last_used"], "size": 1} for k, v in self._seg_cache.items()}
                evicted = evict_lru(state, max_items=self._seg_cache_max)
                for evk in evicted:
                    try:
                        self._seg_cache[evk]["reader"].close()
                    except Exception:
                        pass
                    self._seg_cache.pop(evk, None)
            return reader
        # Existing entry
        reader = reader_entry["reader"]
        # Ensure dims large enough; reopen if needed
        if dims is not None and reader.dims < int(dims):
            seg_path = self._segments_dir / segment_id
            try:
                reader.close()
            except Exception:
                pass
            reader = open_segment(seg_path, dims=dims)
            reader_entry["reader"] = reader
        # Touch LRU
        self._lru_clock += 1
        reader_entry["last_used"] = self._lru_clock
        return reader

    def _q_to_csr(self, q_terms: Sequence[Tuple[int, float]], target_dims: Optional[int] = None) -> Tuple[csr_matrix, int]:
        if not q_terms:
            dims = int(target_dims or 0)
            return csr_matrix((1, dims), dtype=np.float32), dims
        # collapse duplicates: sum weights per term id
        agg: Dict[int, float] = {}
        for tid, wt in q_terms:
            t = int(tid)
            agg[t] = agg.get(t, 0.0) + float(wt)
        cols = np.fromiter(agg.keys(), dtype=np.int32)
        data = np.fromiter((agg[t] for t in agg.keys()), dtype=np.float32)
        # Optionally sort by col
        order = np.argsort(cols, kind="mergesort")
        cols = cols[order]
        data = data[order]
        indptr = np.asarray([0, cols.shape[0]], dtype=np.int64)
        if target_dims is not None:
            # Ensure query term ids fit under target dims (manifest encoder vocab)
            if cols.size > 0 and int(cols.max()) >= int(target_dims):
                raise ValueError(
                    f"Query term id {int(cols.max())} >= target dims {int(target_dims)}; encoder/index mismatch"
                )
            dims = int(target_dims)
        else:
            dims = int(cols.max() + 1)
        q = csr_matrix((data, cols, indptr), shape=(1, dims), dtype=np.float32)
        return q, dims

    def _load_manifest(self) -> Dict[str, Any]:
        if self._manifest_path.exists():
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"version": 1, "dims": 0, "segments": []}

    def _write_manifest_atomic(self, manifest: Dict[str, Any]) -> None:
        tmp = self._manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, self._manifest_path)
        self._manifest = manifest

    @contextlib.contextmanager
    def _manifest_lock(self, timeout: float = 30.0):
        """
        Cross-platform file lock guarding manifest/segment critical section.
        Locks self._manifest_lock_path; waits up to `timeout` seconds.
        """
        lock_file = str(self._manifest_lock_path)
        # Ensure directory exists
        self._manifest_lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Open/ensure the lock file exists
        fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o666)
        start = time.time()
        acquired = False
        lock_impl = None  # "fcntl" | "msvcrt" | "pid"

        # Allow tests to force the sidecar PID lock path for deterministic contention
        force_pid_lock = os.environ.get("RECOLLEX_FORCE_PID_LOCK") == "1"
        use_fcntl = (fcntl is not None) and not force_pid_lock
        use_msvcrt = (msvcrt is not None) and not force_pid_lock

        try:
            if use_fcntl:
                # POSIX path
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                        acquired = True
                        lock_impl = "fcntl"
                        break
                    except (BlockingIOError, OSError):
                        if (time.time() - start) > timeout:
                            raise TimeoutError("Timeout acquiring manifest lock")
                        time.sleep(0.05)
            elif use_msvcrt:  # pragma: no cover
                # Windows path
                # msvcrt.locking locks a byte range; lock 1 byte at start
                while True:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                        acquired = True
                        lock_impl = "msvcrt"
                        break
                    except OSError:  # pragma: no cover
                        if (time.time() - start) > timeout:
                            raise TimeoutError("Timeout acquiring manifest lock")
                        time.sleep(0.05)
            else:
                # Fallback: advisory spin on exclusive create of a .pid sidecar
                sidecar = lock_file + ".pid"
                while True:
                    try:
                        pid_fd = os.open(sidecar, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
                        os.write(pid_fd, str(os.getpid()).encode("ascii"))
                        os.close(pid_fd)
                        acquired = True
                        lock_impl = "pid"
                        break
                    except FileExistsError:
                        if (time.time() - start) > timeout:
                            raise TimeoutError("Timeout acquiring manifest lock")
                        time.sleep(0.05)
            yield
        finally:
            try:
                if acquired:
                    if lock_impl == "fcntl" and fcntl is not None:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    elif lock_impl == "msvcrt" and msvcrt is not None:  # pragma: no cover
                        try:
                            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
                        except Exception:  # pragma: no cover
                            pass
                    elif lock_impl == "pid":
                        try:
                            if os.path.exists(lock_file + ".pid"):
                                os.remove(lock_file + ".pid")
                        except Exception:
                            pass
            finally:
                try:
                    os.close(fd)
                except Exception:
                    pass

    # --------------- public API ---------------

    def add_many(
        self,
        docs: Iterable[Dict[str, Any]],
        segment_id: Optional[str] = None,
        dims: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Build one segment from docs, write CSR arrays, atomically swap manifest,
        and update docs/bitmaps/stats/kv in a single SQL transaction.

        Each doc: {
          "doc_id": str|int (must be numeric-parsable),
          "indices": List[int],
          "data": List[float],
          "text": Optional[str],
          "tags": Optional[Dict[str, Any]]   # contributes to tag:<k>=<v> bitmaps
        }
        """
        docs_list = list(docs)
        if not docs_list:
            return {"n_docs": 0, "nnz": 0}

        # Stage CSR
        indptr: List[int] = [0]
        indices_all: List[int] = []
        data_all: List[float] = []
        row_ids: List[str] = []
        nnz = 0
        max_tid = -1

        def as_int_doc_id(did: str) -> int:
            try:
                return int(did)
            except Exception:
                raise ValueError(f"doc_id must be numeric (bitmap domain is int); got '{did}'")

        updated_bitmaps: Dict[str, Roaring] = {}
        df_increments: Dict[int, int] = {}
        docs_meta: List[Tuple[str, Optional[str], Any, Optional[int]]] = []
        dids_added_int: List[int] = []

        for d in docs_list:
            did = str(d["doc_id"])
            _ = as_int_doc_id(did)  # validate
            idxs = [int(x) for x in d["indices"]]
            vals = [float(x) for x in d["data"]]
            if len(idxs) != len(vals):
                raise ValueError(f"indices/data length mismatch for doc_id={did}")

            indices_all.extend(idxs)
            data_all.extend(vals)
            nnz += len(idxs)
            indptr.append(nnz)
            row_ids.append(did)
            if idxs:
                max_tid = max(max_tid, max(idxs))

            for tid in set(idxs):
                df_increments[tid] = df_increments.get(tid, 0) + 1

            docs_meta.append((did, d.get("text"), d.get("tags"), d.get("seq")))

        # Critical section: guard segment id allocation, segment write, SQL updates, and manifest swap
        with self._manifest_lock(timeout=30.0):
            # Reload manifest to avoid using a stale in-memory copy
            current_manifest = self._load_manifest()

            # Segment id
            if segment_id is None:
                existing = [seg.get("name") for seg in current_manifest.get("segments", [])]
                seg_num = 0
                while True:
                    cand = f"seg_{seg_num:03d}"
                    if cand not in existing:
                        segment_id = cand
                        break
                    seg_num += 1

            # Dims
            inferred_dims = (max_tid + 1) if max_tid >= 0 else 0
            D = int(dims if dims is not None else inferred_dims)
            if D < inferred_dims:
                raise ValueError(f"dims ({D}) smaller than required ({inferred_dims}) for new segment")

            # Persist CSR arrays
            seg_path = self._segments_dir / str(segment_id)
            meta = write_segments(
                seg_path,
                np.asarray(indptr, dtype=np.int64),
                np.asarray(indices_all, dtype=np.int32),
                np.asarray(data_all, dtype=np.float32),
                row_ids,
            )

            # Single SQL transaction: docs, bitmaps, stats, kv
            with self._store.transaction():
                # assign seq at add-time
                cur_next_seq = self._store.get_kv("next_seq")
                next_seq = int(cur_next_seq) if cur_next_seq is not None else 1

                def get_or_new_bm(name: str) -> Roaring:
                    bm = updated_bitmaps.get(name)
                    if bm is not None:
                        return bm
                    blob = self._store.get_bitmap(name)
                    if blob is None:
                        bm = Roaring()
                    else:
                        bm = deserialize_bitmap_blob(blob)
                    updated_bitmaps[name] = bm
                    return bm

                # Upsert docs and accumulate bitmaps
                for row_off, (did, text, tags, opt_seq) in enumerate(docs_meta):
                    rec = DocRecord(
                        doc_id=did,
                        segment_id=str(segment_id),
                        row_offset=int(row_off),
                        seq=int(opt_seq) if opt_seq is not None else int(next_seq),
                        text=text,
                        tags=tags,
                    )
                    self._store.upsert_doc(rec)
                    did_int = int(did)
                    dids_added_int.append(did_int)
                    # Tags: support dict (k=v -> tag:k=v) and list/tuple/set (-> tag:<tag>)
                    if tags:
                        if isinstance(tags, dict):
                            for k, v in tags.items():
                                get_or_new_bm(f"{TAG_PREFIX}{k}={v}").add(did_int)
                        elif isinstance(tags, (list, tuple, set)):
                            for t in tags:
                                get_or_new_bm(f"{TAG_PREFIX}{str(t)}").add(did_int)
                        else:
                            get_or_new_bm(f"{TAG_PREFIX}{str(tags)}").add(did_int)

                    row_start = indptr[row_off]
                    row_end = indptr[row_off + 1]
                    row_terms = set(indices_all[row_start:row_end])
                    for tid in row_terms:
                        get_or_new_bm(f"{TERM_PREFIX}{int(tid)}").add(did_int)

                    used = int(opt_seq) if opt_seq is not None else int(next_seq)
                    next_seq = max(int(next_seq), int(used)) + 1

                # DF stats
                for tid, delta in df_increments.items():
                    key = f"term_df:{int(tid)}"
                    cur = self._store.get_stat(key) or 0
                    self._store.put_stat(key, int(cur) + int(delta))

                # Write updated bitmaps
                for name, bm in updated_bitmaps.items():
                    self._store.put_bitmap(name, bm.serialize())

                # Persist next_seq
                self._store.put_kv("next_seq", str(next_seq))

                # Update live_docs and its count
                bm_live = get_or_new_bm(LIVE_DOCS)
                if dids_added_int:
                    for x in dids_added_int:
                        bm_live.add(int(x))
                    self._store.put_bitmap(LIVE_DOCS, bm_live.serialize())
                    self._store.put_stat("live_docs_count", int(len(bm_live)))

            # Atomic manifest swap (based on freshly loaded manifest)
            manifest = dict(current_manifest)
            m_dims = int(manifest.get("dims", 0) or 0)
            if m_dims == 0:
                manifest["dims"] = int(D)
            elif D != 0 and D != m_dims:
                raise ValueError(f"Manifest dims ({m_dims}) != provided dims ({D})")
            seg_entry = {"name": str(segment_id), "rows": [0, int(meta["n_rows"])]}
            segments = list(manifest.get("segments", []))
            segments.append(seg_entry)
            manifest["segments"] = segments
            self._write_manifest_atomic(manifest)

        return {"n_docs": int(meta["n_rows"]), "nnz": int(meta["nnz"])}

    def add(self, text_or_batch: Union[str, List[Any]], tags: Optional[Union[Dict[str, Any], Sequence[str]]] = None, timestamp: Optional[int] = None) -> Union[int, List[int]]:
        # Batch mode: list means batch. Each item: (text, tags, timestamp) or {"text":..., "tags":..., "timestamp":...}
        if isinstance(text_or_batch, list):
            items = text_or_batch
            enc = self._ensure_encoder()
            texts = [(it["text"] if isinstance(it, dict) else it[0]) for it in items]
            encs = enc.encode_many(texts)
            with self._store.transaction():
                ids = [self._allocate_doc_id() for _ in items]
            docs = []
            for did, it, (idxs, vals) in zip(ids, items, encs):
                if isinstance(it, dict):
                    text = it["text"]
                    tags_i = it.get("tags")
                    ts = it.get("timestamp", it.get("seq", None))
                else:
                    text, tags_i, ts = it
                # Preserve dict tags as-is; normalize sequences to list[str]
                if isinstance(tags_i, dict):
                    tags_norm: Any = tags_i
                else:
                    tags_norm = list(tags_i or [])
                docs.append({
                    "doc_id": did,
                    "indices": idxs,
                    "data": vals,
                    "text": text,
                    "tags": tags_norm,
                    "seq": (int(ts) if ts is not None else None),
                })
            self.add_many(docs, segment_id=None, dims=enc.dims)
            return [int(x) for x in ids]

        # Single-item mode (back-compat)
        # Accept either plain string or a tuple like (text, tags?, timestamp?)
        if isinstance(text_or_batch, tuple):
            text = text_or_batch[0]
            tags_i = text_or_batch[1] if len(text_or_batch) > 1 else tags
            ts_local = int(text_or_batch[2]) if len(text_or_batch) > 2 else (int(timestamp) if timestamp is not None else None)
        else:
            text = text_or_batch
            tags_i = tags
            ts_local = int(timestamp) if timestamp is not None else None

        # Preserve dict tags; normalize sequences to list[str]
        if isinstance(tags_i, dict):
            tags_local = tags_i
        else:
            tags_local = list(tags_i or [])

        enc = self._ensure_encoder()
        idxs, vals = enc.encode(text)
        with self._store.transaction():
            new_id = self._allocate_doc_id()
        doc = {
            "doc_id": new_id,
            "indices": idxs,
            "data": vals,
            "text": text,
            "tags": tags_local,
            "seq": ts_local,
        }
        self.add_many([doc], segment_id=None, dims=enc.dims)
        return int(new_id)

    def _remove_doc_terms_and_update_df(self, did_int: int, rec: DocRecord) -> None:
        """
        Remove doc id from its term bitmaps and decrement corresponding DF stats.
        Safe to call inside an active store.transaction().
        """
        try:
            reader = self._open_segment(rec.segment_id, dims=None)
            indptr = reader.indptr
            indices = reader.indices
            roff = int(rec.row_offset)
            start = int(indptr[roff])
            end = int(indptr[roff + 1])
        except Exception:
            return
        terms = set(int(t) for t in indices[start:end].tolist())
        for t in terms:
            name = f"{TERM_PREFIX}{int(t)}"
            bm_t = self._get_bitmap(name)
            removed = False
            try:
                bm_t.remove(int(did_int))
                removed = True
                self._store.put_bitmap(name, bm_t.serialize())
            except KeyError:
                pass
            if removed:
                stat_key = f"term_df:{int(t)}"
                cur = int(self._store.get_stat(stat_key) or 0)
                new = max(0, cur - 1)
                if new != cur:
                    self._store.put_stat(stat_key, new)

    def _remove_doc_tags_and_update_bitmaps(self, did_int: int, rec: DocRecord) -> None:
        """
        Remove doc id from all relevant tag bitmaps based on rec.tags.
        Safe to call inside an active store.transaction().
        """
        tags = rec.tags
        if not tags:
            return
        try:
            if isinstance(tags, dict):
                for k, v in tags.items():
                    name = f"{TAG_PREFIX}{k}={v}"
                    bm = self._get_bitmap(name)
                    try:
                        bm.remove(int(did_int))
                        self._store.put_bitmap(name, bm.serialize())
                    except KeyError:
                        pass
            elif isinstance(tags, (list, tuple, set)):
                for t in tags:
                    name = f"{TAG_PREFIX}{str(t)}"
                    bm = self._get_bitmap(name)
                    try:
                        bm.remove(int(did_int))
                        self._store.put_bitmap(name, bm.serialize())
                    except KeyError:
                        pass
            else:
                name = f"{TAG_PREFIX}{str(tags)}"
                bm = self._get_bitmap(name)
                try:
                    bm.remove(int(did_int))
                    self._store.put_bitmap(name, bm.serialize())
                except KeyError:
                    pass
        except Exception:
            # Be defensive; tag removal should not abort the whole remove flow
            pass

    def _live_docs_remove(self, ids: Iterable[int]) -> None:
        """
        Remove given ids from LIVE_DOCS bitmap and update its count stat.
        Safe to call inside an active store.transaction().
        """
        bm_live = self._get_bitmap(LIVE_DOCS)
        if not bm_live:
            return
        to_remove = Roaring()
        for did in ids:
            try:
                to_remove.add(int(did))
            except Exception:
                pass
        if len(to_remove) > 0:
            bm_live -= to_remove
            self._store.put_bitmap(LIVE_DOCS, bm_live.serialize())
            self._store.put_stat("live_docs_count", int(len(bm_live)))

    def _docs_by_id(self, doc_ids: Sequence[str], build_index: bool = True) -> Tuple[List[DocRecord], Optional[Dict[str, DocRecord]]]:
        """
        Fetch documents by ids and return (list, optional dict_indexed_by_doc_id).
        Pass build_index=False when you only need the list to skip building the dict and save work.
        """
        recs = self._store.get_docs_many(list(doc_ids))
        return recs, ({r.doc_id: r for r in recs} if build_index else None)

    def _normalize_doc_ids(self, id_or_ids: Union[str, int, Iterable[Union[str, int]]]) -> List[int]:
        """
        Coerce input id(s) into a list of ints; drop any non-numeric values.
        """
        if isinstance(id_or_ids, (list, tuple, set)):
            it = id_or_ids
        else:
            it = [id_or_ids]
        out: List[int] = []
        for did in it:
            try:
                out.append(int(did))
            except Exception:
                continue
        return out

    def _group_candidate_docs(
        self,
        doc_ids: Sequence[str],
    ) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, int], str], Dict[str, DocRecord]]:
        """
        Given candidate doc_ids, fetch records and build:
          - per_segment_rows: {segment_id: [row_offset, ...]}
          - index: {(segment_id, row_offset): doc_id}
          - doc_by_id: {doc_id: DocRecord}
        """
        recs, doc_by_id = self._docs_by_id(doc_ids, build_index=True)
        per_segment_rows: Dict[str, List[int]] = {}
        index: Dict[Tuple[str, int], str] = {}
        for rec in recs:
            roff = int(rec.row_offset)
            per_segment_rows.setdefault(rec.segment_id, []).append(roff)
            index[(rec.segment_id, roff)] = rec.doc_id
        # doc_by_id is Optional in _docs_by_id; assert for type
        assert doc_by_id is not None
        return per_segment_rows, index, doc_by_id

    def _csr_for_segment(self, seg_id: str, dims: Optional[int]) -> csr_matrix:
        """
        Get or build the CSR matrix for a segment, handling LRU touch and eviction.
        """
        reader = self._open_segment(seg_id, dims=dims)
        key = (seg_id, int(reader.dims))
        entries = self._csr_cache["entries"]
        entry = entries.get(key)
        if entry is None:
            n_rows = int(reader.indptr.shape[0] - 1)
            X = csr_matrix((reader.data, reader.indices, reader.indptr), shape=(n_rows, int(reader.dims)))
            size_bytes = int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
            self._lru_clock += 1
            entries[key] = {"csr": X, "last_used": self._lru_clock, "size": size_bytes}
            self._csr_cache["total_size"] = int(self._csr_cache["total_size"]) + size_bytes
            # Enforce eviction policy (count first, then RAM if configured)
            evicted = evict_lru(self._csr_cache, max_items=self._csr_cache_max)
            for evk in evicted:
                sz = int(entries.get(evk, {}).get("size", 0))
                entries.pop(evk, None)
                self._csr_cache["total_size"] = int(self._csr_cache["total_size"]) - sz
            return X
        # Touch LRU and return existing CSR
        self._lru_clock += 1
        entry["last_used"] = self._lru_clock
        return entry["csr"]

    def remove(self, id_or_ids: Union[str, int, List[Union[str, int]]]) -> None:
        ids = self._normalize_doc_ids(id_or_ids if isinstance(id_or_ids, list) else [id_or_ids])
        if not ids:
            return
        with self._store.transaction():
            # Mark tombstones
            bm = self._get_bitmap(TOMBSTONES)
            for did in ids:
                bm.add(int(did))
            self._store.put_bitmap(TOMBSTONES, bm.serialize())

            # DF/term bitmap maintenance
            for did in ids:
                rec = self._store.get_doc(str(did))
                if rec is None:
                    continue
                try:
                    self._remove_doc_tags_and_update_bitmaps(int(did), rec)
                except Exception:
                    pass
                try:
                    self._remove_doc_terms_and_update_df(int(did), rec)
                except Exception:
                    pass

            # Update live_docs via helper
            self._live_docs_remove(ids)

    def remove_by(
        self,
        *,
        all_of_tags: Optional[Sequence[str]] = None,
        one_of_tags: Optional[Sequence[str]] = None,
        none_of_tags: Optional[Sequence[str]] = None,
        dry_run: bool = False,
        project: Optional[str] = None,
    ) -> int:
        """
        Remove all documents matching the provided scope (tags/project).
        Returns the number of docs that would be (or were) removed.
        """
        filters = _apply_project_to_filters(project, None)
        base = self._build_base_bitmap(
            q_terms=[],
            exclude_doc_ids=None,
            filters=filters,
            tags_all_of=all_of_tags,
            tags_one_of=one_of_tags,
            tags_none_of=none_of_tags,
        )
        if len(base) == 0:
            return 0
        ids = [int(x) for x in base]
        if dry_run:
            return len(ids)
        self.remove(ids)
        return len(ids)

    def search_terms(
        self,
        q_terms: Sequence[Tuple[int, float]],
        *,
        k: int = 50,
        profile: str = "rag",
        exclude_doc_ids: Optional[Iterable[str]] = None,
        override_knobs: Optional[Dict[str, Any]] = None,
        rerank_top_m: Optional[int] = None,  # placeholder; no reranker wired
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, str]] = None,             # legacy k=v
        tags_all_of: Optional[Sequence[str]] = None,          # new API
        tags_one_of: Optional[Sequence[str]] = None,          # new API
        tags_none_of: Optional[Sequence[str]] = None,         # new API (NOT)
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        - Build base bitmap: filters ∩ ...  minus tombstones and exclude_doc_ids.
        - filter_fn -> (must_terms, should_terms); candidate supplier -> candidates bitmap.
        - Group by segment; score per segment; merge; attach metadata.
        - If profile is recent/log_recent, bypass term gating and rank by seq desc.
        """
        override_knobs = override_knobs or {}

        # Choose policy/profile
        filter_fn, order = make_profile(profile, override_knobs)

        # Guard: if scoring profile and manifest dims known, ensure all query term ids fit
        D_manifest_check = int(self._manifest.get("dims", 0) or 0)
        if ((order != "recent") or (min_score is not None)) and D_manifest_check > 0 and q_terms:
            max_tid = max(int(t) for t, _ in q_terms)
            if max_tid >= D_manifest_check:
                raise ValueError(
                    f"Query term id {int(max_tid)} >= target dims {int(D_manifest_check)}; encoder/index mismatch"
                )

        # Build base
        filters = _apply_project_to_filters(project, filters)
        base = self._build_base_bitmap(
            q_terms=q_terms,
            exclude_doc_ids=exclude_doc_ids,
            filters=filters,
            tags_all_of=tags_all_of,
            tags_one_of=tags_one_of,
            tags_none_of=tags_none_of,
        )

        # Early exit if base empty and we have no terms (nothing to rank) -- only for score-based profiles
        if order != "recent" and len(base) == 0 and not q_terms:
            return []

        # Helpers for filter_fn
        def get_df(tid: int) -> int:
            val = self._store.get_stat(f"term_df:{int(tid)}")
            return int(val or 0)

        def get_bitmap_missing_none(name: str) -> Optional[Roaring]:
            blob = self._store.get_bitmap(name)
            return deserialize_bitmap_blob(blob) if blob is not None else None

        # Policy produces MUST/SHOULD
        must_terms: List[int]
        should_terms: List[int]
        if order == "recent":
            must_terms, should_terms = ([], [])
        else:
            must_terms, should_terms = filter_fn(
                q_terms=q_terms,
                filters=filters,
                get_bitmap=get_bitmap_missing_none,
                get_df=get_df,
                base_bitmap=base,
                exclude_doc_ids=exclude_doc_ids,
                knobs=override_knobs,
            )

        # Candidates
        budget = override_knobs.get("budget", None)
        if order == "recent":
            # DB-backed recency if scope is the universe; else bitmap-backed.
            def is_universe_scope() -> bool:
                # Universe when there are no positive restrictions (filters/all_of/one_of),
                # and none_of is empty or ["everything"].
                has_positive = bool(filters)
                if tags_all_of and not _has_only_everything(tags_all_of):
                    has_positive = True
                if tags_one_of and not _has_only_everything(tags_one_of):
                    has_positive = True
                none_restricts = bool(tags_none_of) and not _has_only_everything(tags_none_of)
                return (not has_positive) and (not none_restricts)

            if is_universe_scope():
                cand_doc_ids: List[str] = []
                lim = int(budget or k)
                ex = set(str(x) for x in (exclude_doc_ids or []))
                live = self._get_bitmap(LIVE_DOCS)
                tomb = self._get_bitmap(TOMBSTONES)
                for did in self._store.iter_recent_doc_ids(lim * 2):
                    if did in ex:
                        continue
                    try:
                        did_int = int(did)
                    except Exception:
                        did_int = None
                    # Exclude tombstoned docs (prefer LIVE_DOCS membership if present)
                    if live and did_int is not None and (did_int not in live):
                        continue
                    if (not live) and tomb and did_int is not None and (did_int in tomb):
                        continue
                    cand_doc_ids.append(did)
                    if len(cand_doc_ids) >= lim:
                        break
                if not cand_doc_ids:
                    return []
            else:
                candidates = candidate_supplier_recent(base, budget=budget)
                if len(candidates) == 0:
                    return []
                cand_doc_ids = [str(int(x)) for x in candidates]
        else:
            candidates = candidate_supplier_default(
                must_terms=must_terms,
                should_terms=should_terms,
                base_bitmap=base,
                get_bitmap=get_bitmap_missing_none,
                budget=budget,
            )
            if len(candidates) == 0:
                return []
            cand_doc_ids = [str(int(x)) for x in candidates]

        # Recency branch: rank by seq desc using store.get_doc
        if order == "recent":
            per_segment_rows, index, doc_by_id = self._group_candidate_docs(cand_doc_ids)

            # Optional: if a min_score is provided and we have query terms, gate by score >= threshold
            if min_score is not None and q_terms:
                D_manifest = int(self._manifest.get("dims", 0) or 0)
                q_csr, q_dims = self._q_to_csr(q_terms, target_dims=(D_manifest if D_manifest > 0 else None))
                thr = float(min_score)
                # Compute scores per segment; keep only rows with score >= thr
                passing_by_seg: Dict[str, set[int]] = {}
                for seg_id, rows in per_segment_rows.items():
                    X_use = self._csr_for_segment(seg_id, q_dims if q_dims > 0 else None)
                    if q_dims > 0:
                        scores = score_csr_slice(q_csr, {"csr": X_use}, rows)
                    else:
                        scores = [(int(r), 0.0) for r in rows]
                    keep = {int(r) for r, s in scores if float(s) >= thr}
                    if keep:
                        passing_by_seg[seg_id] = keep

                per_segment: Dict[str, List[Tuple[int, int]]] = {}
                for seg_id, rows in per_segment_rows.items():
                    keep = passing_by_seg.get(seg_id)
                    if not keep:
                        continue
                    filtered: List[Tuple[int, int]] = []
                    for r in rows:
                        if r in keep:
                            rec = doc_by_id[index[(seg_id, r)]]
                            filtered.append((int(r), int(rec.seq)))
                    if filtered:
                        per_segment[seg_id] = filtered
                if not per_segment:
                    return []
            else:
                per_segment = {}
                # Build (row_offset, seq) pairs from per_segment_rows
                for seg_id, rows in per_segment_rows.items():
                    recs_seg = [doc_by_id[index[(seg_id, r)]] for r in rows]
                    per_segment[seg_id] = [(int(r), int(rec.seq)) for r, rec in zip(rows, recs_seg)]

            merged = rank_merge_recent(per_segment, k)
            out: List[Dict[str, Any]] = []
            for seg_id, row_off, seq in merged:
                did = index.get((seg_id, int(row_off)))
                doc = doc_by_id.get(did) if did is not None else None
                if doc is None:
                    tags_list, tags_dict = [], {}
                else:
                    tags_list, tags_dict = _tags_views(doc.tags)
                out.append({
                    "doc_id": did,
                    "segment_id": seg_id,
                    "row_offset": int(row_off),
                    "seq": int(seq),
                    "score": 0.0,
                    "text": None if doc is None else doc.text,
                    "tags": tags_list,
                    "tags_list": tags_list,
                    "tags_dict": tags_dict,
                })
            return out

        # Score branch: csr slice + heap merge
        D_manifest = int(self._manifest.get("dims", 0) or 0)
        q_csr, q_dims = self._q_to_csr(q_terms, target_dims=(D_manifest if D_manifest > 0 else None))

        # Group candidates by segment (and collect row offsets) via store lookups
        per_segment_rows, index, doc_by_id = self._group_candidate_docs(cand_doc_ids)

        # Score per segment
        per_segment_scores: Dict[str, List[Tuple[int, float]]] = {}
        for seg_id, rows in per_segment_rows.items():
            X_use = self._csr_for_segment(seg_id, q_dims if q_dims > 0 else None)
            if q_dims > 0:
                scores = score_csr_slice(q_csr, {"csr": X_use}, rows)
            else:
                # No terms: noop scores (all zeros)
                scores = [(int(r), 0.0) for r in rows]
            per_segment_scores[seg_id] = scores

        merged = rank_merge_heap(per_segment_scores, k)
        if min_score is not None:
            thr = float(min_score)
            merged = [(seg_id, row_off, score) for (seg_id, row_off, score) in merged if float(score) >= thr]

        # Optional rerank placeholder (no-op)
        if rerank_top_m and rerank_top_m > 0:
            pass  # integrate reranker here when available

        # Attach metadata
        out: List[Dict[str, Any]] = []
        for seg_id, row_off, score in merged:
            did = index.get((seg_id, int(row_off)))
            doc = doc_by_id.get(did) if did is not None else None
            if doc is None:
                tags_list, tags_dict = [], {}
            else:
                tags_list, tags_dict = _tags_views(doc.tags)
            out.append({
                "doc_id": did,
                "segment_id": seg_id,
                "row_offset": int(row_off),
                "score": float(score),
                "seq": None if doc is None else int(doc.seq),
                "text": None if doc is None else doc.text,
                "tags": tags_list,
                "tags_list": tags_list,
                "tags_dict": tags_dict,
            })
        return out

    def search(
        self,
        text_or_texts: Union[str, List[str]],
        all_of_tags: Optional[Sequence[str]] = None,
        one_of_tags: Optional[Sequence[str]] = None,
        none_of_tags: Optional[Sequence[str]] = None,
        k: int = 50,
        profile: str = "rag",
        exclude_doc_ids: Optional[Iterable[str]] = None,
        override_knobs: Optional[Dict[str, Any]] = None,
        rerank_top_m: Optional[int] = None,
        min_score: Optional[float] = None,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Batch mode: list means batch
        if isinstance(text_or_texts, list):
            enc = self._ensure_encoder()
            q_encs = enc.encode_many([t or "" for t in text_or_texts])
            outs: List[List[Dict[str, Any]]] = []
            for idxs, vals in q_encs:
                q_terms = [(int(t), float(w)) for t, w in zip(idxs, vals)]
                outs.append(self.search_terms(
                    q_terms=q_terms,
                    k=int(k),
                    profile=profile,
                    exclude_doc_ids=exclude_doc_ids,
                    override_knobs=override_knobs,
                    rerank_top_m=rerank_top_m,
                    min_score=min_score,
                    filters=None,
                    tags_all_of=all_of_tags,
                    tags_one_of=one_of_tags,
                    tags_none_of=none_of_tags,
                    project=project,
                ))
            return outs

        # Single-item mode (back-compat)
        text = text_or_texts
        q_terms = self._q_terms_from_text(text) if text else []
        return self.search_terms(
            q_terms=q_terms,
            k=int(k),
            profile=profile,
            exclude_doc_ids=exclude_doc_ids,
            override_knobs=override_knobs,
            rerank_top_m=rerank_top_m,
            min_score=min_score,
            filters=None,
            tags_all_of=all_of_tags,
            tags_one_of=one_of_tags,
            tags_none_of=none_of_tags,
            project=project,
        )

    def last(self, k: int = 50, project: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convenience for recency-first:
        Equivalent to search(q_terms=[], profile='recent', score noop, rank by seq desc).
        """
        return self.search_terms(
            q_terms=[],
            k=int(k),
            profile="recent",
            exclude_doc_ids=None,
            override_knobs={},  # knobs unused for recent
            rerank_top_m=None,
            filters=None,
            project=project,
        )

    def close(self) -> None:
        """
        Close open resources: segment readers and the underlying store.
        Safe to call multiple times.
        """
        try:
            for ent in list(self._seg_cache.values()):
                try:
                    ent["reader"].close()
                except Exception:
                    pass
            self._seg_cache.clear()
        except Exception:
            pass
        try:
            if hasattr(self._store, "close"):
                self._store.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
