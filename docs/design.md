# Recollex Developer Guide

Sparse, filterable, RAM-friendly indexer for SPLADE-style vectors on Windows/Linux/macOS.

See docs/code_style.md for the project’s preferred code style (function-first hooks; ABCs for stateful components). This document is for maintainers; for black-box API usage see the README.

## 1) Mental model

- Store vectors as **CSR** triplets on disk. Slice rows. Dot with query.
- Store filters as **Roaring bitmaps** inside **SQLite** BLOBs. Intersect fast.
- Append in **segments**. Delete via **tombstone bitmap**. Compact later.
- Default encoder: SPLADE (seerware/Splade_PP_en_v2) via ONNX Runtime by default; sentence_transformers is used only when backend != "onnx". Pooling: ReLU → log(1+·) → reduce across tokens (max or sum). Use the same pooling for docs and queries.
- Behavior profiles: paraphrase_hp (high precision), rag (high recall), recent (recency-first; aliases: log_recent, recency, log). Profiles select hook presets and runtime knobs.
- Everything is local. No DB server. Optional ANN is out of scope here.

---

## 2) On-disk layout

```
recollex/
  manifest.json                  # index metadata
  segments/
    seg_000/
      indptr.npy  indices.npy  data.npy  row_ids.npy
    seg_001/ ...
  meta.sqlite                    # docs + bitmaps + catalogs
```

### manifest.json

dims is set when the first segment is written (typically equals the encoder tokenizer's vocab_size) and you do not configure it manually.
Subsequent segments must match manifest.dims; queries are validated against manifest.dims.

```json
{
  "version": 1,
  "dims": 30522,
  "segments": [
    { "name": "seg_000", "rows": [0, 100000] },
    { "name": "seg_001", "rows": [100000, 150000] }
  ]
}
```

---

## 3) SQLite schema (minimum)

```sql
-- Documents
CREATE TABLE IF NOT EXISTS docs(
  doc_id TEXT PRIMARY KEY,
  segment_id TEXT NOT NULL,
  row_offset INTEGER NOT NULL,
  seq INTEGER NOT NULL,       -- global insertion order, monotonic
  text TEXT,
  tags TEXT                   -- JSON string (arbitrary key/values)
);

CREATE INDEX IF NOT EXISTS docs_seg_off ON docs(segment_id, row_offset);
CREATE INDEX IF NOT EXISTS docs_seq ON docs(seq);

-- Bitmaps: serialized Roaring as TEXT (BMFilter.serialize() latin-1 string)
CREATE TABLE IF NOT EXISTS bitmaps(
  name TEXT PRIMARY KEY,     -- e.g. 'tag:user=u123', 'tag:tenant=acme', 'tombstones'
  data TEXT NOT NULL,        -- BMFilter.serialize()
  last_used INTEGER          -- unix time (for cache eviction heuristics)
);

-- Optional DF/catalogs for terms
CREATE TABLE IF NOT EXISTS stats(
  key TEXT PRIMARY KEY,      -- e.g. 'term_df:12345'
  value INTEGER
);

-- Generic key/value for custom metadata (JSON as TEXT)
CREATE TABLE IF NOT EXISTS kv(
  key TEXT PRIMARY KEY,
  value TEXT
);
```

**Conventions**

- Bitmap keys: `tag:<k>=<v>`, `term:<tid>`, special `tombstones`.
- Keep hot bitmaps cached in RAM. Update `last_used` on read.

---

## 4) CSR storage

Save CSR arrays per segment:

- `indptr.npy : int64  | len=N+1`
- `indices.npy: int32  | len=nnz`
- `data.npy   : float32| len=nnz`
- `row_ids.npy: bytes/utf8 or int64 | len=N` (maps row→doc_id)

Open for serving with:

```python
indptr  = np.load("indptr.npy",  mmap_mode="r")
indices = np.load("indices.npy", mmap_mode="r")
data    = np.load("data.npy",    mmap_mode="r")
X = csr_matrix((data, indices, indptr), shape=(N, D))   # zero-copy views
```

- Keep CSR arrays on the filesystem (or fsspec); do not store them in the SQL DB. This remains true even if the metadata backend later uses Postgres.

---

## 5) Query algorithm (default)

Input: `q_terms=[(term_id, weight)]`, `filters={k:v}`, `k`, `budget`.

1. **Base filter**

   - Intersect tag bitmaps from `filters` → `B`.
   - Subtract tombstones bitmap.

2. **Adaptive term gating**

   - Rank query terms by `weight × idf`.
   - MUST = greedy AND until `|B ∩ MUST| ≤ budget` or min_must reached.
   - SHOULD = top-N remaining terms (OR).
   - Candidates `C = (B ∩ MUST) ∩ OR(SHOULD)`.

3. **Score**

   - Group `C` by `segment_id`.
   - For each segment: slice rows `X_seg[offsets]`; compute exact scores `q_csr @ X_seg[offsets].T`.
   - Merge top-K across segments.

4. **Return**

   - Attach `text`, `tags` from `docs`.

**Defaults**
`budget=50_000`, `min_must=2`, `should_cap=100`, drop top 1–2% highest-DF terms.

Profiles and knobs

- paraphrase_hp (high precision):
  - min_must: 3–5, should_cap: 16–32, df_drop_top_percent: 3–5, budget: 5k–10k
  - Optional reranker over top 50–100 candidates.
- rag (high recall):
  - min_must: 0–1, should_cap: 100–300, df_drop_top_percent: 0.5–1, budget: 50k–150k
- recent (recency-first):
  - Candidates = base bitmap (after filters/tombstones); rank by docs.seq desc; ignore dot scores for ordering.

Exclusions

- Exclude specific doc_ids by subtracting a temporary bitmap of those ids from the base bitmap prior to candidate generation.

---

## 6) Extensibility hooks (function-first)

Hooks are plain callables you wire together. Pass a function directly or a dotted path; classes with **call** are also accepted and will be instantiated. Keep behavior pure; put state in ABC-backed components (see note below).

SciPy (sparse CSR), PyRoaring (BitMap), and SPLADE encoder are required runtime dependencies; no fallbacks.

### Function shapes (type aliases)

```python
from typing import Callable, Iterable, Tuple, Dict, Any, List, Optional

# Shapes are for documentation/type-checking; no Protocols required.
FilterFn = Callable[
    [Iterable[Tuple[int, float]], Optional[Dict[str, str]], Callable[[str], "Roaring"],
     Callable[[int], int], "Roaring", Optional[Iterable[str]], Dict[str, Any]],
    Tuple[List[int], List[int]]  # (must_term_ids, should_term_ids)
]

CandidateSupplierFn = Callable[
    [Iterable[int], Iterable[int], "Roaring", Callable[[str], "Roaring"], int],
    "Roaring"
]

ScoreFn = Callable[
    ["csr_matrix", Dict[str, Any], Iterable[int]],
    List[Tuple[int, float]]  # [(row_offset, score)]
]

RankMergeFn = Callable[
    [Dict[str, List[Tuple[int, float]]], int],
    List[Tuple[str, int, float]]  # [(seg_id, row_offset, score)]
]

EvictFn = Callable[[Dict[str, Any], int], List[str]]
```

### Default implementations (functions)

- **filter_policy_default**: greedy MUST growth by measured Δcardinality; SHOULD = next top-weight terms.
- **candidate_supplier_default**: `C = base ∩ AND(MUST) ∩ OR(SHOULD)` using Roaring fast ops.
- **score_csr_slice**: assume q_csr is a 1×D SciPy CSR; build `X_top = X[offsets]`, `scores = (q_csr @ X_top.T).A1`.
- **score_accumulator**: overlap accumulation over postings for tiny `|C|`.
- **rank_merge_heap**: k-way heap merge of per-segment results.
- **evict_lru**: evict least-recently-used bitmaps from RAM.
- **filter_policy_paraphrase_hp**: greedy MUST growth tuned for high precision; small SHOULD set.
- **filter_policy_rag**: minimal MUST; larger SHOULD for high recall.
- **filter_policy_recent**: bypass term gating; use base bitmap only.
- **candidate_supplier_recent**: returns base (minus tombstones/exclusions), capped by budget.
- **score_hook_noop**: no-op scorer (score=0.0), for recent-mode ranking-by-seq.
- **rank_merge_recent**: sort candidates by docs.seq desc across segments.

### Wiring

```python
from recollex.hooks import (
    filter_policy_default,
    candidate_supplier_default,
    score_csr_slice,
    rank_merge_heap,
    evict_lru,
)
from recollex.utils import resolve_hooks

# Code-first config (no YAML)
hook_specs = {
    "filter_policy": filter_policy_default,            # function
    "candidate_supplier": candidate_supplier_default,  # function
    "score_hook": score_csr_slice,                     # function
    "rank_merge": rank_merge_heap,                     # function
    "evict": evict_lru,                                # function
    # You can also put classes/instances or dotted paths here if desired.
    # "score_hook": "recollex.hooks.score_csr_slice",
    # "custom": MyCallableClass,
    # "custom2": MyCallableInstance,
}

# Optional per-hook ctor kwargs for class specs
ctor_kwargs = {
    # "custom": {"arg": 123},
}

hooks = resolve_hooks(hook_specs, ctor_kwargs)

class Recollex:
    def __init__(self):
        self.filter_fn = hooks["filter_policy"]
        self.candidate_supplier = hooks["candidate_supplier"]
        self.score_fn = hooks["score_hook"]
        self.rank_merge = hooks["rank_merge"]
        self.evict_fn = hooks["evict"]
```

**Private extensions**
Ship proprietary hooks as functions (or classes with **call**) in a private wheel and reference via dotted path:

```yaml
hooks:
  filter_policy: "recollector_pro.belief.filter_policy_v2"
  score_hook: "recollector_pro.scoring.coref_aware"
```

### Stateful components use ABCs

Use abstract base classes for lifecycle/state:

- MetadataStore (docs/bitmaps/stats/kv; transactions, caches)
- SegmentReader (CSR arrays, row lookups)
- Encoder (SPLADE wrapper; dims; close)

See recollex/abcs.py for the ABC definitions of MetadataStore and SegmentReader.

These hold state and can be dataclasses. Behavior remains pure and injected via functions.

See also: docs/code_style.md for the project’s code style and philosophy.

---

## 7) Config (YAML) [illustrative]

```yaml
index_path: ./recollex
# dims derives from the encoder tokenizer; do not configure

encoder:
  model: "seerware/Splade_PP_en_v2"
  # ONNX Runtime is the default; you usually don't need to configure a backend flag
  pooling: "max" # "max" or "sum" (must match for docs/queries)

runtime:
  profile: "rag" # "paraphrase_hp" | "rag" | "recent"
  budget: 50000
  min_must: 1
  should_cap: 100
  df_drop_top_percent: 1
  exclude_doc_ids: [] # optional per-call parameter takes precedence

hooks:
  filter_policy: "recollex.hooks.filter_policy_default"
  candidate_supplier: "recollex.hooks.candidate_supplier_default"
  score_hook: "recollex.hooks.score_csr_slice"
  rank_merge: "recollex.hooks.rank_merge_heap"
  evict: "recollex.hooks.evict_lru"

reranker:
  enabled: false
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_m: 0 # number of candidates to rerank (0 disables)

cache:
  bitmap_ram_mb: 512
  preload_bitmaps:
    - "tag:tenant=acme"
    - "tag:type=query"
```

---

## 8) Build/append/seal

**Append** new docs to an active in-RAM buffer. When buffer hits N docs:

0. Assign a monotonically increasing seq to each new doc at add-time (global insertion order).
1. Write `indptr/indices/data/row_ids` to `segments/seg_XXX/`.
2. `fsync`.
3. Write `manifest.tmp` then atomic `rename → manifest.json`.
4. Upsert docs (including seq), bitmaps, and stats/kv inside a single SQL transaction.

**Tombstone**
Add `doc_id` to `bitmaps['tombstones']` (replace row with new BLOB). Queries subtract it.

**Compaction**
Pick segments with `dead_ratio > threshold`. Rebuild a fresh segment with live rows. Swap manifest.

---

## 9) Concurrency and crash safety

- Readers open segment `.npy` with `mmap_mode="r"`.
- Only the indexer writes. Use SQLite transactions for `docs/bitmaps`.
- Manifest swap is atomic `rename`. Never edit in place.
- Bitmap updates: write a **new** BLOB row, then `UPDATE` pointer in one transaction.
- Metadata backend is SQLite by default. If you later adopt SQLAlchemy/Postgres, keep CSR segments as files; only docs/bitmaps/stats/kv move to the new backend.

---

## 9.1) Backends (metadata)

- Default: SQLite (direct). Tables: docs, bitmaps, stats, kv.
- Optional (future): SQLAlchemy Core backend for SQLite/Postgres using the same schema.
- Keep CSR segments as files (local FS or fsspec). Do not store CSR in the DB.
- Optional adapters (e.g., Etcher) can implement the same metadata interfaces without changing the core engine.

---

## 10) Performance targets (defaults)

- nnz/doc ≈ 200, `float32` → \~1.6 KB/doc for CSR.
- `budget=50k` keeps slice small.
- p95 retrieve+score (k=100) on RAM-resident segments: 5–20 ms.
- Very small candidate sets: `score_accumulator` beats slicing.
- SPLADE sparse dot scores differ from dense cosine/IP; avoid fixed absolute cutoffs unless calibrated. Profiles should control gating instead.

---

## 11) Minimal public API

```python
engine = Recollex.open("./recollex")

engine.add_many(iter_docs)
engine.remove(doc_ids)

# Text path (black-box)
results = engine.search(
  "postgres connection pool",
  k=50,
  profile="rag",                           # "paraphrase_hp" | "rag" | "recent"
  exclude_doc_ids=[],                      # optional exclusions
)
# Advanced: explicit sparse query terms
results2 = engine.search_terms(
  q_terms=[(tid, wt), ...],
  k=50,
  profile="rag",
  filters={"tenant":"acme","user":"u123"}, # or tags_* scope
)
# -> [{"doc_id":..., "score":..., "segment_id":"seg_000","row_offset":1234, "tags":{...}, "text":"...", "seq": 123456}]

# Convenience for recency-first:
recent = engine.last(filters={"tenant":"acme","user":"u123"}, k=50)
# equivalent to search(..., text="", profile="recent")
```

---

## 12) CLI (suggested)
Only prefetch and clean are provided out of the box; the rest are illustrative examples.

```
recollex init PATH --dims D
recollex add PATH docs.jsonl
recollex seal PATH
recollex tombstone PATH ids.txt
recollex search PATH --query "..." --filters tenant=acme user=u123 --k 20
recollex compact PATH --threshold 0.2
```

---

## 13) Testing

- **Correctness**: for a small corpus, compare scores to brute-force dense dot over densified rows.
- **Filter logic**: unit tests for bitmap Δcardinality and MUST/SHOULD policy.
- **Crash safety**: kill during seal; verify manifest+segments reopen.
- **Windows**: tests for `np.load(..., mmap_mode="r")` and `rename()` swaps.

---

## 14) Optional modules

- `hooks/filter_policy_default.py`
- `hooks/score_csr_slice.py`
- `hooks/score_accumulator.py` (Numba/Cython variant)
- `io/bitmap_sqlite.py` (encode/decode Roaring BLOBs)
- `io/segments.py` (open/close segments, maps)
- `eval/bench.py` (latency and recall harness)
- `hooks/filter_policy_paraphrase_hp.py`
- `hooks/filter_policy_rag.py`
- `hooks/filter_policy_recent.py`
- `hooks/candidate_supplier_recent.py`
- `hooks/score_hook_noop.py`
- `hooks/rank_merge_recent.py`
- `encoder/splade_pp_v2.py` # SPLADE wrapper (Torch/ONNX, pooling)

---

## 15) What stays generic

- `tags` are unopinionated. they could be any unique identifier
- Hooks are thin and documented. Your private belief/coref/scoring lives behind them.

This is the full surface new developers need: storage, query path, schemas, and hook points to extend behavior without touching the core.
