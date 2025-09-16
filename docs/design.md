# Recollex — Developer Guide

Sparse, filterable, RAM-friendly indexer for SPLADE-style vectors on Windows/Linux/macOS.

## 1) Mental model

* Store vectors as **CSR** triplets on disk. Slice rows. Dot with query.
* Store filters as **Roaring bitmaps** inside **SQLite** BLOBs. Intersect fast.
* Append in **segments**. Delete via **tombstone bitmap**. Compact later.
* Everything is local. No DB server. Optional ANN is out of scope here.

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

```json
{
  "version": 1,
  "dims": 250000,
  "segments": [
    {"name":"seg_000","rows":[0,100000]},
    {"name":"seg_001","rows":[100000,150000]}
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
  text TEXT,
  tags TEXT                  -- JSON string (arbitrary key/values)
);

CREATE INDEX IF NOT EXISTS docs_seg_off ON docs(segment_id, row_offset);

-- Bitmaps: serialized Roaring as BLOB
CREATE TABLE IF NOT EXISTS bitmaps(
  name TEXT PRIMARY KEY,     -- e.g. 'tag:user=u123', 'tag:tenant=acme', 'tombstones'
  data BLOB NOT NULL,
  last_used INTEGER          -- unix time (for cache eviction heuristics)
);

-- Optional DF/catalogs for terms
CREATE TABLE IF NOT EXISTS stats(
  key TEXT PRIMARY KEY,      -- e.g. 'term_df:12345'
  value INTEGER
);
```

**Conventions**

* Bitmap keys: `tag:<k>=<v>`, `term:<tid>`, special `tombstones`.
* Keep hot bitmaps cached in RAM. Update `last_used` on read.

---

## 4) CSR storage

Save CSR arrays per segment:

* `indptr.npy : int64  | len=N+1`
* `indices.npy: int32  | len=nnz`
* `data.npy   : float32| len=nnz`
* `row_ids.npy: bytes/utf8 or int64 | len=N`  (maps row→doc\_id)

Open for serving with:

```python
indptr  = np.load("indptr.npy",  mmap_mode="r")
indices = np.load("indices.npy", mmap_mode="r")
data    = np.load("data.npy",    mmap_mode="r")
X = csr_matrix((data, indices, indptr), shape=(N, D))   # zero-copy views
```

---

## 5) Query algorithm (default)

Input: `q_terms=[(term_id, weight)]`, `filters={k:v}`, `k`, `budget`.

1. **Base filter**

   * Intersect tag bitmaps from `filters` → `B`.
   * Subtract tombstones bitmap.

2. **Adaptive term gating**

   * Rank query terms by `weight × idf`.
   * MUST = greedy AND until `|B ∩ MUST| ≤ budget` or min\_must reached.
   * SHOULD = top-N remaining terms (OR).
   * Candidates `C = (B ∩ MUST) ∩ OR(SHOULD)`.

3. **Score**

   * Group `C` by `segment_id`.
   * For each segment: slice rows `X_seg[offsets]`; compute exact scores `q_csr @ X_seg[offsets].T`.
   * Merge top-K across segments.

4. **Return**

   * Attach `text`, `tags` from `docs`.

**Defaults**
`budget=50_000`, `min_must=2`, `should_cap=100`, drop top 1–2% highest-DF terms.

---

## 6) Extensibility hooks

Use simple **function hooks**. Accept callables or dotted paths. Keep the core OSS, your advanced logic private.

### Hook registry

```python
from typing import Protocol, Iterable, Tuple, Dict, Any, Callable, List
import importlib

def load_func(spec_or_fn):
    if callable(spec_or_fn): return spec_or_fn
    mod, _, name = spec_or_fn.rpartition('.')
    return getattr(importlib.import_module(mod), name)

# ---------- Protocols ----------
class FilterPolicy(Protocol):
    def __call__(
        self,
        q_terms: Iterable[Tuple[int, float]],
        base_bitmap: "Roaring",
        df_lookup: Callable[[int], int],
        budget: int,
        min_must: int,
        should_cap: int
    ) -> Dict[str, Any]:  # {"must":[int], "should":[int], "exclude":[int]}
        ...

class CandidateSupplier(Protocol):
    def __call__(
        self,
        must_terms: Iterable[int],
        should_terms: Iterable[int],
        base_bitmap: "Roaring",
        bitmap_get: Callable[[str], "Roaring"],
        budget: int
    ) -> "Roaring":  # candidate doc_id set
        ...

class ScoreHook(Protocol):
    def __call__(
        self,
        q_terms: Iterable[Tuple[int, float]],
        seg_view: "SegmentCSR",                  # exposes CSR arrays + row lookup
        doc_offsets: Iterable[int],              # row indices in this segment
        l2_normalized: bool = True
    ) -> List[Tuple[int, float]]:                # [(row_offset, score)]
        ...

class RankMergeHook(Protocol):
    def __call__(
        self,
        per_segment_results: Dict[str, List[Tuple[int, float]]],  # seg_id -> offsets/scores
        k: int
    ) -> List[Tuple[str, int, float]]:           # [(seg_id, row_offset, score)]
        ...

class EvictHook(Protocol):
    def __call__(self, cache_stats: Dict[str, Any], bytes_to_free: int) -> List[str]: ...
```

### Default implementations

* **filter\_policy\_default**: greedy MUST growth by measured Δcardinality; SHOULD = next top-weight terms.
* **candidate\_supplier\_default**: `C = base ∩ AND(MUST) ∩ OR(SHOULD)` using Roaring fast ops.
* **score\_csr\_slice**: build `X_top = X[offsets]`, `scores = (q_csr @ X_top.T).A1`.
* **score\_accumulator**: overlap accumulation over postings for tiny `|C|`.
* **rank\_merge\_heap**: k-way heap merge of per-segment results.
* **evict\_lru**: evict least-recently-used bitmaps from RAM.

### Wiring

```python
class Recollex:
    def __init__(self, cfg):
        self.filter_policy = load_func(cfg["hooks"]["filter_policy"])
        self.candidate_supplier = load_func(cfg["hooks"]["candidate_supplier"])
        self.score_hook = load_func(cfg["hooks"]["score_hook"])
        self.rank_merge = load_func(cfg["hooks"]["rank_merge"])
        self.evict = load_func(cfg["hooks"]["evict"])
```

**Private extensions**
Ship proprietary hooks as a private wheel and reference via dotted path:

```yaml
hooks:
  filter_policy: "recollector_pro.belief.filter_policy_v2"
  score_hook:    "recollector_pro.scoring.coref_aware"
```

---

## 7) Config (YAML)

```yaml
index_path: ./recollex
dims: 250000

runtime:
  budget: 50000
  min_must: 2
  should_cap: 100
  df_drop_top_percent: 1

hooks:
  filter_policy: "recollex.hooks.filter_policy_default"
  candidate_supplier: "recollex.hooks.candidate_supplier_default"
  score_hook: "recollex.hooks.score_csr_slice"
  rank_merge: "recollex.hooks.rank_merge_heap"
  evict: "recollex.hooks.evict_lru"

cache:
  bitmap_ram_mb: 512
  preload_bitmaps:
    - "tag:tenant=acme"
    - "tag:type=query"
```

---

## 8) Build/append/seal

**Append** new docs to an active in-RAM buffer. When buffer hits N docs:

1. Write `indptr/indices/data/row_ids` to `segments/seg_XXX/`.
2. `fsync`.
3. Write `manifest.tmp` then atomic `rename → manifest.json`.
4. Upsert bitmaps in SQLite inside a transaction.

**Tombstone**
Add `doc_id` to `bitmaps['tombstones']` (replace row with new BLOB). Queries subtract it.

**Compaction**
Pick segments with `dead_ratio > threshold`. Rebuild a fresh segment with live rows. Swap manifest.

---

## 9) Concurrency and crash safety

* Readers open segment `.npy` with `mmap_mode="r"`.
* Only the indexer writes. Use SQLite transactions for `docs/bitmaps`.
* Manifest swap is atomic `rename`. Never edit in place.
* Bitmap updates: write a **new** BLOB row, then `UPDATE` pointer in one transaction.

---

## 10) Performance targets (defaults)

* nnz/doc ≈ 200, `float32` → \~1.6 KB/doc for CSR.
* `budget=50k` keeps slice small.
* p95 retrieve+score (k=100) on RAM-resident segments: 5–20 ms.
* Very small candidate sets: `score_accumulator` beats slicing.

---

## 11) Minimal public API

```python
engine = Recollex.open("./recollex", cfg)

engine.add_many(iter_docs)         # build-time only; not for serving
engine.tombstone(doc_ids)          # logical delete
engine.compact(threshold=0.2)      # offline maintenance

results = engine.search(
  q_terms=[(tid, wt), ...],
  filters={"tenant":"acme","user":"u123"},
  k=50
)
# -> [{"doc_id":..., "score":..., "segment_id":"seg_000","row_offset":1234, "tags":{...}, "text":"..."}]
```

---

## 12) CLI (suggested)

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

* **Correctness**: for a small corpus, compare scores to brute-force dense dot over densified rows.
* **Filter logic**: unit tests for bitmap Δcardinality and MUST/SHOULD policy.
* **Crash safety**: kill during seal; verify manifest+segments reopen.
* **Windows**: tests for `np.load(..., mmap_mode="r")` and `rename()` swaps.

---

## 14) Optional modules (OSS examples)

* `hooks/filter_policy_default.py`
* `hooks/score_csr_slice.py`
* `hooks/score_accumulator.py`  (Numba/Cython variant)
* `io/bitmap_sqlite.py`         (encode/decode Roaring BLOBs)
* `io/segments.py`              (open/close segments, maps)
* `eval/bench.py`               (latency and recall harness)

---

## 15) What stays generic in OSS

* `tags` are unopinionated. they could be any unique identifier
* Hooks are thin and documented. Your private belief/coref/scoring lives behind them.

This is the full surface new developers need: storage, query path, schemas, and hook points to extend behavior without touching the core.
