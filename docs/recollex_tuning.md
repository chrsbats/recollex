# Recollex Tuning (advanced)

Most users should not need this. Defaults are chosen to “just work.”
Use these knobs only if you have clear performance or behavior goals.

Contents
- Profiles and per-call knobs
- k, score thresholds, and result shaping
- Caches and memory limits
- Encoder/model precision and providers
- Metadata store (SQLite) cache
- Concurrency/locking notes
- Practical recipes

Profiles and per-call knobs
- Recollex.search(...) and search_terms(...) accept:
  - profile: "rag" (default), "recent", or "paraphrase_hp"
  - override_knobs: dict with:
    - budget: int candidates to consider downstream (post-filter)
    - min_must: int minimum MUST terms during gating
    - should_cap: int cap on SHOULD terms
    - df_drop_top_percent: float percentage of highest-DF terms to drop (0–100)

Default knobs by profile
- rag (high recall)
  - budget: 150_000
  - min_must: 0
  - should_cap: 200
  - df_drop_top_percent: 0.5
- paraphrase_hp (high precision)
  - budget: 10_000
  - min_must: 3
  - should_cap: 24
  - df_drop_top_percent: 3.0
- recent (recency-first)
  - Ignores scores for ordering; ranks by seq desc.
  - budget, if provided, caps how many “recent” candidates are taken.

Usage examples
```python
# Larger recall window for broad search
hits = rx.search("network timeout", k=100, profile="rag",
                 override_knobs={"budget": 250_000, "should_cap": 300})

# Tighter, precision-leaning search
hits = rx.search("redis retry policy", k=20, profile="paraphrase_hp",
                 override_knobs={"min_must": 4, "budget": 8_000})

# Recent with explicit budget (useful when excluding many ids)
hits = rx.search("", profile="recent", k=50, override_knobs={"budget": 100})
```

Notes
- Exclusions: exclude_doc_ids affects only numeric ids; non-numeric are ignored.
- Tag scoping: all_of_tags (AND), one_of_tags (OR), none_of_tags (NOT).
  - The special "everything" in any tag list is treated as no restriction.

k, score thresholds, and result shaping
- There is no min_score parameter. Fetch a larger k and filter client-side:
```python
pool = rx.search("redis", k=500, profile="rag")
filtered = [r for r in pool if r["score"] >= 0.2]
# Optionally sort filtered by recency
filtered.sort(key=lambda r: r["seq"] or 0, reverse=True)
top = filtered[:50]
```
- Relationship of k vs. budget:
  - Ensure budget >> k so gating produces enough candidates for reordering/thresholding.
  - For “recent then only keep those that match a text,” run a score search to build the pool, then sort by seq.

Caches and memory limits
- Segment reader cache (count-based LRU)
  - Controls reopened NpySegmentReader instances.
  - Constructor arg: seg_cache_max (default 64).
- CSR cache (count + optional RAM cap)
  - Stores per-segment SciPy CSR matrices to avoid rebuilding.
  - Constructor args:
    - csr_cache_max (default 128)
    - csr_ram_limit_bytes (default 512 MiB; set None to disable RAM-based eviction)
  - Eviction is LRU-by-count first, then by RAM limit if set.

Examples
```python
# Memory-constrained environment
rx = Recollex("./idx",
              seg_cache_max=32,
              csr_cache_max=64,
              csr_ram_limit_bytes=256 * 1024 * 1024)

# Generous memory
rx = Recollex("./idx",
              seg_cache_max=128,
              csr_cache_max=256,
              csr_ram_limit_bytes=None)  # count-based only
```

Encoder/model precision and providers
- Default encoder: ONNX SPLADE; precision auto-selected:
  - If an accelerator ORT is installed or providers are available → fp16
  - Otherwise → int8 on CPU
- Override precision with env var (read at runtime):
  - RECOLLEX_ONNX_PRECISION=int8|fp16|fp32
- Prefetch the chosen precision (avoid first-call download):
```bash
recollex-prefetch                  # auto-select precision
recollex-prefetch --quant=fp16     # force fp16
recollex-clean                     # remove all cached model files
recollex-clean --quant=fp16        # remove a single precision
```
- Providers are chosen from what ONNX Runtime exposes (priority: CUDA, ROCm, DirectML, CoreML, CPU). No manual config required.

Metadata store (SQLite) cache
- SQLiteMetadataStore keeps a small in-process LRU of bitmap blobs (TEXT, latin-1).
  - Constructor arg: bitmap_cache_size (default 256)
```python
from recollex.io import SQLiteMetadataStore
store = SQLiteMetadataStore("./idx/meta.sqlite", bitmap_cache_size=512)
rx = Recollex("./idx", store=store)
```

Concurrency and locking
- Manifest is guarded by a cross-platform file lock. It uses fcntl (POSIX) or msvcrt (Windows), with a sidecar .pid fallback.
- Timeout: 30s by default for critical sections (segment write + manifest swap).
- Test/CI override:
  - RECOLLEX_FORCE_PID_LOCK=1 forces the .pid fallback path (useful for deterministic tests).
- You normally don’t need to change this in applications.

Practical recipes

- Increase recall without overwhelming scoring
  - Set profile="rag", raise budget moderately (e.g., 200k), keep k reasonable (e.g., 50–200).
- Tighten precision for short queries
  - profile="paraphrase_hp" or override min_must to 3–5, and keep budget ~5–10k.
- Recent lists under heavy exclusions
  - Use profile="recent" with override_knobs={"budget": k*2 or k*3} to compensate for excluded ids.
- Memory pressure with many segments
  - Reduce csr_cache_max and set csr_ram_limit_bytes; seg_cache_max can also be reduced.
- Batch heavy workloads
  - Prefer batch search: rx.search([...], k=...) to share encoder work across queries.
- Large-k with client-side thresholds
  - Increase k and budget; filter scores in your code; sort by desired key (score or seq) after filtering.

Troubleshooting
- ValueError: Query term id >= target dims
  - Your query contains a term id ≥ manifest.dims (encoder/index mismatch). Ensure you use the same encoder that built the index.
- onnxruntime is not installed
  - Install the appropriate package:
    - CPU: pip install recollex
    - GPU/accelerators: pip install recollex-gpu[cuda|rocm|directml|silicon]
- Empty results on empty query
  - Default profile="rag" returns no results for empty text. Use profile="recent" for recency lists.
