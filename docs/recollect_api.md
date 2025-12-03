# Recollex API (black-box usage)

This guide covers the public surface you need to build and query an index. It intentionally avoids internals and tuning.

If you haven’t installed the package or reviewed the quickstart, see the README.

What you can do
- Create or open an index directory.
- Add documents (single or batch) with tags and a timestamp.
- Search by text (ranked by score).
- Get most recent documents (optionally scoped by tags).
- Exclude specific doc_ids from results.
- Remove documents.

Import
```python
from recollex import Recollex
```

Create/open an index
- Recollex(path) auto-creates the directory (and SQLite metadata) if missing, or loads an existing index.
```python
rx = Recollex("./my_index")        # or Recollex.open("./my_index")
```

Add documents

Single add (simple)
- Assigns an integer doc_id automatically.
- tags: sequence of strings (each becomes a tag like "tenant:acme").
- timestamp: any monotonically increasing int (e.g., int(time.time())).
  - If omitted, the engine assigns a sequence value; provide one if you plan to use recency features.
```python
import time
did = rx.add(
    "Redis quickstart",
    tags=["tenant:acme", "topic:db"],
    timestamp=int(time.time())
)
```

Batch add via add (tuples or dicts)
- Pass a list of items; returns a list of assigned int doc_ids.
- Tuple form: (text, tags, timestamp)
- Dict form: {"text": str, "tags": Sequence[str], "timestamp": int}
```python
items = [
    ("Postgres tips", ["tenant:acme", "topic:db"], int(time.time())),
    {"text": "SQLite notes", "tags": ["tenant:acme", "topic:db"], "timestamp": int(time.time())+1},
]
ids = rx.add(items)
```

Advanced batch add via add_many
- Use when you already have sparse vectors (indices/data) or want structured tags.
- Input is a list of dicts. doc_id must be numeric (int or numeric string).
- Returns {"n_docs": N, "nnz": total_nonzeros}.
Doc schema:
```python
docs = [
  {
    "doc_id": 101,                 # int or numeric string
    "indices": [2, 7, 9],          # term ids (non-negative)
    "data":    [0.3, 0.8, 0.2],    # weights; same length as indices
    "text": "Optional blob",
    "tags": ["tenant:acme", "topic:db"]  # or dict: {"tenant": "acme", "topic": "db"}
    # "seq": int(time.time()),      # optional; if omitted, engine assigns
  },
]
rx.add_many(docs)  # {"n_docs": 1, "nnz": 3}
```
Notes:
- If tags is a dict, entries produce tag bitmaps like tag:tenant=acme.
- If tags is a sequence of strings, each string is used as-is (e.g., "tenant:acme").
- dims and segment selection are automatic; you don’t need to set them.

Search

1) Highest score (default profile="rag")
```python
results = rx.search("postgres connection pool", k=5)
for r in results:
    print(r["doc_id"], round(r["score"], 4), r["tags"])
```

2) Highest score within tag scope
- all_of_tags: intersection (must contain all)
- one_of_tags: union (must contain at least one)
- none_of_tags: exclusion
```python
results = rx.search(
    "database best practices",
    all_of_tags=["tenant:acme", "topic:db"],
    none_of_tags=["topic:food"],
    k=10,
)
```

3) Most recent (recency-first)
- Ignores dot-product scores for ordering; ranks by seq descending.
```python
recent = rx.search("", profile="recent", k=5)
```

3a) Most recent within a tag scope
```python
recent_scoped = rx.search("", profile="recent", all_of_tags=["tenant:acme"], k=5)
```

4) Score thresholding (client-side)
- There’s no min_score parameter. Fetch a larger k and filter in your code.
```python
pool = rx.search("redis", all_of_tags=["tenant:acme"], k=200)
filtered = [r for r in pool if r["score"] >= 0.2]
```

4a) “Recent, filtered by score threshold” (two-step)
- Get candidates by score within your scope, filter, then order by recency.
```python
pool = rx.search("redis", all_of_tags=["tenant:acme"], k=500)
pool = [r for r in pool if r["score"] >= 0.2]
pool.sort(key=lambda r: r["seq"] or 0, reverse=True)
top_recent_scored = pool[:20]
```

Batch search
- Pass a list of texts; return is a list of result-lists in the same order.
```python
batches = rx.search(["redis", "postgres"], all_of_tags=["tenant:acme"], k=5)
# batches[0] -> results for "redis"; batches[1] -> results for "postgres"
```

Exclude specific doc_ids
- Non-numeric values are ignored.
```python
results = rx.search("db", all_of_tags=["tenant:acme"], exclude_doc_ids=[str(did)], k=10)
```

Remove documents
- Remove one or many by id (ints or strings). No-op for unknown ids.
```python
rx.remove(did)
rx.remove([did1, did2, did3])
```

Convenience: recent
```python
recent = rx.last(k=20)                         # global recent
recent_tenant = rx.last(filters={"tenant":"acme"}, k=20)  # structured tag scope (key=value)
```

Result shape
Each result is a dict:
- doc_id: str
- segment_id: str
- row_offset: int
- score: float (0.0 for profile="recent")
- seq: int | None
- text: Optional[str]
- tags: Optional[dict or list], matching how it was added

Notes and behavior
- Smart defaults: you don’t need to configure caches, dims, or providers.
- Empty text:
  - profile="rag" (default): returns empty results.
  - profile="recent": returns most recent documents (optionally scoped by tags).
- Tags:
  - Use strings like "tenant:acme" with add/search helpers.
  - For key=value style (tag:k=v), use add_many with tags as dicts.
  - The special string "everything" in a tag list is treated as no restriction.
- Exclusions: exclude_doc_ids only affect numeric ids.
- k defaults to 50; increase if you plan to apply client-side score thresholds or resorting.
- Model cache: the first encode (or the recollex-prefetch CLI) downloads the model under ./models/<name>/; precision is auto-selected (override with --quant).

Advanced (optional)
- search_terms(q_terms=[(tid, wt), ...], ...): send your own sparse query terms.
- override_knobs, rerank_top_m: reserved for advanced scenarios; safe to ignore in normal usage.
