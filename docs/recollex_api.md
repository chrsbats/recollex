# Recollex API 

What you can do
- Create/open an index directory.
- Add documents (single or batch) with tags and a timestamp.
- Search by text (ranked by score) or by recency.
- Scope queries with tags; exclude specific doc_ids.
- Remove documents.

Import
```python
from recollex import Recollex
```

API at a glance
- Open:
  - rx = Recollex("./index_dir")  # or Recollex.open("./index_dir")
- Add:
  - rx.add(text, tags=None, timestamp=None) -> int
  - rx.add([ (text, tags, timestamp), {"text":..., "tags":[...], "timestamp":...} | {"text":...,"tags":[...],"seq":...}, ... ]) -> List[int]
    - Tuple form must be exactly (text, tags, timestamp).
    - Dict form: pass "timestamp" (preferred) or "seq". If you omit both, the engine assigns a sequence value; pass one if you care about recency ordering.
- Add (advanced, pre-encoded):
  - rx.add_many([{doc_id, indices, data, text?, tags?, seq?}, ...]) -> {"n_docs","nnz"}
- Search:
  - rx.search(text, k=50, all_of_tags=None, one_of_tags=None, none_of_tags=None, profile="rag", exclude_doc_ids=None) -> List[result]
  - rx.search([text, ...], ...) -> List[List[result]]  # same order as inputs
  - rx.last(filters=None, k=50) -> List[result]  # recency shortcut
- Remove:
  - rx.remove(id | [ids]) -> None

Result object (dict)
- doc_id: str
- segment_id: str
- row_offset: int
- score: float  # 0.0 for profile="recent"
- seq: int | None
- text: Optional[str]
- tags: Optional[dict or list]  # matches how you added the doc

Common tasks

1) Create/open an index
```python
rx = Recollex("./my_index")
```

2) Add docs (single)
```python
import time
did = rx.add("Redis quickstart", tags=["tenant:acme", "topic:db"], timestamp=int(time.time()))
```

3) Add docs (batch via add)
```python
items = [
  ("Postgres tips", ["tenant:acme","topic:db"], int(time.time())),
  {"text":"SQLite notes","tags":["tenant:acme","topic:db"],"timestamp":int(time.time())+1},
  {"text":"SQLite notes v2","tags":["tenant:acme","topic:db"],"seq":int(time.time())+2},
]
ids = rx.add(items)
```

4) Add docs (batch via add_many; pre-encoded)
```python
docs = [{
  "doc_id": 101,                  # int or numeric string
  "indices": [2,7,9], "data": [0.3,0.8,0.2],
  "text": "Custom vector doc",
  "tags": {"tenant":"acme","topic":"db"},   # or ["tenant:acme","topic:db"]
  "seq": int(time.time()),                  # optional
}]
rx.add_many(docs)
```

5) Search (top‑k by score; default profile="rag")
```python
hits = rx.search("postgres connection pool", k=5)
```

6) Search within tags
```python
hits = rx.search(
  "database best practices",
  all_of_tags=["tenant:acme", "topic:db"],    # intersection
  one_of_tags=None,                           # union if provided
  none_of_tags=["topic:food"],                # exclusion
  k=10,
)
```

7) Most recent (recency‑first), optionally scoped
```python
recent = rx.search("", profile="recent", k=5)
recent_scoped = rx.search("", profile="recent", all_of_tags=["tenant:acme"], k=5)
# Shortcut:
recent2 = rx.last(k=5)
recent3 = rx.last(filters={"tenant":"acme"}, k=5)  # key=value scope (structured tags)
```

8) Batch search
```python
batches = rx.search(["redis", "postgres"], all_of_tags=["tenant:acme"], k=5)
# batches[0] -> results for "redis"; batches[1] -> results for "postgres"
```

9) Exclude specific doc_ids
```python
hits = rx.search("db", all_of_tags=["tenant:acme"], exclude_doc_ids=[str(did)], k=10)
# Non-numeric values are ignored.
```

10) Remove docs
```python
rx.remove(did)
rx.remove([did1, did2, did3])
```

Recipes

- “Recent, filtered by score threshold”:
  1) Score search within scope, fetch larger k.
  2) Filter client‑side by score, then sort by seq desc.
```python
pool = rx.search("redis", all_of_tags=["tenant:acme"], k=500)
pool = [r for r in pool if r["score"] >= 0.2]
pool.sort(key=lambda r: r["seq"] or 0, reverse=True)
top_recent_scored = pool[:20]
```

Notes
- Smart defaults: no need to configure caches, dims, or providers.
- Empty text:
  - profile="rag" → empty results.
  - profile="recent" → most recent (optionally scoped by tags).
- Tags:
  - add/search: use strings like "tenant:acme".
  - add_many: pass dict for key=value style (becomes tag:tenant=acme).
  - Special: "everything" inside a tag list means “no restriction” for that list.
- Exclusions: exclude_doc_ids affects only numeric ids.
- k defaults to 50; increase if you plan client‑side score thresholds or reordering.
- Model cache: first encode (or recollex-prefetch) downloads the model under ./models/<name>/; precision auto‑selected (override with --quant).
- Advanced: search_terms(q_terms=[(tid, wt), ...]) exists but is optional for most users.
