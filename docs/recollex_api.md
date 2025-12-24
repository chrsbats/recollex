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
  - rx.search(text, k=50, all_of_tags=None, one_of_tags=None, none_of_tags=None,
              profile="rag", exclude_doc_ids=None, override_knobs=None,
              min_score=None, project=None) -> List[result]
  - rx.search([text, ...], ...) -> List[List[result]]  # same order as inputs
  - rx.last(k=50, project=None) -> List[result]  # recency shortcut
- Remove:
  - rx.remove(id | [ids]) -> None  # accepts int or str; non-numeric values are ignored
  - rx.remove_by(all_of_tags=None, one_of_tags=None, none_of_tags=None,
                 dry_run=False, project=None) -> int
    - Removes all docs matching the provided scope. Returns the count of removed docs.
    - dry_run=True returns the count without deleting.

Result object (dict)
- doc_id: str
- segment_id: str
- row_offset: int
- score: float  # 0.0 for profile="recent"
- seq: int | None
- text: Optional[str]
- tags: list[str]              # canonical tag strings, e.g. ["project:local","doc_key:doc.123"]
- tags_list: list[str]         # alias for tags
- tags_dict: dict[str, str]    # parsed view from "k:v" tags, e.g. {"project":"local","doc_key":"doc.123"}

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

7) Project/tenant scoping

Project is a very common tag used for multi-tenant or per-project isolation. It is a convenience over a structured tag:

- project="local" is equivalent to scoping by the tag {"project": "local"}.

Examples:

```python
# Index docs with a project tag
rx.add("Doc A", tags={"project": "local", "doc_key": "doc.1"}, timestamp=int(time.time()))
rx.add("Doc B", tags={"project": "remote", "doc_key": "doc.2"}, timestamp=int(time.time())+1)

# Search only within project="local"
hits_local = rx.search("Doc", project="local", k=10)

# Remove all docs for a given project
removed = rx.remove_by(project="local")

# Recent within a project
recent_local = rx.last(project="local", k=5)
```

Docs that use list-style tags like ["project:local", ...] are still supported, but the project= parameter is defined in terms of structured dict tags {"project": "..."}.

8) Most recent (recency‑first), optionally scoped
```python
recent = rx.search("", profile="recent", k=5)
recent_scoped = rx.search("", profile="recent", all_of_tags=["tenant:acme"], k=5)
# Shortcut:
recent2 = rx.last(k=5)
recent3 = rx.search("", profile="recent", all_of_tags=[("tenant", "acme")], k=5)
```

9) Batch search
```python
batches = rx.search(["redis", "postgres"], all_of_tags=["tenant:acme"], k=5)
# batches[0] -> results for "redis"; batches[1] -> results for "postgres"
```

10) Exclude specific doc_ids
```python
hits = rx.search("db", all_of_tags=["tenant:acme"], exclude_doc_ids=[str(did)], k=10)
# accepts int or str; non-numeric values are ignored.
```

11) Remove docs
```python
rx.remove(did)
rx.remove([did1, did2, did3])
# Remove by scope (tags)
n = rx.remove_by(all_of_tags=["tenant:acme"])  # remove all docs for tenant:acme
# You can also use structured tags:
m = rx.remove_by(all_of_tags=[("tenant", "acme")], dry_run=True)  # count only, no delete
```

For structured access to tags (e.g., project, doc_key), prefer tags_dict:

```python
for hit in hits:
    project = hit["tags_dict"].get("project")
    doc_key = hit["tags_dict"].get("doc_key")
```

Recipes

- “Recent, filtered by score threshold”: one call using min_score (keeps only docs with score >= threshold, ordered by recency)
```python
top_recent_scored = rx.search("redis", all_of_tags=["tenant:acme"], profile="recent", min_score=0.2, k=20)
```

Notes
- Smart defaults: no need to configure caches, dims, or providers.
- Empty text:
  - profile="rag" → empty results.
  - profile="recent" → most recent (optionally scoped by tags).
- Tags:
  - Tags are small labels, internally stored as strings like "key:value".
  - Input:
    - Preferred (structured): dict `{"project": "local", "doc_key": "doc.123"}`.
      - These become bitmaps like `tag:project=local`, `tag:doc_key=doc.123`.
    - Also allowed (flat): list of strings `["project:local", "tenant:acme"]`.
      - These become bitmaps like `tag:project:local`, `tag:tenant:acme`.
    - `add_many`: dict tags `{"tenant":"acme","topic":"db"}` use the same `tag:key=value` form.
  - Query (scoping) examples:
    - Flat string tags:
      ```python
      rx.search("q", all_of_tags=["tenant:acme", "topic:db"])
      rx.search("q", one_of_tags=["tenant:acme", "tenant:beta"])
      rx.search("q", none_of_tags=["topic:internal"])
      ```
    - Structured key/value tags (tuple or single-entry dict):
      ```python
      rx.search("q", all_of_tags=[("tenant", "acme"), {"topic": "db"}])
      rx.search("q", one_of_tags=[("tenant", "acme"), ("tenant", "beta")])
      rx.search("q", none_of_tags=[{"topic": "internal"}])
      ```
  - Output:
    - `tags`: `list[str]` – canonical tag strings, e.g. `["project:local","doc_key:doc.123"]`.
    - `tags_list`: `list[str]` – alias for `tags`.
    - `tags_dict`: `dict[str,str]` – parsed from `"k:v"` entries in `tags/tags_list`.
  - Special: `"everything"` inside a tag list means “no restriction” for that list.
- Exclusions: exclude_doc_ids accepts ints or strings; only numeric ids affect results; non-numeric values are ignored.
- doc_id typing: search results expose doc_id as str; add() returns int ids; remove() accepts int or str (non-numeric values are ignored).
- k defaults to 50; increase if you plan client‑side score thresholds or reordering.
- Model cache: first encode (or recollex-prefetch) downloads the model under ./models/<name>/; precision auto‑selected (override with --quant).
- Advanced: search_terms(q_terms=[(tid, wt), ...]) exists but is optional for most users.
- Advanced: override_knobs (for search/search_terms) allows tuning filtering/gating knobs: min_must, should_cap, budget, df_drop_top_percent.
- min_score (optional float): for score profiles, filters out results with score < min_score. For profile="recent" with a non-empty query, keeps only docs with score >= min_score but still orders by recency (ignored if the query is empty).
