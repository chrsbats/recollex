# Recollex

A minimal, local sparse indexer for SPLADE-style vectors with SQLite metadata and Roaring bitmaps for fast filtering.

What it does

- Stores document vectors as CSR in filesystem segments.
- Stores filters (tags, term postings, tombstones) as Roaring bitmaps in SQLite.
- Lets you add docs (text + tags) and run queries (text + tags) with exact sparse dot-product scoring.
- Supports recency-first ranking profile.
- Uses an opensource, quantized version of splade++ model via the Onnx runtime (no torch install required) (a non-quantized version also available)

## Install

- CPU (default): pip install recollex
- CUDA (NVIDIA): pip install recollex-cuda
- ROCm (AMD): pip install recollex-rocm
- DirectML (Windows): pip install recollex-directml
- Apple Silicon (macOS arm64): pip install recollex-silicon
- Note: install only one ONNX Runtime provider wheel per environment.

## Development install

- Using Pixi:
  - pixi install
  - pixi run prefetch (auto-selects ONNX precision: int8/fp16/fp32 and downloads to local directory)
  - pixi run clean (deletes the model from local directory)
  - pixi run prefetch --quant=fp16 (manually select the quant size)
  - pixi run clean --quant=fp16 (manually select the quant size)
- Using Pip and CLI:
  - pip install -e ".[cpu]" # or [gpu], [rocm], [directml], [silicon]
  - recollex-prefetch (auto)
  - recollex-prefetch --quant=fp16 (or int8, fp32)
  - recollex-clean
  - recollex-clean --quant=fp16

## Quickstart (really simple)

- Give it a directory. It will create a new index there (or load an existing one). Add docs and search.

```python
import time
from recollex import Recollex

# 1) Open or create an index
index = Recollex("./recollex_index")  # Recollex(path) auto-creates the directory (or loads existing).

# 2) Add a couple docs (text + tags); timestamp is any monotonically increasing int
d1 = index.add("Redis quickstart", tags=["tenant:acme", "topic:db"], timestamp=int(time.time()))
d2 = index.add("Postgres tips and tricks", tags=["tenant:acme", "topic:db"], timestamp=int(time.time()) + 1)

# 3) Search
results = index.search("postgres connection pool", k=3)
for r in results:
    print(r["doc_id"], round(r["score"], 4), r["tags"])
```

Advanced usage (tags, recency, exclusions)

```python
# Tag-filtered search (all_of/one_of/none_of)
results = index.search(
    "database best practices",
    all_of_tags=["tenant:acme", "topic:db"],   # must have BOTH tags
    none_of_tags=["topic:food"],               # exclude these
    k=5,
    profile="rag",                              # default
)
for r in results:
    print(r["doc_id"], round(r["score"], 4), r["tags"])

# Recency-first (ignores dot scores for ordering)
recent = index.search(
    text="",                     # empty text: no terms
    all_of_tags=["tenant:acme"], # optional scope by tags
    k=5,
    profile="recent",
)
for r in recent:
    print("RECENT:", r["doc_id"], r["seq"], r["tags"])

# Exclude specific doc_ids
excluded = index.search(
    "db",
    all_of_tags=["tenant:acme"],
    exclude_doc_ids=[str(d1)],  # exclude first doc
    k=5,
)
for r in excluded:
    print("NO-EX:", r["doc_id"], round(r["score"], 4), r["tags"])
```

## Notes

- Smart defaults:
  - Recollex(path) auto-creates the directory and SQLite metadata if missing, or loads an existing index if present.
- Tags as strings:
  - index.add accepts a Sequence[str]. Each string becomes a tag bitmap named tag:<string>.
  - Pass the same strings to all_of_tags, one_of_tags, or none_of_tags when searching.
- Structured tags (key=value):
  - If you need tag:k=v style, use add_many and pass tags as a dict for each doc (e.g., {"tenant":"acme","topic":"db"}). The engine will populate bitmaps tag:tenant=acme, tag:topic=db.
- Model download:
  - The first encode or recollex-prefetch will download the SPLADE ONNX model into ./models/seerware\_\_Splade_PP_en_v2/. Precision is auto-selected (int8/fp16/fp32) based on your onnxruntime install/providers; override with --quant.
  - Use recollex-clean to remove the model cache (or a single precision with --quant).
- Storage layout:
  - Index data live under ./recollex_index/ (manifest.json + segments/\*). Metadata are in meta.sqlite.
- Profiles:
  - profile="rag" (default) favors recall; profile="recent" ranks by seq (timestamp) descending.
- Thread-safety:
  - The engine keeps a simple in-memory CSR cache per segment. Avoid sharing a single Recollex instance across threads without external synchronization.
- Splade++ model weights are quantized version of those provided by https://donkeystereotype.com/ (https://huggingface.co/prithivida/Splade_PP_en_v2) under Apache 2.0 license. It's also worth following donkeystereotype aka Prithiviraj Damodaran on github (https://github.com/PrithivirajDamodaran) (he makes a lot of good stuff).
