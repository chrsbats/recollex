# Recollex

A minimal, local sparse indexer for SPLADE-style vectors with SQLite metadata and Roaring bitmaps for fast filtering.

[![PyPI](https://img.shields.io/pypi/v/recollex.svg)](https://pypi.org/project/recollex/)
[![Python](https://img.shields.io/pypi/pyversions/recollex.svg)](https://pypi.org/project/recollex/)
[![Build](https://github.com/chrsbats/recollex/actions/workflows/ci.yml/badge.svg)](https://github.com/chrsbats/recollex/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-90%25%2B-brightgreen)](https://github.com/chrsbats/recollex/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What it does

- Creates an embedding for documents using an opensource version of Splade++
- Stores these document vectors as sparse array in filesystem segments.
- Stores filters (tags, term postings, tombstones) as Roaring bitmaps in SQLite.
- Lets you add docs (text + tags) and run queries (text + tags) with exact sparse dot-product scoring.
- Supports recency-first ranking profile.
- The Splade++ model is available in int8 quantized ONNX format for speed on CPU (no torch install required). An fp16 version built for GPUs is also available as well as a non-quantized fp32 version.

---

## Installation

Choose the package that matches your hardware.
Each build includes its own optimized ONNX Runtime provider — install **only one** variant per environment.

| Platform                    | Command                              |
| --------------------------- | ------------------------------------ |
| CPU (default)               | `pip install recollex`               |
| NVIDIA CUDA                 | `pip install recollex-gpu[cuda]`     |
| AMD ROCm                    | `pip install recollex-gpu[rocm]`     |
| Windows DirectML            | `pip install recollex-gpu[directml]` |
| Apple Silicon (macOS arm64) | `pip install recollex-gpu[silicon]`  |

> Install **only one** ONNX Runtime variant per environment.

---

## Development Setup

### Using Pixi (recommended)

#### Install (pick one)

```bash
# Install the default CPU environment
pixi install

# For development install choose
pixi install -e dev

# For a hardware-acceleration use the feature flags
pixi install -e cuda       # NVIDIA
pixi install -e directml   # Windows NVIDIA / AMD
```

**Note:** Silicon and ROCm require PyPI installs. They are not available via the shared Pixi lock due to versioning conflicts.

#### Scripts

```bash
# Utility tasks
pixi run prefetch                # auto-detects model precision (int8/fp16/fp32)
pixi run test                    # run tests
pixi run clean                   # remove downloaded models

# Manual quantization control
pixi run prefetch --quant=fp16
pixi run clean --quant=fp16
```

---

### Using Pip (for local development or CI)

```bash
# Install the editable package for your variant
pip install -e ".[cpu]"   # or [gpu], [rocm], [directml], [silicon]

# Run the CLI utilities
recollex-prefetch                # auto precision
recollex-prefetch --quant=fp16   # manual precision
recollex-clean
recollex-clean --quant=fp16
```

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
  - index.add accepts a `Sequence[str]`. Each string becomes a tag bitmap named `tag:<string>`.
  - Pass the same strings to `all_of_tags`, `one_of_tags`, or `none_of_tags` when searching.
- Structured tags (key=value):
  - If you need `tag:k=v` style, use `add_many` and pass tags as a dict for each doc (e.g., `{"tenant":"acme","topic":"db"}`). The engine will populate bitmaps `tag:tenant=acme`, `tag:topic=db`.
- Model download:
  - The first encode or recollex-prefetch will download the SPLADE ONNX model into `./models/seerware__Splade_PP_en_v2/`. Precision is auto-selected (int8/fp16/fp32) based on your onnxruntime install/providers; override with --quant.
  - Use recollex-clean to remove the model cache (or a single precision with --quant).
- Storage layout:
  - Index data lives under `./recollex_index/` (manifest.json + segments/\*). Metadata lives in `meta.sqlite`.
- Profiles:
  - `profile="rag"` (default) favors recall; `profile="recent"` ranks by seq (timestamp) descending.
- Thread-safety:
  - The engine keeps a simple in-memory CSR cache per segment. Avoid sharing a single Recollex instance across threads without external synchronization.
- Splade++ model weights are quantized version of those provided by https://huggingface.co/prithivida/Splade_PP_en_v2 under Apache 2.0 license.
- If you like this model it's also worth following **donkeystereotype** aka **Prithiviraj Damodaran** on github: https://github.com/PrithivirajDamodaran

For internal design and contributor guidelines, see docs/design.md and docs/code_style.md.
For the public API, see docs/recollex_api.md. Tuning notes: docs/recollex_tuning.md.
Batch note: index.search([...], k=...) returns List[List[result]] (one result list per input text).
