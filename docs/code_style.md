# Recollex Code Style & Design Philosophy

Goal: small, composable primitives that are easy to reason about and test.

Core principles
- Function-first for behavior: filtering, scoring, ordering, reranking are pure functions.
- ABCs for stateful components: metadata store, segment reader, encoder; prefer dataclasses.
- No forests of classes for behavior. Prefer higher-order functions and functools.partial.
- Minimal global state. Pass everything needed; return new values.

Behavior shapes (no Protocols required)
- FilterFn(q_terms, filters, get_bitmap, df_lookup, base_bitmap, exclude_ids, knobs) -> (must_term_ids, should_term_ids)
- ScoreFn(q_csr, segment_ctx, row_offsets) -> [(row_offset, score)]
- RankMergeFn(per_segment_results, k) -> [(seg_id, row_offset, score)]
- RerankFn(query_terms, candidates) -> candidates

Loading and wiring
- Accept a callable directly or a dotted path. If a class with __call__ is provided, instantiate it.
- Keep default implementations as standalone functions. Profiles are partials over those functions.

Stateful components (ABCs, often dataclasses)
- MetadataStore: open/close, transactions, bitmap get/put, docs/tags/seq, stats/kv.
- SegmentReader: zero-copy memory-mapped CSR arrays and row_id/offset mapping.
- Encoder: SPLADE model wrapper (ONNX by default; Torch used when backend != "onnx"), dims; encode(texts) -> sparse term vectors.

Testing
- Unit-test pure functions with small fixtures. No DB or FS required.
- For ABCs, use fakes/mocks; integration tests can cover SQLite + np.load(..., mmap_mode="r").
- Deterministic behavior (no hidden caches in functions).

Evolution and compatibility
- Keep function signatures stable. If you need extra knobs, prefer passing a dict (knobs) over adding positional args.
- Provide small adaptor wrappers to bridge old/new signatures when necessary.
- Validate inputs at load-time (e.g., assert callable, inspect.signature if needed) and raise clear errors.

Documentation expectations
- Public functions have concise docstrings with parameter shapes and return types.
- Docs emphasize that “hooks are plain callables; classes with __call__ are accepted; ABCs only for stateful/lifecycle.”
