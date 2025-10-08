from __future__ import annotations

import heapq
import math
import numpy as np
from scipy.sparse import csr_matrix
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from pyroaring import BitMap as Roaring
from recollex.bitmaps import TERM_PREFIX


# ---------------------------
# Helpers
# ---------------------------

def _or_bitmaps(bitmaps: Iterable[Roaring]) -> Roaring:
    acc: Optional[Roaring] = None
    for bm in bitmaps:
        acc = bm if acc is None else (acc | bm)
    return acc if acc is not None else Roaring()

def _and_bitmaps(bitmaps: Iterable[Roaring]) -> Roaring:
    acc: Optional[Roaring] = None
    for bm in bitmaps:
        acc = bm if acc is None else (acc & bm)
    return acc if acc is not None else Roaring()

def _len_bitmap(bm: Roaring) -> int:
    return len(bm)

def _first_n(bm: Roaring, n: int) -> Roaring:
    out = Roaring()
    if n <= 0:
        return out
    if n >= _len_bitmap(bm):
        return bm
    for i, x in enumerate(bm):
        if i >= n:
            break
        out.add(int(x))
    return out


# ---------------------------
# Filter policies
# ---------------------------

def filter_policy_default(
    q_terms: Iterable[Tuple[int, float]],
    filters: Optional[Dict[str, str]] = None,
    get_bitmap: Optional[Callable[[str], Roaring]] = None,
    get_df: Optional[Callable[[int], int]] = None,
    base_bitmap: Optional[Roaring] = None,
    exclude_doc_ids: Optional[Iterable[str]] = None,
    knobs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Greedy MUST growth by weight*idf until |B ∩ MUST| <= budget or min_must reached.
    SHOULD = next top-weight terms capped.

    Returns: (must_term_ids, should_term_ids)

    knobs:
      - budget (int, default 50_000)
      - min_must (int, default 1)
      - should_cap (int, default 100)
      - df_drop_top_percent (float, default 1.0)  # drop top-% highest-DF terms
    """
    knobs = knobs or {}
    budget = int(knobs.get("budget", 50_000))
    min_must = int(knobs.get("min_must", 1))
    should_cap = int(knobs.get("should_cap", 100))
    df_drop_top_percent = float(knobs.get("df_drop_top_percent", 1.0))

    terms = [(int(tid), float(wt)) for tid, wt in q_terms]
    if not terms:
        return ([], [])

    # Estimate IDF; if total docs unknown, use 1/(df+1). Lower df => higher idf.
    def idf_for(tid: int) -> float:
        if get_df is None:
            return 1.0
        df = max(0, int(get_df(tid) or 0))
        return 1.0 / (df + 1.0)

    # Optionally drop top-% highest DF terms
    if get_df is not None and df_drop_top_percent > 0:
        with_df = [(tid, wt, int(get_df(tid) or 0)) for tid, wt in terms]
        with_df.sort(key=lambda x: x[2], reverse=True)
        kdrop = int(math.floor(len(with_df) * (df_drop_top_percent / 100.0)))
        kept = with_df[kdrop:] if kdrop > 0 else with_df
        terms = [(tid, wt) for tid, wt, _df in kept]
        if not terms:
            return ([], [])

    # Rank by weight * idf
    ranked = sorted(terms, key=lambda tw: tw[1] * idf_for(tw[0]), reverse=True)
    must: List[int] = []
    should: List[int] = []

    # If we cannot measure cardinality reduction (no get_bitmap or no base), fallback to first min_must
    if get_bitmap is None or base_bitmap is None:
        must = [tid for tid, _wt in ranked[:max(min_must, 0)]]
        rest = [tid for tid, _wt in ranked if tid not in must]
        should = rest[:max(should_cap, 0)]
        return (must, should)

    # Greedy intersection to drive candidate count below budget (or until min_must)
    current = base_bitmap
    for tid, _wt in ranked:
        # Intersect with this term
        term_bm = get_bitmap(f"{TERM_PREFIX}{tid}")
        if term_bm is None:
            continue
        new_current = current & term_bm
        if _len_bitmap(new_current) <= _len_bitmap(current) and (_len_bitmap(new_current) <= budget or len(must) < min_must):
            must.append(tid)
            current = new_current
        # Early stop if under budget and min_must satisfied
        if _len_bitmap(current) <= budget and len(must) >= min_must:
            break

    # Remaining terms become SHOULD (capped)
    remaining = [tid for tid, _wt in ranked if tid not in must]
    should = remaining[:max(should_cap, 0)]
    return (must, should)


def filter_policy_paraphrase_hp(
    q_terms: Iterable[Tuple[int, float]],
    filters: Optional[Dict[str, str]] = None,
    get_bitmap: Optional[Callable[[str], Any]] = None,
    get_df: Optional[Callable[[int], int]] = None,
    base_bitmap: Optional[Any] = None,
    exclude_doc_ids: Optional[Iterable[str]] = None,
    knobs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[int]]:
    """
    High precision: more MUST, smaller SHOULD, drop a bit more of top-DF.
    """
    knobs = {**(knobs or {})}
    knobs.setdefault("min_must", 3)
    knobs.setdefault("should_cap", 24)
    knobs.setdefault("df_drop_top_percent", 3.0)
    knobs.setdefault("budget", 10_000)
    return filter_policy_default(q_terms, filters, get_bitmap, get_df, base_bitmap, exclude_doc_ids, knobs)


def filter_policy_rag(
    q_terms: Iterable[Tuple[int, float]],
    filters: Optional[Dict[str, str]] = None,
    get_bitmap: Optional[Callable[[str], Any]] = None,
    get_df: Optional[Callable[[int], int]] = None,
    base_bitmap: Optional[Any] = None,
    exclude_doc_ids: Optional[Iterable[str]] = None,
    knobs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[int]]:
    """
    High recall: minimal MUST, larger SHOULD, drop fewer top-DF terms.
    """
    knobs = {**(knobs or {})}
    knobs.setdefault("min_must", 0)
    knobs.setdefault("should_cap", 200)
    knobs.setdefault("df_drop_top_percent", 0.5)
    knobs.setdefault("budget", 150_000)
    return filter_policy_default(q_terms, filters, get_bitmap, get_df, base_bitmap, exclude_doc_ids, knobs)


def filter_policy_recent(
    q_terms: Iterable[Tuple[int, float]],
    filters: Optional[Dict[str, str]] = None,
    get_bitmap: Optional[Callable[[str], Any]] = None,
    get_df: Optional[Callable[[int], int]] = None,
    base_bitmap: Optional[Any] = None,
    exclude_doc_ids: Optional[Iterable[str]] = None,
    knobs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Recency-first: bypass term gating. Use base bitmap only downstream.
    """
    return ([], [])


# ---------------------------
# Candidate suppliers
# ---------------------------

def _trim_to_budget(bm: Roaring, budget: Optional[int]) -> Roaring:
    if budget is None:
        return bm
    return bm if _len_bitmap(bm) <= int(budget) else _first_n(bm, int(budget))

def candidate_supplier_default(
    must_terms: Iterable[int],
    should_terms: Iterable[int],
    base_bitmap: Roaring,
    get_bitmap: Callable[[str], Optional[Roaring]],
    budget: Optional[int] = None,
) -> Roaring:
    """
    C = base ∩ AND(MUST) ∩ OR(SHOULD)
    """
    current = base_bitmap
    # MUST
    for tid in must_terms:
        bm = get_bitmap(f"{TERM_PREFIX}{int(tid)}")
        if bm is None:
            # Skip missing postings
            continue
        current = current & bm

    # SHOULD
    should_list = list(should_terms)
    if should_list:
        postings = [get_bitmap(f"{TERM_PREFIX}{int(tid)}") for tid in should_list]
        postings = [bm for bm in postings if bm is not None]
        if postings:
            s_union = _or_bitmaps(postings)  # type: ignore[arg-type]
            current = current & s_union

    # Do not hard-cap here; policy should have gated to <= budget already.
    # Optionally trim to budget if asked and supported by type.
    return _trim_to_budget(current, budget)


def candidate_supplier_recent(
    base_bitmap: Roaring,
    budget: Optional[int] = None,
) -> Roaring:
    """
    Returns the base bitmap (minus tombstones/exclusions upstream), capped by budget if provided.
    """
    return _trim_to_budget(base_bitmap, budget)


# ---------------------------
# Scorers
# ---------------------------

def score_csr_slice(
    q_csr: csr_matrix,
    segment_ctx: Mapping[str, Any],
    row_offsets: Iterable[int],
) -> List[Tuple[int, float]]:
    """
    Exact scores by slicing CSR rows, then doing one batched sparse dot:
        scores = (q_csr @ X_slice.T).A1

    segment_ctx expects either:
      - 'csr': a scipy.sparse.csr_matrix with shape (n_rows, dims), or
      - 'indptr' (int64), 'indices' (int32), 'data' (float32), and 'dims' (int)
    """
    X = segment_ctx.get("csr")
    if X is None:
        indptr = segment_ctx["indptr"]
        indices = segment_ctx["indices"]
        data = segment_ctx["data"]
        dims = int(segment_ctx["dims"])
        n_rows = int(indptr.shape[0] - 1)
        X = csr_matrix((data, indices, indptr), shape=(n_rows, dims))
    else:
        dims = int(X.shape[1])

    rows = np.fromiter((int(r) for r in row_offsets), dtype=np.int64)
    if rows.size == 0:
        return []

    if q_csr.shape[1] != dims:
        raise ValueError(f"q dims {q_csr.shape[1]} != segment dims {dims}")

    X_top = X[rows]
    scores_mat = q_csr @ X_top.T
    try:
        scores = scores_mat.A1  # available on some SciPy versions
    except AttributeError:
        scores = scores_mat.toarray().ravel()
    return list(zip(rows.tolist(), scores.tolist()))


def score_accumulator(
    q_terms: Iterable[Tuple[int, float]],
    postings_by_term: Mapping[int, Iterable[Tuple[int, float]]],
    row_offsets: Iterable[int],
) -> List[Tuple[int, float]]:
    """
    For tiny candidate sets: accumulate scores from postings.

    postings_by_term maps term_id -> iterable of (row_offset, value)
    Only rows present in 'row_offsets' are returned.
    """
    want = set(int(r) for r in row_offsets)
    w = {int(tid): float(wt) for tid, wt in q_terms}
    scores: Dict[int, float] = {}
    for tid, wt in w.items():
        postings = postings_by_term.get(int(tid))
        if not postings:
            continue
        for row, val in postings:
            r = int(row)
            if r in want:
                scores[r] = scores.get(r, 0.0) + wt * float(val)
    return [(r, scores.get(r, 0.0)) for r in want]


def score_hook_noop(
    q_csr: csr_matrix,
    segment_ctx: Any,
    row_offsets: Iterable[int],
) -> List[Tuple[int, float]]:
    """No-op scorer (score=0.0). Useful when ranking by recency."""
    return [(int(r), 0.0) for r in row_offsets]


# ---------------------------
# Rank merges
# ---------------------------

def rank_merge_heap(
    per_segment: Dict[str, List[Tuple[int, float]]],
    k: int,
) -> List[Tuple[str, int, float]]:
    """
    k-way heap merge of per-segment (row_offset, score) lists.
    Returns [(segment_id, row_offset, score)] sorted by score desc.
    """
    heap: List[Tuple[float, str, int]] = []
    for seg, pairs in per_segment.items():
        for row, score in pairs:
            heapq.heappush(heap, (-float(score), seg, int(row)))
    out: List[Tuple[str, int, float]] = []
    for _ in range(min(k, len(heap))):
        neg, seg, row = heapq.heappop(heap)
        out.append((seg, row, -neg))
    return out


def rank_merge_recent(
    per_segment: Dict[str, List[Tuple[int, int]]],
    k: int,
) -> List[Tuple[str, int, int]]:
    """
    Recency-first merge: treats the second element as 'seq' (monotonic), sorts desc.
    per_segment: {seg_id: [(row_offset, seq), ...]}
    Returns [(seg_id, row_offset, seq)] top-k by seq desc.
    """
    all_items: List[Tuple[str, int, int]] = []
    for seg, pairs in per_segment.items():
        for row, seq in pairs:
            all_items.append((seg, int(row), int(seq)))
    all_items.sort(key=lambda x: x[2], reverse=True)
    return all_items[:k]


# ---------------------------
# Cache eviction
# ---------------------------

def evict_lru(cache_state: Dict[str, Any], max_items: int) -> List[str]:
    """
    Evict least-recently-used entries.

    cache_state conventions (flexible):
      - Either:
          cache_state == {name: {"last_used": int, "size": int}, ...}
        OR:
          cache_state == {"entries": {name: {"last_used": int, "size": int}, ...},
                          "total_size": int,
                          "ram_limit_bytes": int (optional)}

    Behavior:
      - Always enforce count cap first (len <= max_items), then
      - If 'ram_limit_bytes' present, evict further until sum(size) <= limit.
    """
    # Normalize to an 'entries' dict (copy; we do not mutate the input)
    entries: MutableMapping[str, Dict[str, Any]]
    ram_limit: Optional[int] = None

    if "entries" in cache_state:
        entries = dict(cache_state.get("entries", {}))
        ram_limit = cache_state.get("ram_limit_bytes")
    else:
        # assume flat mapping
        entries = {k: dict(v) for k, v in cache_state.items() if isinstance(v, dict)}

    # Sort by last_used ascending (oldest first)
    items = [(name, ent.get("last_used", 0), int(ent.get("size", 0))) for name, ent in entries.items()]
    # Only need ordering of the oldest few; get a sorted prefix if possible
    try:
        oldest = heapq.nsmallest(max(0, len(items)), items, key=lambda x: x[1])
        items = oldest
    except Exception:
        items.sort(key=lambda x: x[1])

    victims: List[str] = []

    # 1) Count-based eviction
    keep = max(0, int(max_items))
    excess = max(0, len(items) - keep)
    if excess > 0:
        victims.extend([name for name, _lu, _sz in items[:excess]])
        items = items[excess:]

    # 2) RAM-based eviction (optional)
    if ram_limit is not None and len(items) > 0:
        total = sum(sz for _n, _lu, sz in items)
        limit = int(ram_limit)
        i = 0
        while total > limit and i < len(items):
            name, _lu, sz = items[i]
            victims.append(name)
            total -= sz
            i += 1

    return victims


def make_profile(
    profile: str,
    override_knobs: Optional[Dict[str, Any]] = None,
) -> Tuple[Callable[..., Tuple[List[int], List[int]]], str]:
    """
    Return (filter_fn, order) for a named profile.
    filter_fn is a callable with knobs pre-applied (via functools.partial).
    order is "score" or "recent".
    """
    p = profile.lower().strip()

    if p in ("paraphrase_hp", "hp", "precise"):
        base = filter_policy_paraphrase_hp
        order = "score"
        base_knobs: Dict[str, Any] = {}
    elif p in ("rag", "recall"):
        base = filter_policy_rag
        order = "score"
        base_knobs = {}
    elif p in ("recent", "log_recent", "recency", "log"):
        base = filter_policy_recent
        order = "recent"
        base_knobs = {}
    else:
        raise ValueError(f"Unknown profile: {profile}")

    knobs = {**base_knobs, **(override_knobs or {})}
    filter_fn = partial(base, knobs=knobs) if knobs else base
    return (filter_fn, order)


__all__ = [
    # filter policies
    "filter_policy_default",
    "filter_policy_paraphrase_hp",
    "filter_policy_rag",
    "filter_policy_recent",
    # candidate suppliers
    "candidate_supplier_default",
    "candidate_supplier_recent",
    # scorers
    "score_csr_slice",
    "score_accumulator",
    "score_hook_noop",
    # rank merges
    "rank_merge_heap",
    "rank_merge_recent",
    # cache eviction
    "evict_lru",
    "make_profile",
]
