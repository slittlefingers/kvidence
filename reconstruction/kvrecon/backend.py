"""Runtime-independent recovery contract. Query must never advance the prefix."""
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np


@dataclass(frozen=True)
class Capabilities:
    vocab_size: int
    embedding_dim: int
    context_capacity: int
    batch_capacity: int
    layers: tuple[int, ...]
    kv_dtype: str
    kv_width: int

    def __post_init__(self):
        if min(self.vocab_size, self.embedding_dim, self.context_capacity,
               self.batch_capacity, self.kv_width) <= 0 or not self.layers:
            raise ValueError("Adapter capabilities require positive dimensions and available layers")


class RecoveryBackend(Protocol):
    """Codes have shape [candidates, width]; values returns original-unit K+V.

    probabilities() returns ALL original IDs ordered by decreasing probability,
    not a vector of probabilities. Ties must use a deterministic order.
    reset replaces the prefix; commit advances exactly once. Query is read-only
    with respect to prefix, logits and position, including internal cache state.
    A serial backend reports batch_capacity=1.
    """
    position: int
    vocab_size: int
    batch_capacity: int
    decoder_seconds: float
    setup_seconds: float

    def reset(self, prefix: Sequence[int]) -> None: ...
    def probabilities(self) -> np.ndarray: ...
    def query(self, ids: Sequence[int]) -> np.ndarray: ...
    def commit(self, token: int) -> np.ndarray: ...
    def values(self, codes: np.ndarray) -> np.ndarray: ...
    def __enter__(self) -> "RecoveryBackend": ...
    def __exit__(self, typ, value, tb): ...


class CandidateBatchBackend:
    """Keep logical search batches unchanged, split into physical backend calls.

    In particular Collision+ still picks the first passing ID in probability
    order, and guided still chooses the minimum over its complete stage.
    This wrapper owns no native resources; enter/exit the underlying backend.
    """
    def __init__(self, backend: RecoveryBackend, logical_capacity: int):
        if logical_capacity <= 0 or backend.batch_capacity <= 0:
            raise ValueError("Batch capacities must be positive")
        self.backend = backend
        self.batch_capacity = logical_capacity
        self.physical_batch_capacity = backend.batch_capacity

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def query(self, ids):
        ids = list(ids)
        if (not ids or len(ids) > self.batch_capacity or len(set(ids)) != len(ids)
                or any(type(i) not in (int, np.int64, np.int32) or not 0 <= i < self.vocab_size for i in ids)):
            raise ValueError("Invalid logical candidate batch")
        position = self.position
        rows = []
        for start in range(0, len(ids), self.physical_batch_capacity):
            chunk = ids[start:start+self.physical_batch_capacity]
            row = np.asarray(self.backend.query(chunk))
            if row.ndim != 2 or row.shape[0] != len(chunk):
                raise ValueError("Backend query returned malformed rows")
            if self.position != position:
                raise ValueError("Backend query changed prefix position")
            rows.append(row)
        return np.concatenate(rows, axis=0)

    def query_retained(self, ids):
        begin=getattr(self.backend,'begin_candidates',None)
        accept=getattr(self.backend,'accept_query',None)
        if not callable(begin) or not callable(accept):
            raise ValueError('CPA requires retained candidate cache support from adapter')
        begin()
        return self.query(ids)


def validate_contract(backend: RecoveryBackend, prefix, ids, *, atol=0., rtol=0.):
    """Destructive, explicit adapter test; restores prefix on exit, not an attack.

    Compares original KV values, reports errors even when tolerances allow them.
    Finite samples validate an implementation; they do not prove equivalence.
    """
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= backend.vocab_size:
        raise ValueError("Supply unique original IDs")
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Invalid contract tolerance")
    wrapped = CandidateBatchBackend(backend, len(ids))
    checks = []

    def compare(name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(name + ": malformed values")
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
        checks.append(dict(name=name, maximum_absolute_error=float(np.max(np.abs(a-b)))))

    try:
        backend.reset(prefix)
        if backend.position != len(prefix):
            raise AssertionError("reset position")
        ranking = backend.probabilities()
        if not np.array_equal(np.sort(ranking), np.arange(backend.vocab_size)):
            raise AssertionError("probability ranking must cover original vocabulary")
        batch = backend.values(wrapped.query(ids))
        np.testing.assert_array_equal(backend.probabilities(), ranking)
        compare("repeat_query", backend.values(wrapped.query(ids)), batch)
        compare("candidate_permutation", backend.values(wrapped.query(ids[::-1])), batch[::-1])
        singles = []
        for token in ids:
            backend.reset(prefix)
            singles.append(backend.values(backend.query([token])))
        compare("batch_vs_fresh_single_queries", np.concatenate(singles), batch)
        backend.reset(prefix)
        compare("commit_vs_query", backend.values(backend.commit(ids[0])), batch[:1])
        if backend.position != len(prefix)+1:
            raise AssertionError("commit must advance exactly one position")
        next_values = backend.values(wrapped.query(ids))
        backend.reset(list(prefix)+[ids[0]])
        compare("commit_vs_prefix_replay", backend.values(wrapped.query(ids)), next_values)
        return dict(passed=True, checks=checks, atol=atol, rtol=rtol,
                    physical_batch_capacity=backend.batch_capacity, candidate_ids=ids,
                    note="Finite adapter contract test, not recovery accuracy or uniqueness proof")
    finally:
        backend.reset(prefix)
