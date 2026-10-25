"""Reusable HNSW over every unmodified Gemma main-embedding row.

L2 here is embedding squared L2, NEVER the real KV acceptance threshold.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
sys.path.insert(0, str(ROOT / 'vendor'))
import hnswlib


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class EmbeddingIndex:
    def __init__(self, ef=512):
        self.metadata = json.loads((ASSETS/'manifest.json').read_text())
        self.index = hnswlib.Index(space='l2', dim=self.metadata['dimension'])
        self.index.load_index(str(ASSETS/'embedding.hnsw'))
        self.index.set_num_threads(4)
        self.index.set_ef(ef)
        assert self.index.get_current_count() == self.metadata['count']

    def query(self, vectors, k=200, ef=None):
        vectors = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        if vectors.shape[1] != self.metadata['dimension'] or not np.isfinite(vectors).all():
            raise ValueError('Expected finite vectors in original embedding coordinates')
        if not 1 <= k <= self.metadata['count']:
            raise ValueError('Invalid k')
        self.index.set_ef(max(k, ef if ef is not None else 2*k))
        return self.index.knn_query(vectors, k=k, num_threads=1)


def build():
    if (ASSETS/'manifest.json').exists() or (ASSETS/'embedding.hnsw').exists():
        raise FileExistsError('Existing index must be preserved')
    ASSETS.mkdir(parents=True, exist_ok=True)
    ep = ASSETS/'embeddings.npy'
    tick = time.perf_counter()
    if not ep.exists():
        sys.path.insert(0, str(ROOT.parent/'ciphertext_collision_comparison'))
        import experiment as source
        rt, prior, tokenizer, runtime = source.load_runtime()
        e = runtime.embeddings(list(range(runtime.lm.args.vocab_size))).numpy()
        assert e.shape == (262144, 1536) and np.isfinite(e).all()
        np.save(ep, e)
        dump(ASSETS/'source.json', dict(model=runtime.metadata(),
             special_token_ids=tokenizer.all_special_ids,
             extraction_seconds=time.perf_counter()-tick))
        del e, runtime, tokenizer, prior
        import gc
        gc.collect()
        rt.mx.clear_cache()
    e = np.load(ep, mmap_mode='r')
    n, d = e.shape
    index = hnswlib.Index(space='l2', dim=d)
    index.init_index(max_elements=n, M=32, ef_construction=200, random_seed=20261019)
    index.set_num_threads(4)
    tick = time.perf_counter()
    for start in range(0, n, 8192):
        stop = min(start+8192, n)
        index.add_items(e[start:stop], np.arange(start, stop), num_threads=4)
        print('BUILD', stop, '/', n, 'seconds', round(time.perf_counter()-tick, 2), flush=True)
    build_seconds = time.perf_counter()-tick
    index.save_index(str(ASSETS/'embedding.hnsw'))
    assert sorted(index.get_ids_list()) == list(range(n))
    # Check all stored vectors, not just self-query recall; duplicate rows are legal.
    for start in range(0, n, 8192):
        stop = min(start+8192, n)
        np.testing.assert_array_equal(index.get_items(np.arange(start, stop)), e[start:stop])
    manifest = dict(count=n, dimension=d, dtype='float32', space='l2',
        returned_distance='squared embedding L2; not raw KV L2',
        M=32, ef_construction=200, seed=20261019, build_threads=4,
        multithreaded_build_bitwise_deterministic=False,
        all_original_ids=True, special_ids_included=True, normalization=False,
        dimensionality_reduction=False, build_seconds=build_seconds,
        hnswlib_version=importlib.metadata.version('hnswlib'),
        every_stored_vector_exactly_verified=True,
        embedding_bytes=ep.stat().st_size,
        index_bytes=(ASSETS/'embedding.hnsw').stat().st_size,
        embeddings_sha256=digest(ep), index_sha256=digest(ASSETS/'embedding.hnsw'),
        source=json.loads((ASSETS/'source.json').read_text()))
    dump(ASSETS/'manifest.json', manifest)
    print('COMPLETE', n, 'rows; every original ID and stored vector checked', flush=True)


def benchmark():
    import torch
    torch.set_num_threads(4)
    e = np.load(ASSETS/'embeddings.npy', mmap_mode='r')
    index = EmbeddingIndex()
    assert sorted(index.index.get_ids_list()) == list(range(len(e)))
    for start in range(0, len(e), 8192):
        stop = min(start+8192, len(e))
        np.testing.assert_array_equal(index.index.get_items(np.arange(start, stop)), e[start:stop])
    rng = np.random.default_rng(20261019)
    selected = rng.choice(len(e), 96, replace=False)
    base = np.array(e[selected], copy=True)
    noise = rng.normal(size=base.shape).astype(np.float32)
    noise *= .1*np.linalg.norm(base, axis=1, keepdims=True)/np.linalg.norm(noise, axis=1, keepdims=True)
    queries = np.concatenate([base[:32], (base+noise)[32:64],
                              .5*(base[64:]+e[rng.choice(len(e),32,replace=False)])])
    # Exact brute force reference on the same CPU and raw coordinates.
    # Batched BLAS is a strong control; also time sequential requests, our use case.
    et = torch.from_numpy(np.array(e, copy=True))
    norm = et.square().sum(1)
    def exact(q):
        qt = torch.from_numpy(np.atleast_2d(q))
        d2 = norm[None, :] + qt.square().sum(1)[:, None] - 2*(qt@et.T)
        vals, ids = torch.topk(d2, 20, dim=1, largest=False)
        return ids.numpy(), vals.numpy()
    exact(queries[:1])
    tick = time.perf_counter()
    gold, gd = exact(queries)
    batch_seconds = time.perf_counter()-tick
    times=[]
    for q in queries:
        tick=time.perf_counter(); exact(q); times.append(time.perf_counter()-tick)
    import mlx.core as mx
    eg=mx.array(et.numpy())
    ng=mx.sum(eg.square(),axis=1)
    mx.eval(eg,ng)
    def exact_gpu(q):
        qt=mx.array(np.atleast_2d(q))
        d2=ng[None,:]+mx.sum(qt.square(),axis=1)[:,None]-2*(qt@eg.T)
        ix=mx.argpartition(d2,19,axis=1)[:,:20]
        vals=mx.take_along_axis(d2,ix,axis=1)
        mx.eval(ix,vals)
        return np.array(ix),np.array(vals)
    exact_gpu(queries[:1])
    tick=time.perf_counter();gi,gv=exact_gpu(queries)
    gpu_batch_seconds=time.perf_counter()-tick
    # Compare distance thresholds; tied IDs need not have identical ordering.
    np.testing.assert_allclose(gv.max(1),gd.max(1),rtol=3e-4,atol=2e-4)
    gpu_times=[]
    for q in queries:
        tick=time.perf_counter(); exact_gpu(q); gpu_times.append(time.perf_counter()-tick)
    runs=[]
    for ef in (64, 200, 800, 2048):
        index.query(queries[:1],20,ef)
        got, distances, latency=[], [], []
        for q in queries:
            tick=time.perf_counter(); ids, ds=index.query(q,20,ef)
            latency.append(time.perf_counter()-tick)
            got.append(ids[0]); distances.append(ds[0])
        got, distances=np.array(got), np.array(distances)
        direct=np.sum((e[got]-queries[:,None,:]).astype(np.float64)**2,axis=-1)
        np.testing.assert_allclose(distances,direct,rtol=2e-5,atol=2e-5)
        recall=[len(set(a)&set(b))/20 for a,b in zip(got,gold)]
        # Tie-aware score: counts vectors no farther than exact kth distance.
        tie_aware=np.mean(direct<=gd[:,-1,None]+2e-4,axis=1)
        runs.append(dict(ef=ef,k=20,recall_at_20=float(np.mean(recall)),
            recall_by_group={name:float(np.mean(recall[s:s+32])) for name,s in
                             [('self',0),('noisy',32),('midpoint',64)]},
            tie_aware_precision_at_20=float(np.mean(tie_aware)),
            mean_ms=float(np.mean(latency)*1000),p95_ms=float(np.percentile(latency,95)*1000),
            speedup_over_sequential_cpu_exact=float(np.mean(times)/np.mean(latency)),
            speedup_over_sequential_gpu_exact=float(np.mean(gpu_times)/np.mean(latency))))
    dump(ROOT/'benchmark.json',dict(queries=96,persisted_index_all_ids_and_vectors_verified=True,
        queries_from='embedding-only self/noise/midpoint; no KV labels',
        exact_cpu_threads=4,hnsw_query_threads=1,k=20,
        exact_sequential_mean_ms=float(np.mean(times)*1000),
        exact_batched_amortized_ms=batch_seconds/len(queries)*1000,
        exact_gpu_sequential_mean_ms=float(np.mean(gpu_times)*1000),
        exact_gpu_batched_amortized_ms=gpu_batch_seconds/len(queries)*1000,
        notes='Synthetic embedding retrieval only; not KV recovery rate. ID recall can penalize tied duplicate vectors.',
        runs=runs))
    print(json.dumps(runs,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['build','benchmark'])
    args=p.parse_args()
    {'build':build,'benchmark':benchmark}[args.stage]()
