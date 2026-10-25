"""Cross-backend proposal fit; only the subsequent Pixel replay can certify it."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from kvrecon.common import dump, digest, EXPERIMENTS
from kvrecon.media.image import build_problem
from kvrecon.media.solver import fit

def main():
    import mlx.core as mx
    from mlx_vlm import load
    model_path = EXPERIMENTS/'kv_guided_search_gemma4_int8/model'
    print('Loading local MLX proposal model', flush=True)
    model, _ = load(str(model_path)); model.eval()
    from kvrecon.cli import private_guard
    private_guard()
    out = HERE/'fit'; out.mkdir(exist_ok=False)
    with np.load(HERE/'input/observed.npz') as d:
        ids, types, pos, target = (d[n].copy() for n in ('input_ids','token_types','media_positions','media_kv'))
    dump(out/'protocol.json', dict(seeds=[17,29], steps=3000, lr=.08,
        target='captured Pixel NPU INT8 layer 5, dequantized original units',
        proposal='local MLX W8 model; not numerically identical to NPU',
        selection='minimum measured Pixel replay KV error; never original image or text',
        observations_sha256=digest(HERE/'input/observed.npz'),
        original_media_accessed=False, original_text_accessed=False,
        evaluation='all per-position raw K+V L2 < 4; semantic content and exact text evaluated separately'))
    for seed in (17,29):
        p = build_problem(model, ids, pos, target, types, steps=3000, lr=.08, seed=seed)
        tick = time.perf_counter()
        features, best, trace = fit(p)
        actual = np.array(p.forward(mx.array(features)))[0]
        distances = np.linalg.norm(actual.astype(np.float64)-target,axis=-1)
        dest = out/f'seed_{seed}'; dest.mkdir()
        np.save(dest/'features.npy', features)
        np.save(dest/'latent.npy', p.best_u)
        np.save(dest/'proposal_kv.npy', actual)
        dump(dest/'trace.json',trace)
        dump(dest/'result.json',dict(best_nmse=best, seconds=time.perf_counter()-tick,
            per_position_l2=distances.tolist(), max_l2=float(distances.max()),
            positions_below_4=int((distances<4).sum()), seed=seed,
            optimizer=p.fit_metadata, backend='MLX proposal only'))
        print(f'Seed {seed}: NMSE={best}, max L2={distances.max()}, pass={(distances<4).sum()}/{len(pos)}',flush=True)
        del p, features, actual
        mx.clear_cache()

if __name__ == '__main__':
    main()
