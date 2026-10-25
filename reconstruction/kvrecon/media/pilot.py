"""Run the image/audio module against existing public MLX pilot observations.

This command fits a representation and verifies it by full MLX replay. It does
not label media content recovered, run text recovery, or claim NPU validation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from ..common import dump, digest, EXPERIMENTS
from .solver import fit


def main(modality, builder):
    parser = argparse.ArgumentParser(description=f'Gemma4 {modality} latent inversion from layer-5 KV')
    parser.add_argument('--observation', type=Path, required=True, help='Public pilot observed/<case> directory')
    parser.add_argument('--model', type=Path, default=EXPERIMENTS/'kv_guided_search_gemma4_int8/model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=.08)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29])
    parser.add_argument('--epsilon', type=float, default=4.)
    args = parser.parse_args()
    if args.steps < 0 or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error('steps >= 0, lr > 0 and epsilon > 0 must be finite')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('seeds must be distinct nonnegative integers')
    if not args.model.is_dir():
        parser.error('--model must be an existing local MLX model directory')
    if args.output.exists():
        raise FileExistsError('Preserve the existing media inversion output')
    if any(part in ('datasets', 'private_eval') for part in args.model.resolve().parts):
        raise ValueError('Model assets cannot be loaded from source datasets or private evaluation')
    # Initialize trusted dependencies before the recovery read guard: mlx_vlm
    # imports its own Python trainer/datasets package, not our source datasets.
    # No observation, original media or label is loaded during this setup.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import mlx.core as mx
    from mlx_vlm import load
    print(f'Loading local Gemma model for {modality}', flush=True)
    model, _ = load(str(args.model)); model.eval()
    from ..cli import private_guard
    private_guard()
    media_roots = [(EXPERIMENTS/name/assets).resolve()
                   for name in ('gemma4_multimodal_layer5', 'gemma4_mixed_media_layer5')
                   for assets in ('assets', 'assets_pilot300')]

    def guard(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes, Path)):
            path = Path(os.fsdecode(values[0])).resolve()
            if any(path.is_relative_to(root) for root in media_roots):
                raise PermissionError('Media inversion cannot read original media assets')
    sys.addaudithook(guard)
    meta = json.loads((args.observation/'metadata.json').read_text())
    if meta.get('modality') != modality or meta.get('layer_1based') != 5 or meta.get('kv_dtype') != 'mlx.core.bfloat16':
        raise ValueError('This import command requires matching modality and the layer-5 BF16 MLX pilot format')
    with np.load(args.observation/'observation.npz', allow_pickle=False) as data:
        pos, targets = data['media_positions'].copy(), data['kv'].copy()
        end = int(pos[-1])+1
        ids, types = data['input_ids'][:, :end].copy(), data['token_types'][:, :end].copy()
    args.output.mkdir(parents=True, exist_ok=False)
    import importlib.metadata
    protocol = dict(modality=modality, observation_metadata=meta,
        observation_sha256=digest(args.observation/'observation.npz'),
        metadata_sha256=digest(args.observation/'metadata.json'),
        model_assets={p.name: digest(p) for p in sorted(args.model.glob('*'))
                      if p.suffix in ('.json', '.safetensors')},
        source_hashes={p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        packages={name: importlib.metadata.version(name) for name in ('mlx', 'mlx-vlm', 'numpy')},
        steps=args.steps, lr=args.lr, seeds=args.seeds, epsilon=args.epsilon,
        selection='minimum KV NMSE across seeds, never true media/features',
        backend='MLX; W8 weights/BF16 KV pilot, not Pixel NPU',
        target_features_accessed=False, original_media_accessed=False,
        subsequent_text_recovery_evaluated=False, full_model_replay=True)
    dump(args.output/'protocol.json', protocol)
    results = []
    for seed in args.seeds:
        out = args.output/f'seed_{seed}'; out.mkdir()
        problem = builder(model, ids, pos, targets, types, steps=args.steps, lr=args.lr, seed=seed)
        initial = problem.project(problem.initial)
        initial_nmse = float(problem.loss(mx.array(problem.initial)).item())
        tick = time.perf_counter()
        features, best, trace = fit(problem)
        fit_seconds = time.perf_counter()-tick
        actual, cache, logits = problem.replay(features)
        predicted = np.array(problem.forward(mx.array(features)).astype(mx.float32))[0]
        np.testing.assert_array_equal(actual, predicted)
        target = np.asarray(targets).reshape(actual.shape)
        distances = np.linalg.norm(actual.astype(np.float64)-target, axis=-1)
        np.save(out/'features.npy', features); np.save(out/'latent.npy', problem.best_u)
        np.save(out/'initial_features.npy', initial)
        dump(out/'optimization.json', trace)
        result = dict(status='fitted', optimizer=problem.fit_metadata, initial_kv_nmse=initial_nmse,
            best_kv_nmse=best, fit_seconds=fit_seconds, full_model_replay_matches=True,
            per_position_l2=distances.tolist(), max_position_l2=float(distances.max()),
            positions_passing_l2=int(np.sum(distances<args.epsilon)), media_positions=len(pos),
            passes_all_positions_l2=bool(np.all(distances<args.epsilon)),
            feature_sha256=digest(out/'features.npy'),
            note='A fitted replacement representation; no original content accuracy or NPU claim')
        dump(out/'result.json', result)
        results.append(dict(seed=seed, **result))
        del problem, initial, features, cache, logits
        mx.clear_cache()
    selected = min(results, key=lambda r: r['best_kv_nmse'])
    np.save(args.output/'features.npy', np.load(args.output/f"seed_{selected['seed']}"/'features.npy', allow_pickle=False))
    dump(args.output/'result.json', dict(status='fitted', selected_seed=selected['seed'],
        best_kv_nmse=selected['best_kv_nmse'], passes_all_positions_l2=selected['passes_all_positions_l2'],
        feature_sha256=digest(args.output/'features.npy'), runs=results,
        note='No original media reconstruction, later-token recovery or NPU validation inferred'))
    print(f"{modality}: selected seed={selected['seed']}, KV NMSE={selected['best_kv_nmse']:.8g}", flush=True)
