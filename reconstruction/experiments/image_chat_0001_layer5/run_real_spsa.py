"""Recover real image positions causally using only phone forwards.

Separate follow-up with the user's revised raw KV L2 < 5 image criterion.
Never use private labels or observed target KV as decoder prefix state.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from spsa_adam import Config, fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from kvrecon.common import dump, digest, EXPERIMENTS
from kvrecon.npu import NpuBackend
from kvrecon.cli import private_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'real_spsa_l2_5')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=29)
    args = parser.parse_args()
    private_guard()
    meta = json.loads((HERE/'input/metadata.json').read_text())
    sections = json.loads((EXPERIMENTS/'npu_candidate_batch/inputs/probe.json').read_text())['model_sections']
    with np.load(HERE/'input/observed.npz') as data:
        targets = data['media_kv'].astype(np.float64)
    cfg = Config(steps=args.steps, seed=args.seed, acceptance_l2=5.)
    out = args.output
    out.mkdir(exist_ok=False)
    dump(out/'protocol.json', dict(
        config=asdict(cfg), sample='0001', target_layer_1based=5,
        acceptance='strict raw concatenated K+V L2 < 5 separately at every image position; query AND commit',
        criterion_change='User revised image threshold after synthetic experiment; original < 4 results preserved',
        synthetic_evidence='seed 29 query and commit L2 4.903690516402157; seed 17 did not reach 5',
        plan='Sequential positions; stop on first failed position. Describe image only after all 260 positions pass.',
        seed_selection='29 chosen from prior synthetic performance, before real target optimization',
        original_image_accessed=False, original_text_accessed=False,
        target_cache_used_as_prefix=False, desktop_model_loaded=False,
        adapter='phone CPU vision_adapter_280', decoder='phone NPU prefill_128 / decode',
        model_sha256=meta['model_sha256'], target_sha256=digest(HERE/'input/observed.npz'),
        source_sha256={p.name:digest(p) for p in (Path(__file__), HERE/'spsa_adam.py')}))
    features = []
    runs = []
    with NpuBackend(sections, 4, out/'device', meta['model_sha256']) as backend:
        backend.reset(meta['public_prefix'])
        for i, target in enumerate(targets):
            position = meta['media_start'] + i
            assert backend.position == position
            folder = out/f'position_{position:04d}'
            latent, result = fit(lambda rows:backend.values(backend.query_vision(rows)), target, cfg, folder)
            assert backend.position == position
            feature = backend.project_vision(latent[None])
            committed = backend.values(backend.commit_embeddings(feature))[0]
            distance = float(np.linalg.norm(committed.astype(np.float64)-target))
            result.update(position=position, commit_l2=distance, passes_commit=distance<cfg.acceptance_l2)
            np.save(folder/'features.npy', feature)
            np.save(folder/'commit_kv.npy', committed)
            dump(folder/'result.json', result)
            runs.append(result)
            dump(out/'runs.json', runs)
            print(f'Real image position {position}: query={result["best_l2"]:.6f}, commit={distance:.6f}', flush=True)
            if not (result['passes_query'] and result['passes_commit']):
                break
            features.append(feature)
            np.save(out/'accepted_features.npy', np.concatenate(features))
        complete = len(features) == len(targets)
        dump(out/'result.json', dict(
            status='all_image_positions_passed' if complete else 'stopped_at_failed_image_position',
            acceptance_l2=cfg.acceptance_l2, total_positions=len(targets),
            accepted_positions=len(features), attempted_positions=len(runs),
            whole_image_representation_available=complete, image_content_evaluated=False,
            subsequent_text_recovered=False, runs=runs))
        # A complete accepted image is required before requesting a description.
        if complete:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(EXPERIMENTS/'kv_guided_search_gemma4_int8/model/tokenizer.json'))
            backend.commit_embeddings(np.load(HERE/'input/end_of_image.npy').reshape(1,1536))
            for token in meta['separator_ids']:
                backend.commit(token)
            question = 'Describe the image briefly. Name the visible objects and actions.'
            for token in tokenizer.encode(question+'<turn|>\n<|turn>model\n', add_special_tokens=False).ids:
                backend.commit(token)
            stop = {1, tokenizer.token_to_id('<turn|>'), tokenizer.token_to_id('<|endoftext|>')}
            generated = []
            for _ in range(160):
                token = int(backend.probabilities()[0])
                if token in stop:
                    break
                generated.append(token)
                backend.commit(token)
            dump(out/'image_description.json', dict(question=question, token_ids=generated,
                text=tokenizer.decode(generated), uses_original_image=False,
                uses_recovered_features=True, truncated=len(generated)==160))


if __name__ == '__main__':
    main()
