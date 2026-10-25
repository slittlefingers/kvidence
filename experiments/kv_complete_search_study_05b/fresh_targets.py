#!/usr/bin/env python3
"""Fresh token targets under the frozen, separately saved protocol."""
import json
import time
from pathlib import Path

import numpy as np
import torch

import experiment as e


def main():
    torch.set_num_threads(4)
    started = time.perf_counter()
    out = e.ROOT / 'fresh_results'
    base_config = json.loads((e.ROOT / 'results/config.json').read_text())
    frozen = base_config['frozen_inputs_sha256'].copy()
    for path in (e.ROOT / 'experiment.py', e.ROOT / 'results/local.json', e.ROOT / 'results/empirical.json'):
        frozen[str(path)] = e.d.digest(path)
    for path, expected in frozen.items():
        assert e.d.digest(Path(path)) == expected
    config = {**base_config, 'cohort': 'fresh', 'source_sha256': e.d.digest(Path(__file__)),
        'study_script_sha256': e.d.digest(e.ROOT / 'experiment.py'),
        'protocol_sha256': e.d.digest(e.ROOT / 'NEW_TARGET_PROTOCOL.md'),
        'frozen_inputs_sha256': frozen, 'fresh_targets_per_pool': 8, 'target_seed_offset': 14000000}
    e.d.save(out / 'config.json', config)
    previous = json.loads((e.d.SEARCH / 'results/trials.json').read_text())
    ids_by_seed, embedding_by_seed, pca_by_seed, phi_by_seed = {}, {}, {}, {}
    geometry, candidate_texts = {}, {}
    for seed in e.d.search.SEEDS:
        with np.load(e.d.DATA / str(seed) / 'features.npz') as raw:
            ids_by_seed[seed] = raw['ids'].copy()
            embedding_by_seed[seed] = torch.from_numpy(raw['embedding'].copy())
            pca_by_seed[seed] = torch.from_numpy(raw['pca'].copy())
            phi_by_seed[seed] = torch.from_numpy(raw['phi'].copy())
            geometry[seed] = e.grouped_labels(raw['pca'].astype(np.float64), 'embedding')
        candidate_texts[seed] = json.loads((e.d.DATA / str(seed) / 'candidates.json').read_text())
    old_ids = {int(ids_by_seed[c['seed']][c['target_index']]) for c in previous}
    forbidden = old_ids.copy()
    new_ids = set()
    cases, local, empirical = [], [], []
    with torch.inference_mode():
        for seed in e.d.search.SEEDS:
            for trial in range(8):
                n = len(ids_by_seed[seed])
                generator = torch.Generator().manual_seed(seed + 14000000 + trial * 101)
                target_index = int(torch.randint(n, (), generator=generator))
                while int(ids_by_seed[seed][target_index]) in forbidden:
                    target_index = int(torch.randint(n, (), generator=generator))
                token_id = int(ids_by_seed[seed][target_index])
                forbidden.add(token_id)
                new_ids.add(token_id)
                initial = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 14100000 + trial))[:200]
                pi = trial % 4
                with np.load(e.d.DATA / str(seed) / f'prefix_{pi}.npz') as raw:
                    tables = {layer: raw[f'kv_{layer}'].copy() for layer in (12, 24)}
                for layer, table in tables.items():
                    case = {'key': f'fresh_s{seed}_t{trial}_l{layer}', 'split': 'fresh', 'seed': seed,
                        'trial': trial, 'prefix_index': pi, 'prefix': e.d.search.prior.PREFIXES[pi],
                        'layer': layer, 'target_index': target_index, 'target_token_id': token_id,
                        'target_text': candidate_texts[seed][target_index]['text'],
                        'initial_indices': initial.tolist(), 'search_seed': seed + 14200000 + trial, 'methods': {}}
                    target = table[target_index].astype(np.float64)
                    scale = max(float(np.abs(target).max()), 1e-12)
                    true = np.abs(table.astype(np.float64) - target).max(axis=1) / scale
                    for method in e.d.METHODS:
                        torch_table = torch.from_numpy(table)
                        trace = e.d.search.search(method, e.d.search.prior.QueryOracle(torch_table),
                            embedding_by_seed[seed], pca_by_seed[seed], phi_by_seed[seed], initial,
                            torch_table[target_index].clone(), case['search_seed'])
                        case['methods'][method] = {**e.d.search.grade(trace, torch_table, target_index), 'trace': trace}
                        assert len(trace['indices']) == len(set(trace['indices'])) == 600
                        for budget in e.d.BUDGETS:
                            queried = np.asarray(trace['indices'][:budget], dtype=np.int64)
                            prediction = e.d.fit(method, embedding_by_seed[seed], torch.from_numpy(queried),
                                torch.from_numpy(table[queried].copy())).numpy()
                            predicted = np.abs(prediction.astype(np.float64) - target).max(axis=1) / scale
                            partitions = {'distance': e.grouped_labels(predicted, 'distance'), 'embedding': geometry[seed]}
                            metadata = {k: case[k] for k in ('key', 'split', 'seed', 'trial', 'layer', 'prefix_index', 'target_index')}
                            metadata.update({'method': method, 'budget': budget, 'prediction_sha256': e.array_hash(prediction)})
                            local.extend(e.local_diagnostic(predicted, true, queried, partitions, metadata))
                            if budget == 600:
                                empirical.extend(e.run_empirical(true, predicted, queried, partitions['distance'], metadata))
                    cases.append(case)
                    print(f'{case["key"]}: cases={len(cases)}, local={len(local)}, empirical={len(empirical)}', flush=True)
    assert len(new_ids) == 16 and not new_ids & old_ids
    assert len(local) == 6144 and len(empirical) == 3840
    assert all(e.d.digest(Path(path)) == expected for path, expected in frozen.items())
    assert e.d.digest(Path(__file__)) == config['source_sha256']
    e.d.save(out / 'cases.json', cases)
    e.d.save(out / 'local.json', local)
    e.d.save(out / 'empirical.json', empirical)
    e.d.save(out / 'integrity.json', {'frozen_inputs_unchanged': True, 'fresh_target_ids': sorted(new_ids),
        'previous_target_ids': sorted(old_ids), 'target_ids_disjoint': True, 'cases': len(cases),
        'new_search_trajectories': len(cases) * len(e.d.METHODS), 'local_rows': len(local),
        'empirical_runs': len(empirical), 'audit_runs': len(empirical) * len(e.DELTAS),
        'seconds_replay_only': time.perf_counter() - started})
    print(json.dumps(json.loads((out / 'integrity.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    main()
