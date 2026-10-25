#!/usr/bin/env python3
"""Recompute interval decisions independently from frozen inputs and predictors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import diagnose as d


def controls():
    # Exact predictor: one unqueried match, two exclusions, no unresolved item.
    table = np.array([[1., 0.], [1., .1], [1., .5], [1., .9]], dtype=np.float32)
    truth, _, bounds, _ = d.intervals(table, table, 0)
    result = d.classify(truth, np.array([0]), bounds, .25)
    for b in result['bounds'].values():
        assert b['excluded_unqueried'] == 2 and b['accepted_unqueried'] == 1
        assert b['unresolved'] == 0 and b['oracle_intervals_classify_all']
        assert b['remaining_to_query_for_verified_set'] == 1
        assert not b['oracle_verified_set_can_stop']
    # Adversarial surrogate: true match has huge predicted distance. The honest
    # max residual must prevent it from being rejected.
    bad = table.copy()
    bad[1] = [100., -100.]
    truth, _, bounds, _ = d.intervals(bad, table, 0)
    result = d.classify(truth, np.array([0]), bounds, .25)
    assert all(b['false_exclusions'] == 0 for b in result['bounds'].values())
    assert all(b['remaining_to_query_for_verified_set'] >= 1 for b in result['bounds'].values())
    # Exact-on-threshold is a match (<=), and an outward interval may leave it
    # unresolved. Directly observing every row removes that ambiguity.
    true_distance = np.array([0., .25, .25001])
    fixed_bounds = {'test': (np.array([0., .25, .25001]), np.array([0., .25, .25001]))}
    result = d.classify(true_distance, np.array([0]), fixed_bounds, .25)['bounds']['test']
    assert result['accepted_unqueried'] == 1 and result['excluded_unqueried'] == 1
    result = d.classify(true_distance, np.arange(3), fixed_bounds, .25)['bounds']['test']
    assert result['oracle_verified_set_can_stop']
    return {'perfect_proxy_can_classify_all': True,
            'interval_matching_is_not_actual_query_verification': True,
            'honest_radius_protects_badly_predicted_match': True,
            'threshold_equality_and_exhaustive_fallback': True}


def main():
    torch.set_num_threads(4)
    obj = json.loads((d.ROOT / 'results/diagnostic.json').read_text())
    config = obj['config']
    assert d.digest(d.ROOT / 'diagnose.py') == config['source_sha256']
    assert d.digest(d.ROOT / 'README.md') == config['protocol_sha256']
    for path, expected in config['frozen_inputs_sha256'].items():
        assert d.digest(Path(path)) == expected, path
    cases = {c['key']: c for c in json.loads((d.SEARCH / 'results/trials.json').read_text())}
    embeddings, tables = {}, {}
    checks = snapshots = 0
    with torch.inference_mode():
        for record in obj['records']:
            c = cases[record['key']]
            seed, pi, layer = c['seed'], c['prefix_index'], c['layer']
            if seed not in embeddings:
                with np.load(d.DATA / str(seed) / 'features.npz') as raw:
                    embeddings[seed] = torch.from_numpy(raw['embedding'].copy())
            key = seed, pi, layer
            if key not in tables:
                with np.load(d.DATA / str(seed) / f'prefix_{pi}.npz') as raw:
                    tables[key] = raw[f'kv_{layer}'].copy()
            table = tables[key]
            method, budget = record['method'], record['budget']
            indices = np.array(c['methods'][method]['trace']['indices'][:budget])
            assert len(np.unique(indices)) == budget
            xidx = torch.from_numpy(indices)
            measured = torch.from_numpy(table[indices].copy())
            if method == 'original':
                predicted = d.search.ridge_predict(embeddings[seed], xidx, measured)[0].numpy()
            else:
                predicted = d.search.rbf_predict(embeddings[seed], xidx, measured).numpy()
            assert hashlib.sha256(predicted.tobytes()).hexdigest() == record['prediction_sha256']
            y = table[c['target_index']].astype(np.float64)
            scale = max(np.linalg.norm(y, ord=np.inf), 1e-12)
            true_distance = np.linalg.norm(table.astype(np.float64) - y, ord=np.inf, axis=1) / scale
            a = np.abs(predicted.astype(np.float64) - y) / scale
            predicted_distance = a.max(axis=1)
            error = np.abs(table.astype(np.float64) - predicted.astype(np.float64)) / scale
            bj = error.max(axis=0)
            bd = np.linalg.norm(true_distance - predicted_distance, ord=np.inf)
            bv = np.max(error)
            pad = 1e-12 * max(1., bv, true_distance.max(), predicted_distance.max())
            np.testing.assert_allclose(record['error']['coordinate_radius'], bj, rtol=0, atol=1e-14)
            assert abs(record['error']['distance_uniform_radius'] - bd) < 1e-14
            assert abs(record['error']['kv_uniform_radius'] - bv) < 1e-14
            lower = np.stack((np.maximum(0., predicted_distance - (bd + pad)),
                              np.maximum(0., predicted_distance - (bv + pad)),
                              np.maximum(0., a - (bj + pad)).max(axis=1)))
            upper = np.stack((predicted_distance + (bd + pad), predicted_distance + (bv + pad),
                              (a + (bj + pad)).max(axis=1)))
            assert np.all(lower <= true_distance[None, :])
            assert np.all(upper >= true_distance[None, :])
            unseen = np.setdiff1d(np.arange(len(table)), indices)
            for evaluation in record['thresholds']:
                eps = evaluation['epsilon']
                truth = set(np.flatnonzero(true_distance <= eps))
                found = truth.intersection(indices.tolist())
                assert len(truth) == evaluation['truth_count']
                assert len(found) == evaluation['queried_matches']
                assert len(truth - found) == evaluation['unqueried_matches']
                for i, bound in enumerate(d.BOUNDS):
                    excluded = set(unseen[lower[i, unseen] > eps].tolist())
                    accepted = set(unseen[upper[i, unseen] <= eps].tolist())
                    ambiguous = set(unseen.tolist()) - excluded - accepted
                    assert not (excluded & truth) and accepted <= truth
                    assert not (accepted & excluded)
                    cell = evaluation['bounds'][bound]
                    assert cell['excluded_unqueried'] == len(excluded)
                    assert cell['accepted_unqueried'] == len(accepted)
                    assert cell['unresolved'] == len(ambiguous)
                    assert cell['remaining_to_query_for_verified_set'] == len(unseen) - len(excluded)
                    assert cell['hypothetical_total_after_fallback'] == len(table) - len(excluded)
                    assert cell['oracle_intervals_classify_all'] == (not ambiguous)
                    assert cell['oracle_verified_set_can_stop'] == (len(unseen) == len(excluded))
                    assert cell['false_exclusions'] == cell['false_inclusions'] == 0
                    # Hypothetical fallback must recover precisely ALL matches.
                    fallback = set(unseen.tolist()) - excluded
                    assert found | (fallback & truth) == truth
                    checks += 1
            if record['key'] == d.EXAMPLE:
                with np.load(d.ROOT / 'results' / f'example_{method}_{budget}.npz') as raw:
                    np.testing.assert_array_equal(raw['queried_indices'], indices)
                    np.testing.assert_allclose(raw['true_distance'], true_distance, rtol=0, atol=1e-14)
                    for i, bound in enumerate(d.BOUNDS):
                        np.testing.assert_allclose(raw[bound + '_lower'], lower[i], rtol=0, atol=1e-14)
                        np.testing.assert_allclose(raw[bound + '_upper'], upper[i], rtol=0, atol=1e-14)
            snapshots += 1
    assert snapshots == 384 and checks == 2304
    assert d.aggregate(obj['records']) == obj['summary']
    result = {'passed': True, 'controls': controls(), 'independently_refitted_snapshots': snapshots,
              'independently_checked_classifications': checks, 'all_intervals_cover_stored_table': True,
              'false_exclusions': 0, 'false_inclusions': 0, 'hypothetical_fallback_always_recovers_all': True,
              'frozen_inputs_unchanged': True, 'source_sha256': d.digest(Path(__file__)),
              'diagnostic_json_sha256': d.digest(d.ROOT / 'results/diagnostic.json')}
    d.save(d.ROOT / 'results/validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
