#!/usr/bin/env python3
"""Independent finite-table scoring and exact combinatorial audit checks."""
import argparse
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

import experiment as e


def combinatorial_controls():
    count = 0
    for n in range(1, 10):
        for delta in (Fraction(1, 2), Fraction(1, 4), Fraction(1, 20), Fraction(1, 100)):
            m = n - (n * delta.numerator // delta.denominator)
            for k in range(n + 1):
                truth = set(range(k))
                failed = total = 0
                for sample in itertools.combinations(range(n), m):
                    observed = set(sample)
                    if observed & truth:
                        observed = set(range(n))  # required full fallback
                    failed += bool(truth - observed)
                    total += 1
                actual = Fraction(failed, total)
                expected = Fraction(math.comb(n - m, k), math.comb(n, k)) if 0 < k <= n - m else Fraction(0)
                assert actual == expected and actual <= delta
                assert abs(e.exact_miss_probability(n, m, k) - float(expected)) < 1e-14
                count += 1
    # Tight nonzero 5% case: one hidden match, exactly one of 20 possible audit
    # omissions loses it. This is a necessary failure case, not a flaky test.
    n, m = 20, 19
    failures = 0
    for omitted in range(n):
        ids = np.array([i for i in range(n) if i != omitted])
        failures += 0 not in ids
    assert Fraction(failures, n) == Fraction(1, 20)
    # Exact-threshold matches and empty calibration groups must not disappear.
    labels = np.array([0, 0, 1, 1])
    radius = e.group_radius(labels, np.array([0., 1., 2., 3.]), 2, sample=np.array([0]))
    assert radius[0] == 0 and np.isinf(radius[1])
    for delta in (0., .05, .01):
        oracle = e.ReplayOracle(np.array([.25, .3, .1, .9]))
        result = e.audit_exclusions(oracle, np.arange(4), .25, delta, 10)
        assert set(oracle.matches(.25)) == {0, 2} and result['queries'] == 4
    try:
        oracle.query(np.array([0]), 'duplicate')
        raise AssertionError('Duplicate query was not rejected')
    except AssertionError as exc:
        assert str(exc) == 'Repeated query'
    # Black-box witness: identical queried answers, different correct sets.
    a = np.ones(10)
    b = a.copy()
    b[-1] = 0.
    assert np.array_equal(a[:9], b[:9])
    assert set(np.flatnonzero(a <= .25)) != set(np.flatnonzero(b <= .25))
    return {'exhaustively_enumerated_audit_settings': count,
            'singleton_5_percent_failure_is_tight': True, 'empty_group_not_excluded': True,
            'exact_threshold_and_zero_delta': True, 'duplicate_queries_rejected': True,
            'black_box_indistinguishability_witness': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = e.ROOT / ('fresh_results' if args.fresh else 'results')
    config = json.loads((folder / 'config.json').read_text())
    source = 'fresh_targets.py' if args.fresh else 'experiment.py'
    protocol = 'NEW_TARGET_PROTOCOL.md' if args.fresh else 'README.md'
    assert e.d.digest(e.ROOT / source) == config['source_sha256']
    assert e.d.digest(e.ROOT / protocol) == config['protocol_sha256']
    for path, expected in config['frozen_inputs_sha256'].items():
        assert e.d.digest(Path(path)) == expected, path
    local = json.loads((folder / 'local.json').read_text())
    empirical = json.loads((folder / 'empirical.json').read_text())
    local_index, empirical_index = {}, {}
    for r in local:
        local_index.setdefault((r['key'], r['method'], r['budget']), []).append(r)
    for r in empirical:
        empirical_index.setdefault((r['key'], r['method']), []).append(r)
    cases_path = folder / 'cases.json' if args.fresh else e.d.SEARCH / 'results/trials.json'
    cases = json.loads(cases_path.read_text())
    if args.fresh:
        previous = {(r['key'], r['method'], r['budget']): r['prediction_sha256'] for r in local}
        token_ids = {}
        for seed in e.d.search.SEEDS:
            with np.load(e.d.DATA / str(seed) / 'features.npz') as raw:
                token_ids[seed] = raw['ids'].copy()
        old_cases = json.loads((e.d.SEARCH / 'results/trials.json').read_text())
        old_ids = {int(token_ids[c['seed']][c['target_index']]) for c in old_cases}
        new_ids = {int(token_ids[c['seed']][c['target_index']]) for c in cases}
        assert len(new_ids) == 16 and not old_ids & new_ids
        for c in cases:
            assert c['target_token_id'] == int(token_ids[c['seed']][c['target_index']])
            expected_initial = torch.randperm(10000, generator=torch.Generator().manual_seed(
                c['seed'] + 14100000 + c['trial']))[:200].tolist()
            assert c['initial_indices'] == expected_initial
    else:
        prior = json.loads((e.d.ROOT / 'results/diagnostic.json').read_text())
        previous = {(r['key'], r['method'], r['budget']): r['prediction_sha256'] for r in prior['records']}
    embeddings, geometry, tables = {}, {}, {}
    nlocal = nempirical = naudits = nfits = 0
    with torch.inference_mode():
        for c in cases:
            seed, pi, layer = c['seed'], c['prefix_index'], c['layer']
            if seed not in embeddings:
                with np.load(e.d.DATA / str(seed) / 'features.npz') as raw:
                    embeddings[seed] = torch.from_numpy(raw['embedding'].copy())
                    geometry[seed] = e.grouped_labels(raw['pca'].astype(np.float64), 'embedding')
            key = seed, pi, layer
            if key not in tables:
                with np.load(e.d.DATA / str(seed) / f'prefix_{pi}.npz') as raw:
                    tables[key] = raw[f'kv_{layer}'].copy()
            table = tables[key]
            target = table[c['target_index']].astype(np.float64)
            scale = max(float(np.linalg.norm(target, ord=np.inf)), 1e-12)
            true = np.linalg.norm(table.astype(np.float64) - target, ord=np.inf, axis=1) / scale
            for method in e.d.METHODS:
                trace = c['methods'][method]['trace']['indices']
                for budget in e.d.BUDGETS:
                    initial = np.asarray(trace[:budget], dtype=np.int64)
                    if method == 'original':
                        pred = e.d.search.ridge_predict(embeddings[seed], torch.from_numpy(initial),
                                                        torch.from_numpy(table[initial].copy()))[0].numpy()
                    else:
                        pred = e.d.search.rbf_predict(embeddings[seed], torch.from_numpy(initial),
                                                      torch.from_numpy(table[initial].copy())).numpy()
                    assert e.array_hash(pred) == previous[(c['key'], method, budget)]
                    scores = (torch.from_numpy(pred) - torch.from_numpy(table[c['target_index']])).square().mean(1)
                    scores[initial] = float('inf')
                    ranking = scores.argsort()
                    if method == 'original':
                        original_ranking = e.d.search.old.proxy_ranking(embeddings[seed], torch.from_numpy(initial),
                            torch.from_numpy(table[initial].copy()), torch.from_numpy(table[c['target_index']]))
                        assert torch.equal(ranking, original_ranking)
                    if budget < 600:
                        assert ranking[:20].tolist() == trace[budget:budget + 20]
                    nfits += 1
                    predicted = np.linalg.norm(pred.astype(np.float64) - target, ord=np.inf, axis=1) / scale
                    partitions = {'distance': e.grouped_labels(predicted, 'distance'), 'embedding': geometry[seed]}
                    for levels in partitions.values():
                        for low, high in zip(e.GROUP_COUNTS, e.GROUP_COUNTS[1:]):
                            assert max(np.bincount(levels[high])) - min(np.bincount(levels[high])) <= 1
                            assert all(len(set(levels[low][levels[high] == group])) == 1 for group in range(high))
                    unseen = np.setdiff1d(np.arange(len(table)), initial)
                    residual = predicted - true
                    pad = 1e-12 * max(1., np.abs(residual).max(), predicted.max(), true.max())
                    for row in local_index[(c['key'], method, budget)]:
                        labels = partitions[row['partition']][row['groups']]
                        assert e.array_hash(labels) == row['partition_sha256']
                        if row['bound'] == 'symmetric':
                            radius = np.array([np.max(np.abs(residual[labels == g])) for g in range(row['groups'])])
                        else:
                            radius = np.array([max(0., np.max(residual[labels == g])) for g in range(row['groups'])])
                        lower = np.maximum(0., predicted - radius[labels] - pad)
                        assert np.all(lower <= true)
                        eps = row['epsilon']
                        rejected = set(unseen[lower[unseen] > eps].tolist())
                        truth = set(np.flatnonzero(true <= eps).tolist())
                        assert not rejected & truth
                        assert len(rejected) == row['excluded']
                        assert len(unseen) - len(rejected) == row['remaining']
                        assert len(table) - len(rejected) == row['hypothetical_total']
                        assert len(truth) == row['truth_count']
                        assert len(truth.intersection(initial.tolist())) == row['queried_matches']
                        assert row['verified_can_stop'] == (row['remaining'] == 0)
                        assert abs(radius[labels].mean() - row['weighted_mean_radius']) < 1e-14
                        nlocal += 1
                    if budget != 600:
                        continue
                    for row in empirical_index[(c['key'], method)]:
                        labels = partitions['distance'][row['groups']]
                        order = np.random.default_rng(row['calibration_seed']).permutation(len(table))
                        initial_mask = np.zeros(len(table), dtype=bool)
                        initial_mask[initial] = True
                        cal = order[~initial_mask[order]][:row['calibration_count']]
                        assert e.array_hash(cal) == row['calibration_indices_sha256']
                        radius = []
                        for g in range(row['groups']):
                            ids = cal[labels[cal] == g]
                            radius.append(max(0., float(residual[ids].max())) if len(ids) else np.inf)
                        radius = np.asarray(radius)
                        pad_cal = 1e-12 * max(1., float(predicted.max()), float(np.abs(residual[cal]).max()))
                        lower = np.maximum(0., predicted - radius[labels] - pad_cal)
                        unqueried = ~initial_mask
                        unqueried[cal] = False
                        eps = row['epsilon']
                        excluded = np.flatnonzero(unqueried & (lower > eps))
                        supplement = np.flatnonzero(unqueried & (lower <= eps))
                        queried = np.concatenate((initial, cal, supplement))
                        assert e.array_hash(queried) == row['query_order_sha256']
                        assert len(queried) == len(np.unique(queried)) == row['queries']
                        assert row['stages'] == {'initial_search': 600, 'calibration': len(cal), 'supplement': len(supplement)}
                        assert row['lower_bound_violations_all_candidates'] == int((lower > true).sum())
                        missed = excluded[true[excluded] <= eps]
                        assert sorted(missed.tolist()) == row['missed_indices']
                        assert len(missed) == row['missed']
                        assert row['complete'] == (len(missed) == 0)
                        assert row['found'] == int((true[queried] <= eps).sum())
                        assert row['truth_count'] == int((true <= eps).sum())
                        assert row['exact_fallback_queries'] == len(table)
                        nempirical += 1
                        for audit in row['audits']:
                            delta = Fraction(str(audit['delta']))
                            leave = len(excluded) * delta.numerator // delta.denominator
                            take = len(excluded) - leave
                            sequence = np.random.default_rng(audit['seed']).permutation(excluded)
                            triggered = bool(np.any(true[sequence[:take]] <= eps))
                            extra = sequence if triggered else sequence[:take]
                            audited = np.concatenate((queried, extra))
                            assert e.array_hash(audited) == audit['query_order_sha256']
                            assert len(audited) == len(np.unique(audited)) == audit['queries']
                            stages = row['stages'].copy()
                            stages['audit'] = take
                            if triggered:
                                stages['fallback'] = leave
                            assert audit['stages'] == stages
                            after = [] if triggered else sequence[take:][true[sequence[take:]] <= eps].tolist()
                            assert sorted(after) == audit['missed_indices']
                            assert len(after) == audit['missed'] and audit['complete'] == (len(after) == 0)
                            k, n = len(missed), len(excluded)
                            exact = Fraction(math.comb(leave, k), math.comb(n, k)) if 0 < k <= leave else Fraction(0)
                            assert exact <= delta
                            assert abs(float(exact) - audit['exact_conditional_failure_probability']) < 1e-14
                            assert audit['fallback_triggered'] == triggered
                            naudits += 1
            print(f'audited {c["key"]}', flush=True)
    factor = 1 if args.fresh else 2
    assert (nfits, nlocal, nempirical, naudits) == (192 * factor, 6144 * factor, 3840 * factor, 7680 * factor)
    # Refinement and switching to one-sided envelopes cannot reduce oracle pruning.
    keyed = {(r['key'], r['method'], r['budget'], r['partition'], r['groups'], r['bound'], r['epsilon']): r for r in local}
    for row in local:
        base = (row['key'], row['method'], row['budget'], row['partition'])
        if row['bound'] == 'symmetric':
            assert keyed[base + (row['groups'], 'one_sided', row['epsilon'])]['excluded'] >= row['excluded']
        index = e.GROUP_COUNTS.index(row['groups'])
        if index:
            parent = keyed[base + (e.GROUP_COUNTS[index - 1], row['bound'], row['epsilon'])]
            assert row['excluded'] >= parent['excluded']
    for path, expected in config['frozen_inputs_sha256'].items():
        assert e.d.digest(Path(path)) == expected, path
    result = {'passed': True, 'cohort': 'fresh' if args.fresh else 'previous',
        'fresh_target_ids_disjoint': args.fresh, 'controls': combinatorial_controls(), 'refitted_snapshots': nfits,
        'independently_checked_local_decisions': nlocal, 'independently_checked_empirical_runs': nempirical,
        'independently_checked_audits': naudits, 'frozen_inputs_unchanged': True,
        'oracle_local_false_exclusions': 0, 'oracle_refinement_monotone': True,
        'empirical_omissions_preserved_in_results': sum(r['missed'] > 0 for r in empirical),
        'audit_omissions_preserved_in_results': sum(a['missed'] > 0 for r in empirical for a in r['audits']),
        'every_analytic_audit_failure_probability_within_delta': True,
        'source_sha256': e.d.digest(Path(__file__)),
        'local_sha256': e.d.digest(folder / 'local.json'), 'empirical_sha256': e.d.digest(folder / 'empirical.json')}
    e.d.save(folder / 'validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
