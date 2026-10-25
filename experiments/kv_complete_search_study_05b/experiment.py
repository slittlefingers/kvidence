#!/usr/bin/env python3
"""Frozen-trajectory local diagnostics and independently audited complete search."""
import hashlib
import importlib.util
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exclusion_diagnostic', ROOT.parent / 'kv_exclusion_diagnostic_05b/diagnose.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
GROUP_COUNTS = (1, 8, 32, 128)
CALIBRATIONS = (200, 600, 1200)
REPEATS = tuple(range(5))
DELTAS = (.05, .01)


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def grouped_labels(features, kind):
    """Nested balanced partitions using ONLY available features/predictions."""
    n = len(features)
    if kind == 'distance':
        groups = [np.argsort(features, kind='stable')]
    else:
        groups = [np.arange(n)]
    result = {1: np.zeros(n, dtype=np.int64)}
    while len(groups) < max(GROUP_COUNTS):
        next_groups = []
        for ids in groups:
            if kind == 'embedding':
                x = features[ids]
                dim = int(x.var(axis=0).argmax())
                ids = ids[np.argsort(x[:, dim], kind='stable')]
            mid = len(ids) // 2
            next_groups.extend((ids[:mid], ids[mid:]))
        groups = next_groups
        if len(groups) in GROUP_COUNTS:
            labels = np.empty(n, dtype=np.int64)
            for group, ids in enumerate(groups):
                labels[ids] = group
            result[len(groups)] = labels
    return result


def group_radius(labels, residual, count, sample=None, symmetric=False):
    if sample is None:
        sample = np.arange(len(labels))
    values = np.abs(residual[sample]) if symmetric else np.maximum(0., residual[sample])
    radius = np.full(count, -np.inf)
    np.maximum.at(radius, labels[sample], values)
    # Empty calibration groups are never excluded.
    radius[np.isneginf(radius)] = np.inf
    return radius


def local_diagnostic(predicted, true, queried, partitions, metadata):
    rows = []
    residual = predicted - true
    pad = 1e-12 * max(1., float(np.abs(residual).max()), float(predicted.max()), float(true.max()))
    unseen = np.ones(len(true), dtype=bool)
    unseen[queried] = False
    for kind, levels in partitions.items():
        for groups, labels in levels.items():
            for symmetric in (True, False):
                radius = group_radius(labels, residual, groups, symmetric=symmetric)
                lower = np.maximum(0., predicted - radius[labels] - pad)
                assert np.all(lower <= true)
                for eps in d.EPSILONS:
                    truth = true <= eps
                    rejected = unseen & (lower > eps)
                    missed = int((truth & unseen).sum())
                    assert not np.any(rejected & truth)
                    rows.append({**metadata, 'partition': kind, 'groups': groups,
                        'bound': 'symmetric' if symmetric else 'one_sided', 'epsilon': eps,
                        'truth_count': int(truth.sum()), 'queried_matches': int((truth & ~unseen).sum()),
                        'unqueried_matches': missed, 'excluded': int(rejected.sum()),
                        'remaining': int((unseen & ~rejected).sum()),
                        'hypothetical_total': len(true) - int(rejected.sum()),
                        'weighted_mean_radius': float(radius[labels].mean()),
                        'max_group_radius': float(radius.max()), 'false_exclusions': 0,
                        'verified_can_stop': not bool((unseen & ~rejected).any()),
                        'partition_sha256': array_hash(labels)})
    return rows


class ReplayOracle:
    """One query represents one candidate KV evaluation; answers are hidden."""
    def __init__(self, answers):
        self._answers = answers
        self._queried = np.zeros(len(answers), dtype=bool)
        self.indices = []
        self.stages = {}

    def query(self, indices, stage):
        ids = np.asarray(indices, dtype=np.int64)
        assert ids.ndim == 1 and len(np.unique(ids)) == len(ids)
        assert np.all((ids >= 0) & (ids < len(self._answers)))
        assert not self._queried[ids].any(), 'Repeated query'
        self._queried[ids] = True
        self.indices.extend(ids.tolist())
        self.stages[stage] = self.stages.get(stage, 0) + len(ids)
        return self._answers[ids].copy()

    def clone(self):
        other = ReplayOracle(self._answers)
        other._queried = self._queried.copy()
        other.indices = self.indices.copy()
        other.stages = self.stages.copy()
        return other

    def matches(self, epsilon):
        ids = np.asarray(self.indices, dtype=np.int64)
        return ids[self._answers[ids] <= epsilon]


def base_search(oracle, initial, predicted, labels, calibration_count, repeat_seed, epsilon):
    """No hidden distances enter the learner except through oracle.query()."""
    n, groups = len(predicted), int(labels.max()) + 1
    oracle.query(initial, 'initial_search')
    unseen = np.ones(n, dtype=bool)
    unseen[initial] = False
    order = np.random.default_rng(repeat_seed).permutation(n)
    calibration = order[unseen[order]][:calibration_count]
    measured = oracle.query(calibration, 'calibration')
    sampled_residual = predicted[calibration] - measured
    radius = np.full(groups, -np.inf)
    np.maximum.at(radius, labels[calibration], np.maximum(0., sampled_residual))
    radius[np.isneginf(radius)] = np.inf
    # Only observed magnitudes affect this numerical margin.
    pad = 1e-12 * max(1., float(predicted.max()), float(np.abs(sampled_residual).max()))
    lower = np.maximum(0., predicted - radius[labels] - pad)
    unseen[calibration] = False
    excluded = np.flatnonzero(unseen & (lower > epsilon))
    supplement = np.flatnonzero(unseen & (lower <= epsilon))
    oracle.query(supplement, 'supplement')
    assert len(oracle.indices) + len(excluded) == n
    return excluded, radius, lower, calibration


def audit_exclusions(oracle, excluded, epsilon, delta, seed):
    """Single fixed audit, independent of the proposal. Match => full fallback."""
    size = len(excluded)
    fraction = Fraction(str(delta))
    leave = (size * fraction.numerator) // fraction.denominator
    take = size - leave
    order = np.random.default_rng(seed).permutation(excluded)
    values = oracle.query(order[:take], 'audit')
    triggered = bool(np.any(values <= epsilon))
    if triggered:
        oracle.query(order[take:], 'fallback')
    return {'audit_population': size, 'audit_sample': take, 'fallback_triggered': triggered,
            'worst_case_failure_bound': leave / size if size else 0.,
            'queries': len(oracle.indices), 'stages': oracle.stages.copy(),
            'query_order_sha256': array_hash(np.asarray(oracle.indices, dtype=np.int64))}


def exact_miss_probability(population, sample, matches):
    if matches == 0 or population == 0:
        return 0.
    # Equivalent to choose(N-k,r)/choose(N,r), numerically stable for small k.
    if population - sample < matches:
        return 0.
    return math.prod((population - sample - j) / (population - j) for j in range(matches))


def run_empirical(true, predicted, initial, partition_levels, metadata):
    records = []
    for repeat in REPEATS:
        seed = metadata['seed'] + 5000000 + metadata['trial'] * 1009 + metadata['layer'] * 19 + repeat * 100003
        if metadata['split'] == 'confirmation':
            seed += 7000000
        for count in CALIBRATIONS:
            for groups in (1, 32):
                labels = partition_levels[groups]
                for epsilon in d.EPSILONS:
                    oracle = ReplayOracle(true)
                    excluded, radius, lower, calibration = base_search(oracle, initial, predicted,
                        labels, count, seed, epsilon)
                    truth = set(np.flatnonzero(true <= epsilon).tolist())
                    found = set(oracle.matches(epsilon).tolist())
                    missed = truth - found
                    assert found <= truth and missed <= set(excluded.tolist())
                    row = {**metadata, 'groups': groups, 'epsilon': epsilon, 'calibration_count': count,
                        'repeat': repeat, 'calibration_seed': seed, 'queries': len(oracle.indices),
                        'truth_count': len(truth), 'found': len(found), 'missed': len(missed),
                        'complete': not missed, 'excluded': len(excluded),
                        'empty_calibration_groups': int(np.isinf(radius).sum()),
                        'lower_bound_violations_all_candidates': int((lower > true).sum()),
                        'stages': oracle.stages.copy(), 'calibration_indices_sha256': array_hash(calibration),
                        'query_order_sha256': array_hash(np.asarray(oracle.indices, dtype=np.int64)),
                        'missed_indices': sorted(missed), 'audits': []}
                    for delta in DELTAS:
                        audit_seed = seed + 30000001 + int(delta * 10000)
                        audited = oracle.clone()
                        audit = audit_exclusions(audited, excluded, epsilon, delta, audit_seed)
                        after = truth - set(audited.matches(epsilon).tolist())
                        audit.update({'delta': delta, 'seed': audit_seed, 'missed': len(after),
                            'complete': not after, 'missed_indices': sorted(after),
                            'exact_conditional_failure_probability': exact_miss_probability(
                                len(excluded), audit['audit_sample'], len(missed))})
                        assert audit['exact_conditional_failure_probability'] <= delta + 1e-15
                        assert audit['worst_case_failure_bound'] <= delta + 1e-15
                        assert audit['queries'] >= 600 + math.ceil((1 - delta) * (len(true) - 600) - 1e-10)
                        if audit['fallback_triggered']:
                            assert audit['queries'] == len(true) and not after
                        row['audits'].append(audit)
                    exact = oracle.clone()
                    exact.query(excluded, 'exact_fallback')
                    assert len(exact.indices) == len(true)
                    assert set(exact.matches(epsilon).tolist()) == truth
                    row['exact_fallback_queries'] = len(exact.indices)
                    records.append(row)
    return records


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    prior_path = d.ROOT / 'results/diagnostic.json'
    previous = json.loads(prior_path.read_text())
    frozen = previous['config']['frozen_inputs_sha256'].copy()
    frozen[str(prior_path)] = d.digest(prior_path)
    frozen[str(d.ROOT / 'diagnose.py')] = previous['config']['source_sha256']
    for path, expected in frozen.items():
        assert d.digest(Path(path)) == expected, path
    prior_snapshots = {(r['key'], r['method'], r['budget']): r for r in previous['records']}
    config = {'model': previous['config']['model'], 'revision': previous['config']['revision'],
        'groups': GROUP_COUNTS, 'budgets': d.BUDGETS, 'methods': d.METHODS, 'epsilons': d.EPSILONS,
        'calibrations': CALIBRATIONS, 'repeats': REPEATS, 'deltas': DELTAS, 'primary_groups': 32,
        'primary_calibration_count': 600, 'primary_epsilon': .25, 'primary_layer': 12,
        'old_search_unchanged': True, 'new_model_forwards': 0, 'hidden_table_replay': True,
        'query_costs_include_all_stages': True, 'frozen_inputs_sha256': frozen,
        'source_sha256': d.digest(Path(__file__)), 'protocol_sha256': d.digest(ROOT / 'README.md')}
    out = ROOT / 'results'
    d.save(out / 'config.json', config)
    cases = json.loads((d.SEARCH / 'results/trials.json').read_text())
    embedding, geometry, tables = {}, {}, {}
    local_rows, empirical_rows = [], []
    refit_parity = 0
    with torch.inference_mode():
        for c in cases:
            seed, pi, layer = c['seed'], c['prefix_index'], c['layer']
            if seed not in embedding:
                with np.load(d.DATA / str(seed) / 'features.npz') as raw:
                    embedding[seed] = torch.from_numpy(raw['embedding'].copy())
                    geometry[seed] = grouped_labels(raw['pca'].astype(np.float64), 'embedding')
            key = seed, pi, layer
            if key not in tables:
                with np.load(d.DATA / str(seed) / f'prefix_{pi}.npz') as raw:
                    tables[key] = raw[f'kv_{layer}'].copy()
            table = tables[key]
            target = table[c['target_index']].astype(np.float64)
            scale = max(float(np.abs(target).max()), 1e-12)
            true = np.abs(table.astype(np.float64) - target).max(axis=1) / scale
            for method in d.METHODS:
                trace = c['methods'][method]['trace']['indices']
                for budget in d.BUDGETS:
                    initial = np.asarray(trace[:budget], dtype=np.int64)
                    prediction = d.fit(method, embedding[seed], torch.from_numpy(initial),
                                       torch.from_numpy(table[initial].copy())).numpy()
                    assert array_hash(prediction) == prior_snapshots[(c['key'], method, budget)]['prediction_sha256']
                    refit_parity += 1
                    predicted = np.abs(prediction.astype(np.float64) - target).max(axis=1) / scale
                    partitions = {'distance': grouped_labels(predicted, 'distance'), 'embedding': geometry[seed]}
                    metadata = {k: c[k] for k in ('key', 'split', 'seed', 'trial', 'layer', 'prefix_index', 'target_index')}
                    metadata.update({'method': method, 'budget': budget})
                    local_rows.extend(local_diagnostic(predicted, true, initial, partitions, metadata))
                    if budget == 600:
                        empirical_rows.extend(run_empirical(true, predicted, initial, partitions['distance'], metadata))
            print(f'{c["key"]}: local={len(local_rows)}, empirical={len(empirical_rows)}', flush=True)
    assert len(local_rows) == 12288 and len(empirical_rows) == 7680
    assert all(d.digest(Path(path)) == expected for path, expected in frozen.items())
    assert config['source_sha256'] == d.digest(Path(__file__))
    d.save(out / 'local.json', local_rows)
    d.save(out / 'empirical.json', empirical_rows)
    d.save(out / 'integrity.json', {'frozen_inputs_unchanged': True, 'refit_prediction_hash_matches': refit_parity,
        'local_rows': len(local_rows), 'empirical_runs': len(empirical_rows),
        'audit_runs': len(empirical_rows) * len(DELTAS), 'exact_fallback_runs': len(empirical_rows),
        'seconds_replay_diagnostic_only': time.perf_counter() - started,
        'source_sha256': config['source_sha256']})
    print(json.dumps(json.loads((out / 'integrity.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    main()
