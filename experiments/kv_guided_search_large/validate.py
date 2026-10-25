#!/usr/bin/env python3
"""Independent checks of query traces and every reported success count."""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
for folder in ('results', 'replication'):
    path = root / folder
    m = json.loads((path / 'metrics.json').read_text())
    candidates = json.loads((path / 'candidates.json').read_text())
    trials = json.loads((path / 'trials.json').read_text())
    assert m['embedding_table_shape'] == [151936, 896]
    assert len(candidates) == len({c['token_id'] for c in candidates}) == m['candidate_count']
    assert len(trials) == m['trials_per_scenario_per_layer'] * 2 * len(m['layers'])
    for r in trials:
        initial = r['initial_indices']
        assert len(initial) == len(set(initial)) == m['initial_queries']
        original = None
        for trace in r['methods'].values():
            ids, distances = trace['indices'], trace['normalized_mse']
            assert len(ids) == len(set(ids)) == len(distances) == max(m['budgets'])
            assert ids[:len(initial)] == initial
            assert all(0 <= i < m['candidate_count'] for i in ids)
            assert all(math.isfinite(v) and v >= 0 for v in distances)
            if original is None:
                original = distances[:len(initial)]
            assert original == distances[:len(initial)]
            for budget in m['budgets']:
                best = min(range(budget), key=lambda i: distances[i])
                if r['target_index'] in ids[:budget]:
                    assert ids[best] == r['target_index']
    for layer, scenarios in m['summary'].items():
        for scenario, methods in scenarios.items():
            cases = [r for r in trials if str(r['layer']) == layer and r['scenario'] == scenario]
            for method, budgets in methods.items():
                for budget, reported in budgets.items():
                    n = int(budget)
                    hits = sum(r['target_index'] in r['methods'][method]['indices'][:n] for r in cases)
                    recovered = 0
                    for r in cases:
                        t = r['methods'][method]
                        best = min(range(n), key=lambda i: t['normalized_mse'][i])
                        recovered += t['indices'][best] == r['target_index']
                    assert reported['trials'] == len(cases)
                    assert reported['hits'] == hits == recovered == reported['exact_recoveries']
                    assert reported['hit_rate'] == hits / len(cases)
    print(f'{folder}: {len(trials)} cases verified, including independent summary recount.')
