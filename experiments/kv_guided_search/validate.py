#!/usr/bin/env python3
"""Audit saved search traces independently of the candidate-selection code."""
import argparse
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('folders', nargs='*', default=['results', 'replication'])
args = parser.parse_args()
for folder in args.folders:
    path = Path(folder)
    if not path.is_absolute():
        path = root / path
    metrics = json.loads((path / 'metrics.json').read_text())
    candidates = json.loads((path / 'candidates.json').read_text())
    trials = json.loads((path / 'trials.json').read_text())
    count = metrics['candidate_count']
    initial_count = metrics['initial_queries']
    maximum = max(metrics['budgets'])
    assert len(candidates) == len({c['token_id'] for c in candidates}) == count
    assert len(trials) == metrics['trials_per_scenario_per_layer'] * 4
    for trial in trials:
        initial = trial['initial_indices']
        assert len(set(initial)) == len(initial) == initial_count
        original_distances = None
        for name, trace in trial['methods'].items():
            ids, distances = trace['indices'], trace['normalized_mse']
            assert len(ids) == len(set(ids)) == len(distances) == maximum
            assert ids[:initial_count] == initial
            assert all(0 <= i < count for i in ids)
            assert all(math.isfinite(v) and v >= 0 for v in distances)
            if original_distances is None:
                original_distances = distances[:initial_count]
            assert original_distances == distances[:initial_count]
            for budget in metrics['budgets']:
                best = min(range(budget), key=lambda i: distances[i])
                if trial['target_index'] in ids[:budget]:
                    assert ids[best] == trial['target_index']
    print(f'{path.name}: {len(trials)} cases, {count} candidates; shared observations, unique budgets, finite distances and exact identification verified.')
