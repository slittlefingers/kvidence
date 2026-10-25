#!/usr/bin/env python3
"""Toy full-candidate forward matching vs trained KV inverter; not Collision+."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModel
from run import ROOT, Inverter


def collision(model, candidate_ids, target):
    """Greedy full-set search, reusing the selected prefix KV; no ground-truth prefix."""
    past = None
    selected = []
    count = len(candidate_ids)
    for position in range(target.shape[0]):
        expanded = None if past is None else tuple((k.expand(count, -1, -1, -1),
                                                    v.expand(count, -1, -1, -1)) for k, v in past)
        out = model(input_ids=candidate_ids[:, None], past_key_values=expanded, use_cache=True)
        k, v = out.past_key_values[-1]
        features = torch.cat([k[:, :, -1, :].reshape(count, -1),
                              v[:, :, -1, :].reshape(count, -1)], -1)
        best = (features - target[position]).square().sum(-1).argmin().item()
        selected.append(best)
        past = tuple((k[best:best+1].clone(), v[best:best+1].clone()) for k, v in out.past_key_values)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    cache = torch.load(ROOT / 'results/kv_features.pt', weights_only=True)['test']
    records = [json.loads(s) for s in (ROOT / 'results/test.jsonl').read_text().splitlines()][:args.sequences]
    model = AutoModel.from_pretrained(ROOT / 'model', local_files_only=True, attn_implementation='eager').eval()
    model.requires_grad_(False)
    full_layers = model.h
    report = {'description': 'Closed 126-class full forward-matching baseline, not paper Collision+',
              'device': 'cpu', 'threads': 4, 'sequences': len(records), 'tokens_per_sequence': 16,
              'startup_and_disk_io_excluded': True,
              'mlp_training_excluded_from_online_times': True,
              'collision_reuses_predicted_prefix_cache': True,
              'collision_stops_model_at_observed_layer': True, 'layers': {}}
    with torch.inference_mode():
        for layer in (4, 8):
            model.h = torch.nn.ModuleList(list(full_layers)[:layer])
            checkpoint = torch.load(ROOT / f'results/inverter_layer_{layer}.pt', weights_only=True)
            probe = Inverter(checkpoint['feature_width'], checkpoint['classes']).eval()
            probe.load_state_dict(checkpoint['state_dict'])
            ids = torch.tensor(checkpoint['token_ids'])
            index = {tid: i for i, tid in enumerate(ids.tolist())}
            target = cache[layer][:len(records)]
            mean, scale = checkpoint['mean'], checkpoint['scale']
            # Warm both paths; initialization is not counted as online work.
            collision(model, ids, target[0])
            for _ in range(10):
                probe((target[0] - mean) / scale).argmax(-1)
            matching_times = []
            matching_correct = 0
            mlp_correct = 0
            # Measure small inference with repeated calls to reduce clock-resolution noise.
            mlp_times = []
            for i, record in enumerate(records):
                expected = [index[tid] for tid in record['token_ids']]
                start = time.perf_counter()
                guessed = collision(model, ids, target[i])
                matching_times.append(time.perf_counter() - start)
                matching_correct += guessed == expected
                start = time.perf_counter()
                for _ in range(100):
                    inferred = probe((target[i] - mean) / scale).argmax(-1)
                mlp_times.append((time.perf_counter() - start) / 100)
                mlp_correct += inferred.tolist() == expected
            matching = statistics.median(matching_times)
            direct = statistics.median(mlp_times)
            result = {'collision_median_sequence_ms': matching * 1000,
                      'inverter_median_sequence_ms': direct * 1000,
                      'median_latency_ratio': matching / direct,
                      'collision_exact_sequences': matching_correct,
                      'inverter_exact_sequences': mlp_correct,
                      'collision_times_ms': [t * 1000 for t in matching_times],
                      'inverter_times_ms': [t * 1000 for t in mlp_times]}
            assert matching_correct == mlp_correct == len(records), result
            report['layers'][str(layer)] = result
            print(f'Layer {layer}: full matching {matching*1000:.3f} ms, inverter {direct*1000:.3f} ms; ratio {matching/direct:.1f}x', flush=True)
    (ROOT / 'results/benchmark.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
