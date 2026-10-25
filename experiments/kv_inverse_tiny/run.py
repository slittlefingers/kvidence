#!/usr/bin/env python3
"""Train KV -> token probes on a frozen pretrained model, excluding layer 1."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'roneneldan/TinyStories-1M'
REVISION = '77f1b168e219585646439073245fe87e56b3023e'
WORDS = '''red blue green black white brown small big little happy sad good bad young old
cat dog bird fish horse cow duck pig bear lion tiger rabbit mouse fox wolf sheep
boy girl man woman child mother father friend king queen baby sister brother
house room door window garden tree flower grass river lake hill road city village
sun moon star sky rain snow wind cloud day night morning evening light dark
book story toy ball game food bread milk water apple orange banana cake cookie
run walk jump play read write sing dance look find help make take give love
warm cold soft hard fast slow clean quiet loud round long short sweet kind'''.split()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def vocabulary(tokenizer):
    words = []
    for word in WORDS:
        if len(tokenizer.encode(' ' + word, add_special_tokens=False)) == 1:
            words.append(word)
        if len(words) == 64:
            break
    assert len(words) == 64
    letters = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    digits = list('0123456789')
    terms = words + letters + digits
    ids = [tokenizer.encode(' ' + term, add_special_tokens=False) for term in terms]
    assert all(len(row) == 1 for row in ids)
    ids = [row[0] for row in ids]
    assert len(ids) == len(set(ids)) == 126
    return terms, ids, words, letters, digits


def dataset(tokenizer, terms, ids, words, letters, digits, count, seed, seen,
            length=16, kinds=('mixed', 'alnum')):
    rng = random.Random(seed)
    by_term = dict(zip(terms, ids))
    records = []
    while len(records) < count:
        kind = kinds[len(records) % len(kinds)]
        if kind == 'mixed':
            row = [rng.choice(words if rng.random() < .8 else digits) for _ in range(length)]
        elif kind == 'alnum':
            row = rng.choices(letters + digits, k=length)
        else:
            row = rng.choices(digits, k=length)
        text = ''.join(' ' + word for word in row)
        if text in seen:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        # Canonical tokenizer output, not hand-created impossible token sequences.
        assert token_ids == [by_term[word] for word in row]
        seen.add(text)
        records.append({'kind': kind, 'text': text, 'token_ids': token_ids})
    return records


def features(model, records, layers, batch=64):
    output = {layer: [] for layer in layers}
    with torch.inference_mode():
        for start in range(0, len(records), batch):
            tokens = torch.tensor([r['token_ids'] for r in records[start:start + batch]])
            result = model(input_ids=tokens, use_cache=True, return_dict=True)
            for layer in layers:
                key, value = result.past_key_values[layer - 1]
                # [batch, heads, position, head_dim] -> [batch, position, 2*hidden]
                k = key.permute(0, 2, 1, 3).reshape(tokens.shape[0], tokens.shape[1], -1)
                v = value.permute(0, 2, 1, 3).reshape_as(k)
                output[layer].append(torch.cat([k, v], dim=-1).clone())
    return {layer: torch.cat(parts) for layer, parts in output.items()}


class Inverter(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, classes))

    def forward(self, x):
        return self.net(x)


def predict(probe, x, mean, scale):
    with torch.inference_mode():
        logits = [probe((chunk - mean) / scale) for chunk in x.flatten(0, 1).split(2048)]
    return torch.cat(logits).argmax(-1).reshape(x.shape[:2])


def metrics(pred, labels, records):
    result = {}
    for kind in ['all'] + sorted({r['kind'] for r in records}):
        rows = list(range(len(records))) if kind == 'all' else [i for i, r in enumerate(records) if r['kind'] == kind]
        equal = pred[rows] == labels[rows]
        result[kind] = {'sequences': len(rows), 'tokens': equal.numel(),
                        'token_accuracy': equal.float().mean().item(),
                        'exact_sequence_accuracy': equal.all(-1).float().mean().item()}
    return result


def labels_for(records, ids):
    mapping = {token_id: label for label, token_id in enumerate(ids)}
    return torch.tensor([[mapping[token_id] for token_id in row['token_ids']] for row in records])


def train_probe(x, labels, vx, vl, classes, epochs, seed):
    torch.manual_seed(seed)
    mean = x.flatten(0, 1).mean(0)
    scale = x.flatten(0, 1).std(0).clamp_min(1e-5)
    tx = ((x.flatten(0, 1) - mean) / scale).contiguous()
    ty = labels.flatten()
    probe = Inverter(tx.shape[-1], classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    curve = []
    best_accuracy, best_epoch, best_state = -1, -1, None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        probe.train()
        order = torch.randperm(len(tx))
        total = 0.0
        for index in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(tx[index]), ty[index])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        probe.eval()
        validation = (predict(probe, vx, mean, scale) == vl).float().mean().item()
        curve.append({'epoch': epoch, 'train_loss': total / len(tx), 'validation_token_accuracy': validation})
        if validation > best_accuracy:
            best_accuracy, best_epoch = validation, epoch
            best_state = {name: tensor.detach().clone() for name, tensor in probe.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f'  epoch={epoch:02d} loss={total / len(tx):.4f} validation={validation:.4%}', flush=True)
    probe.load_state_dict(best_state)
    return probe, mean, scale, curve, best_epoch, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train-sequences', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    model = AutoModel.from_pretrained(ROOT / 'model', local_files_only=True,
                                      attn_implementation='eager').eval()
    model.requires_grad_(False)
    terms, ids, words, letters, digits = vocabulary(tokenizer)
    seen = set()
    train = dataset(tokenizer, terms, ids, words, letters, digits, args.train_sequences, args.seed, seen)
    validation = dataset(tokenizer, terms, ids, words, letters, digits, 256, args.seed + 1, seen)
    test = dataset(tokenizer, terms, ids, words, letters, digits, 384, args.seed + 2, seen,
                   kinds=('mixed', 'alnum', 'digits'))
    long_test = dataset(tokenizer, terms, ids, words, letters, digits, 96, args.seed + 3, seen,
                        length=32, kinds=('mixed', 'alnum', 'digits'))
    # Validation/test contain no exact training sequence. Same token classes are intentional.
    sets = [set(r['text'] for r in records) for records in (train, validation, test, long_test)]
    assert all(not sets[i] & sets[j] for i in range(4) for j in range(i))
    for name, records in [('train', train), ('validation', validation), ('test', test), ('long_test', long_test)]:
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    write_json(args.output / 'vocabulary.json', [{'class': i, 'token_id': tid, 'decoded': tokenizer.decode([tid])}
                                               for i, tid in enumerate(ids)])
    layers = [4, 8]
    print(f'Frozen pretrained parameters={sum(p.numel() for p in model.parameters()):,}; classes={len(ids)}; layers={layers}', flush=True)
    extraction_start = time.perf_counter()
    all_features = {}
    for name, records in [('train', train), ('validation', validation), ('test', test), ('long_test', long_test)]:
        all_features[name] = features(model, records, layers)
        print(f'Extracted {name}: {len(records)} sequences', flush=True)
    extraction_seconds = time.perf_counter() - extraction_start
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    train_labels, val_labels, test_labels, long_labels = [labels_for(r, ids) for r in (train, validation, test, long_test)]
    counts = torch.bincount(train_labels.flatten(), minlength=len(ids))
    assert (counts > 0).all()
    majority = torch.full_like(test_labels, counts.argmax())
    report = {'model_id': MODEL_ID, 'model_revision': REVISION,
              'pretrained_parameters': sum(p.numel() for p in model.parameters()),
              'original_vocab_size': model.config.vocab_size, 'candidate_classes': len(ids),
              'seed': args.seed, 'device': 'cpu', 'dtype': 'float32',
              'sequence_length': 16, 'long_test_length': 32,
              'train_sequences': len(train), 'validation_sequences': len(validation),
              'test_sequences': len(test), 'long_test_sequences': len(long_test),
              'all_splits_sequence_disjoint': True, 'model_frozen': True,
              'kv_extraction_seconds': extraction_seconds,
              'majority_baseline': metrics(majority, test_labels, test), 'layers': {}}
    examples = []
    for layer in layers:
        print(f'Training layer {layer} KV inverter (no layer-1 input)', flush=True)
        x, vx, ex, lx = [all_features[name][layer] for name in ('train', 'validation', 'test', 'long_test')]
        probe, mean, scale, curve, epoch, seconds = train_probe(x, train_labels, vx, val_labels, len(ids), args.epochs, args.seed + layer)
        pred = predict(probe, ex, mean, scale)
        long_pred = predict(probe, lx, mean, scale)
        # Destroy position/example correspondence while keeping marginal KV distribution.
        generator = torch.Generator().manual_seed(args.seed + 100 + layer)
        order = torch.randperm(ex.shape[0] * ex.shape[1], generator=generator)
        shuffled = ex.flatten(0, 1)[order].reshape_as(ex)
        shuffled_pred = predict(probe, shuffled, mean, scale)
        summary = {'feature_width': x.shape[-1], 'inverter_parameters': sum(p.numel() for p in probe.parameters()),
                   'best_epoch_by_validation': epoch, 'train_seconds': seconds,
                   'test': metrics(pred, test_labels, test),
                   'long_test': metrics(long_pred, long_labels, long_test),
                   'shuffled_kv_control': metrics(shuffled_pred, test_labels, test)}
        report['layers'][str(layer)] = summary
        torch.save({'state_dict': probe.state_dict(), 'mean': mean, 'scale': scale, 'layer': layer,
                    'token_ids': ids, 'model_id': MODEL_ID, 'model_revision': REVISION,
                    'feature_width': x.shape[-1], 'classes': len(ids)}, args.output / f'inverter_layer_{layer}.pt')
        write_json(args.output / f'curve_layer_{layer}.json', curve)
        for kind in ('mixed', 'alnum', 'digits'):
            rows = [i for i, r in enumerate(test) if r['kind'] == kind]
            for i in rows[:3]:
                decoded = tokenizer.decode([ids[label] for label in pred[i].tolist()])
                examples.append({'layer': layer, 'kind': kind, 'original': test[i]['text'],
                                 'reconstructed': decoded, 'exact': bool((pred[i] == test_labels[i]).all())})
        print(json.dumps(summary['test']), flush=True)
    report['model_weights_sha256'] = hashlib.sha256((ROOT / 'model' / 'pytorch_model.bin').read_bytes()).hexdigest()
    write_json(args.output / 'metrics.json', report)
    write_json(args.output / 'examples.json', examples)
    torch.save({name: tensor for name, tensor in all_features.items()}, args.output / 'kv_features.pt')
    print(f'Saved checkpoints, metrics, datasets and examples to {args.output}', flush=True)


if __name__ == '__main__':
    main()
