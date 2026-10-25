#!/usr/bin/env python3
"""Decode saved KV features without loading the language-model weights or labels."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from run import Inverter, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', type=int, choices=[4, 8], default=8)
    parser.add_argument('--split', choices=['train', 'validation', 'test', 'long_test'], default='test')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.results / f'inverter_layer_{args.layer}.pt', weights_only=True)
    tensors = torch.load(args.results / 'kv_features.pt', weights_only=True)
    features = tensors[args.split][args.layer][args.index]
    probe = Inverter(checkpoint['feature_width'], checkpoint['classes']).eval()
    probe.load_state_dict(checkpoint['state_dict'])
    with torch.inference_mode():
        labels = probe((features - checkpoint['mean']) / checkpoint['scale']).argmax(-1)
    recovered = [checkpoint['token_ids'][label] for label in labels.tolist()]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    print(tokenizer.decode(recovered))


if __name__ == '__main__':
    main()
