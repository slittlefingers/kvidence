#!/usr/bin/env python3
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='roneneldan/TinyStories-1M',
    revision='77f1b168e219585646439073245fe87e56b3023e',
    local_dir=Path(__file__).resolve().parent / 'model',
    allow_patterns=['config.json', 'pytorch_model.bin', 'tokenizer.json',
                    'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'],
)
