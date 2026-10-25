#!/usr/bin/env python3
"""Download the exact checkpoint used by this experiment and verify its hash."""
import hashlib
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
snapshot_download(
    repo_id='Qwen/Qwen2.5-0.5B',
    revision='060db6499f32faf8b98477b0a26969ef7d8b9987',
    local_dir=root / 'model',
    allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                    'tokenizer_config.json', 'generation_config.json', 'vocab.json', 'merges.txt'],
)
with (root / 'model/model.safetensors').open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
assert digest == '88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342'
print('Checkpoint SHA-256 verified:', digest)
