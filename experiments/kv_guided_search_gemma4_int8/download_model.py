#!/usr/bin/env python3
import hashlib
from huggingface_hub import snapshot_download
from runtime import ROOT, MODEL, MODEL_ID, REVISION

snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL,
                  allow_patterns=['*.safetensors', '*.json', '*.jinja', '*.model'])
expected = {
    'model-00001-of-00002.safetensors': '0e82359bc2f0effdad15168d05a2ae3d46e72cb3296b5658303a2f78eee9926b',
    'model-00002-of-00002.safetensors': '797c7786d137de2b77826ae838ee366eab5a9479dfca91b345f71b7e99d71cfe',
}
for name, digest in expected.items():
    with (MODEL / name).open('rb') as f:
        observed = hashlib.file_digest(f, 'sha256').hexdigest()
    assert observed == digest, (name, observed)
    print('Verified', name, observed)
