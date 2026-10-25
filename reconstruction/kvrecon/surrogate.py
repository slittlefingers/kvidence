"""Fit and score in the caller's distance space (integer KV defaults to LSB codes).

Predictions are continuous; never round them to INT8 or use them for acceptance.
"""
import importlib.util
from types import SimpleNamespace
import numpy as np
from .common import EXPERIMENTS, digest


class ExistingRidge:
    def __init__(self, embeddings, vocab_size, embedding_dim=None):
        import torch
        import mlx.core as mx
        path=EXPERIMENTS/"ciphertext_full_vocabulary/stable_surrogate.py"
        spec=importlib.util.spec_from_file_location("kvrecon_existing_ridge",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        table=np.load(embeddings,allow_pickle=False)
        if table.ndim!=2 or table.shape[0]!=vocab_size or table.shape[1]<1 or (embedding_dim is not None and table.shape[1]!=embedding_dim) or table.dtype!=np.float32 or not np.isfinite(table).all():
            raise ValueError(f"Expected a complete float32 embedding table with {vocab_size} original rows and matching model dimension")
        self.torch=torch
        self.engine=module.StableSurrogate(SimpleNamespace(mx=mx),torch.from_numpy(table))
        self.metadata=dict(source=str(path),source_sha256=digest(path),embedding_sha256=digest(embeddings),
            shape=list(table.shape),proposal_only=True,
            note="Configured embedding asset is proposal-only, not asserted bit-identical to device embedding quantization. Only real NPU KV is accepted.")

    def scores(self, queried, measured, target, distances):
        tensor=lambda a:self.torch.from_numpy(np.asarray(a,dtype=np.float32))
        return self.engine.fit(queried,tensor(measured),tensor(target),distances)["scores"]
