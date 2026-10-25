"""Adapter registry; external modules expose a factory as module:factory.

An adapter owns manifest interpretation, native construction and snapshot import.
Search never needs to know model tensor names, mask semantics or KV layout.
Only load trusted local Python plugins: importing a module executes its code.
"""
import json
from importlib import import_module
from typing import Protocol
from .backend import Capabilities, RecoveryBackend
from .config import model_dimensions
from .common import quantization, kv_dtype
from .observation import import_snapshot, save_observation


class ModelAdapter(Protocol):
    def read_manifest(self, path): ...
    def dimensions(self, sections, profile) -> dict: ...
    def capabilities(self, sections, profile, layer) -> Capabilities: ...
    def open(self, sections, profile, layer, output, model_hash, **options) -> RecoveryBackend: ...
    def import_observation(self, snapshot, output, sections, model_hash, prefix, start, count, layer): ...
    def save_fixture(self, output, vectors, positions, prefix, layer, sections, model_hash, source): ...


class LiteRtTensorAdapter:
    """Existing Tensor G5 split Gemma graphs, not a universal LiteRT adapter."""
    def read_manifest(self, path):
        return json.loads(path.read_text())["model_sections"]

    def dimensions(self, sections, profile):
        return model_dimensions(sections, profile["batch_signature"])

    def capabilities(self, sections, profile, layer):
        dims = self.dimensions(sections, profile)
        outputs = sections["tf_lite_prefill_decode"]["signatures"][profile["batch_signature"]]["outputs"]
        names = {x["name"] for x in outputs}
        layers = tuple(sorted(int(n.removeprefix("kv_slice_k_")) for n in names
                              if n.startswith("kv_slice_k_") and n.replace("kv_slice_k_", "kv_slice_v_", 1) in names))
        if layer not in layers:
            raise ValueError("Selected layer is unavailable in this adapter")
        quant = quantization(sections, layer, profile["batch_signature"])
        return Capabilities(**dims, layers=layers, kv_dtype=str(kv_dtype(quant)),
                            kv_width=sum(x["width"] for x in quant))

    def open(self, sections, profile, layer, output, model_hash, **options):
        from .npu import NpuBackend
        self.capabilities(sections, profile, layer)
        return NpuBackend(sections, layer, output, model_hash,
                          device_model=profile["device_model"], batch_signature=profile["batch_signature"],
                          mask_token_id=profile["mask_token_id"], **options)

    def import_observation(self, *args):
        return import_snapshot(*args)

    def save_fixture(self, *args):
        return save_observation(*args)

    def first_v_spec(self, sections, profile):
        # First text block: embedding -> norm -> V projection/norm; no prefix or RoPE.
        # Native output is used instead of approximating the NPU quantized projection.
        return dict(layer=0,quantization=quantization(sections,0,profile["batch_signature"]),
                    prefix=[2],context_independent_by_architecture=True,
                    scope="First text-token V of supported Gemma split graphs; not multimodal embeddings")


ADAPTERS = {"litert_tensor_gemma": LiteRtTensorAdapter}


def get_adapter(name) -> ModelAdapter:
    if name in ADAPTERS:
        factory = ADAPTERS[name]
    elif ":" in name:
        module, symbol = name.split(":", 1)
        factory = getattr(import_module(module), symbol)
    else:
        raise ValueError(f"Unknown adapter: {name}")
    adapter = factory()
    for method in ("read_manifest", "dimensions", "capabilities", "open", "import_observation", "save_fixture"):
        if not callable(getattr(adapter, method, None)):
            raise TypeError(f"Adapter missing method: {method}")
    return adapter
