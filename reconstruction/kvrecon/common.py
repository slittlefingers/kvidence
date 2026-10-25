import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
EXPERIMENTS = ROOT.parent / "experiments"
MODEL = WORKSPACE / "annoymous/models/gemma-4-E2B-it_Google_Tensor_G5.litertlm"
DEFAULT_MANIFEST = EXPERIMENTS / "npu_candidate_batch/inputs/probe.json"
DEFAULT_EMBEDDINGS = EXPERIMENTS / "embedding_hnsw/assets/embeddings.npy"
DEFAULT_TOKENIZER = EXPERIMENTS / "kv_guided_search_gemma4_int8/model/tokenizer.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(4 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def quantization(sections, layer, signature="decode"):
    outputs = {x["name"]: x for x in sections["tf_lite_prefill_decode"]["signatures"][signature]["outputs"]}
    parts = []
    for part in ("k", "v"):
        m = outputs[f"kv_slice_{part}_{layer}"]
        width = math.prod(m["shape"][:2]) * m["shape"][3 if part == "k" else 2]
        if m["type"] == 0:
            parts.append(dict(width=width,scale=1.,zero_point=0,dtype="float32"))
        elif m["type"] == 9 and len(m["scale"])==1 and len(m["zero_point"])==1:
            parts.append(dict(width=width,scale=m["scale"][0],zero_point=m["zero_point"][0],dtype="int8"))
        else:
            raise ValueError("Expected float32 or per-tensor INT8 KV")
    return parts


def kv_dtype(quant):
    import numpy as np
    types={p.get("dtype","int8") for p in quant}
    if len(types)!=1 or not types.issubset({"int8","float32"}):
        raise ValueError("Unsupported or mixed KV storage types")
    return np.dtype("<f4" if "float32" in types else "i1")


def dequantize(codes, quant):
    import numpy as np
    if codes.shape[-1] != sum(p["width"] for p in quant):
        raise ValueError("KV vector width mismatch")
    scale = np.concatenate([np.full(p["width"], p["scale"], dtype=np.float64) for p in quant])
    zero = np.concatenate([np.full(p["width"], p["zero_point"], dtype=np.float64) for p in quant])
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid quantization scale")
    return (np.asarray(codes, dtype=np.float64) - zero) * scale


def resolve_distance_space(space, dtype):
    """Integer KV defaults to code/LSB units; floats have no fixed LSB grid."""
    import numpy as np
    integer = np.dtype(dtype).kind in "iu"
    if space not in ("auto", "lsb", "original"):
        raise ValueError("Unknown KV distance space")
    if space == "auto":
        return "lsb" if integer else "original"
    if space == "lsb" and not integer:
        raise ValueError("LSB distance requires native integer KV, not floating/dequantized values")
    return space


def code_values(codes):
    """Widen before subtraction: INT8 differences can exceed the INT8 range."""
    import numpy as np
    codes = np.asarray(codes)
    resolve_distance_space("lsb", codes.dtype)
    return codes.astype(np.float64)


def distance_values(codes, quant, space="auto"):
    import numpy as np
    if np.asarray(codes).shape[-1] != sum(q["width"] for q in quant):
        raise ValueError("KV vector width mismatch")
    space = resolve_distance_space(space, kv_dtype(quant))
    return code_values(codes) if space == "lsb" else dequantize(codes, quant)


def validate_quantization_grid(observed, candidate):
    """Code differences only have meaning on corresponding identical grids."""
    def normalized(parts):
        return [dict(q, dtype=q.get("dtype", "int8")) for q in parts]
    if normalized(observed) != normalized(candidate):
        raise ValueError("Observed and candidate KV quantization grids differ")
