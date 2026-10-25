"""Create minimal recovery input; exclude all nonpublic prefix/cache/labels."""
import gzip
import json
import struct
from pathlib import Path
import numpy as np
from .common import dump, digest, quantization, distance_values, kv_dtype


def save_observation(output, codes, positions, prefix_ids, layer, sections, model_sha256, source):
    return save_canonical_observation(output, codes, positions, prefix_ids, layer,
        quantization(sections, layer), model_sha256, source)


def save_canonical_observation(output, codes, positions, prefix_ids, layer, quant, model_sha256, source):
    """Portable adapter output: ordered K+V rows with explicit scale metadata.

    Adapters with other native dtypes may convert to original-unit float32,
    with scale=1 and zero_point=0, documenting the conversion in source.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    codes = np.asarray(codes, dtype=kv_dtype(quant))
    positions = np.asarray(positions, dtype=np.int64)
    independent = layer == 0
    if len(positions)==0 or (independent and (positions.ndim!=1 or positions[0]<0 or np.any(np.diff(positions)<=0))):
        raise ValueError("First-layer targets require ordered, unique nonnegative positions")
    if not independent and not np.array_equal(positions, np.arange(len(prefix_ids), len(prefix_ids) + len(positions))):
        raise ValueError("First version requires contiguous targets immediately after the public prefix")
    if codes.shape != (len(positions), sum(q["width"] for q in quant)):
        raise ValueError("Invalid observed KV shape")
    np.savez(output / "observed.npz", codes=codes, positions=positions,
             public_prefix_ids=np.asarray(prefix_ids, dtype=np.int64))
    dump(output / "metadata.json", dict(schema=1, layer_index=layer, layer_1based=layer+1,
        quantization=quant, model_sha256=model_sha256, source=source,
        storage="Native KV codes; recovery run records the distance space separately",
        observed_sha256=digest(output / "observed.npz")))


def import_snapshot(snapshot, output, sections, model_sha256, prefix_ids, start, count, layer=4):
    if start<0 or (layer!=0 and start != len(prefix_ids)) or count <= 0:
        raise ValueError("start must equal the number of explicitly public prefix IDs; count must be positive")
    with gzip.open(snapshot, "rb") as f:
        header = f.read(16)
        if len(header) != 16 or header[:8] != b"GHOSTKV1":
            raise ValueError("Unsupported snapshot")
        size = struct.unpack("<Q", header[8:])[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("Unexpected snapshot header length")
        layout = json.loads(f.read(size)); payload = f.read()
    if layout["ringbuffer"] or layout["byte_order"] != "little-endian":
        raise ValueError("Rotating or big-endian snapshots need an explicit slot map")
    if start + count > layout["kv_positions"]:
        raise ValueError("Target extends beyond active KV positions")
    buffers = {b["name"]: b for b in layout["buffers"]}
    vectors = []
    for part in ("k", "v"):
        b = buffers[f"kv_cache_{part}_{layer}"]
        if b["dtype"] not in ("int8","float32") or (b["local"] and layer!=0):
            raise ValueError("Requires a float32/INT8 global producer layer")
        dtype=np.dtype("<f4" if b["dtype"]=="float32" else "i1")
        shape = b["shape"]; strides = b["strides"]; offset = b["payload_offset"]
        if (b['local'] and layout['kv_positions']>shape[b['sequence_axis']]) or start+count>shape[b['sequence_axis']]:
            raise ValueError("First local cache may have shifted/overwritten history; an explicit slot map is required")
        end = offset + (sum((d-1)*s for d,s in zip(shape,strides)) + 1)*dtype.itemsize
        if offset < 0 or end > len(payload):
            raise ValueError("Snapshot buffer exceeds payload")
        a = np.ndarray(shape, dtype=dtype, buffer=payload, offset=offset, strides=tuple(s*dtype.itemsize for s in strides))
        a = np.moveaxis(a, b["sequence_axis"], 0)[start:start+count]
        vectors.append(a.reshape(count, -1).copy())
    save_observation(output, np.concatenate(vectors, axis=1), np.arange(start,start+count), prefix_ids,
        layer, sections, model_sha256, dict(kind="GHOSTKV1 runtime capture", snapshot_sha256=digest(snapshot),
        note="Only selected target-layer rows imported; no target-prefix cache supplied to recovery"))


def load_observation(path, distance_space="original"):
    path = Path(path); meta = json.loads((path / "metadata.json").read_text())
    if digest(path / "observed.npz") != meta["observed_sha256"]:
        raise ValueError("Observation integrity mismatch")
    with np.load(path / "observed.npz", allow_pickle=False) as f:
        codes, positions, prefix = f["codes"].copy(), f["positions"].copy(), f["public_prefix_ids"].copy()
    if meta['layer_index']==0:
        valid = (len(positions)>0 and positions.ndim==1 and len(positions)==len(codes)
                 and positions[0]>=0 and np.all(np.diff(positions)>0))
    else:
        valid = len(prefix)>0 and len(positions)>0 and np.array_equal(positions,np.arange(len(prefix),len(prefix)+len(codes)))
    if not valid:
        raise ValueError("A nonempty public prefix and contiguous target positions are required")
    if codes.dtype != kv_dtype(meta["quantization"]) or codes.ndim != 2:
        raise ValueError("Unsupported KV data or context length")
    return meta, codes, distance_values(codes, meta["quantization"], distance_space), positions, prefix.tolist()
