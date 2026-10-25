"""Shared KV8 reader and observed-grid LSB L2 metric for this isolated rerun."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "ciphertext_collision_comparison"
spec = importlib.util.spec_from_file_location("previous_cipher_experiment", PREVIOUS / "experiment.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
LAYER, GROUP = 5, 64
RADII = (1, 2, 3, 4)
dump = previous.dump


def snapshot(rt, cache, positions):
    """Read actual uint8 codes and stored BF16 affine metadata, without BF16 re-rounding."""
    result = {"q": [], "s": [], "b": []}
    for triple in (cache.keys, cache.values):
        selected = [a[:, :, positions, :] for a in triple]
        batch, heads, count, _ = selected[0].shape
        arrays = [np.array(a.astype(rt.mx.float32) if a.dtype == rt.mx.bfloat16 else a,
                           copy=True).transpose(0, 2, 1, 3).reshape(batch*count, -1)
                  for a in selected]
        packed, scales, biases = arrays
        q = ((packed[..., None] >> np.arange(0, 32, 8, dtype=np.uint32)) & 255)
        q = q.reshape(batch*count, -1).astype(np.uint8)
        for key, a in zip(("q", "s", "b"), (q, scales, biases)):
            result[key].append(a)
    result = {key: np.concatenate(parts, axis=-1) for key, parts in result.items()}
    assert result["q"].shape[-1] == 1024
    assert result["s"].shape[-1] * GROUP == 1024
    return result


def values(raw):
    return (raw["q"].astype(np.float64) * np.repeat(raw["s"].astype(np.float64), GROUP, axis=-1)
            + np.repeat(raw["b"].astype(np.float64), GROUP, axis=-1))


def lsb_l2(candidate, target):
    scale = np.repeat(np.abs(target["s"].astype(np.float64)), GROUP, axis=-1)
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Observed quantization step is zero/nonfinite; LSB metric undefined")
    normalized = (values(candidate)-values(target))/scale
    # Full L2 over K+V, no RMS division, coordinate gate, clipping or rounding.
    return np.sqrt(np.sum(normalized*normalized, axis=-1))


class Oracle:
    def __init__(self, rt, runtime, cache, ids, batch=64):
        self.rt, self.runtime, self.cache, self.ids = rt, runtime, cache, ids
        self.batch = batch
        self.count = 0

    def query(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        self.runtime.base.layers = self.runtime.full_layers[:LAYER]
        parts = []
        for start in range(0, len(indices), self.batch):
            chunk = indices[start:start+self.batch]
            branch = self.rt.fork_cache(self.cache, len(chunk), LAYER)
            h = self.runtime.base(self.rt.mx.array(self.ids[chunk].tolist())[:, None], cache=branch)
            self.rt.mx.eval(h)
            parts.append(snapshot(self.rt, branch[LAYER-1], [branch[LAYER-1].offset-1]))
            self.count += len(chunk)
        return {key: np.concatenate([part[key] for part in parts]) for key in ("q", "s", "b")}


def load():
    return previous.load_runtime()


def acquisition():
    out = ROOT / "kv8_unit_probe"
    if (out / "observed/metadata.json").exists():
        raise FileExistsError("Unit probe already exists; preserve the recorded check")
    rt, prior, tokenizer, runtime = load()
    source = previous.DATASET / "prompts/encrypted/0001.txt"
    message = source.read_bytes().decode("utf-8")
    rendered, full_ids, prefix, positions, payload = previous.split_input(tokenizer, message)
    ids = np.asarray(previous.candidate_pool(tokenizer), dtype=np.int64)
    cache, _ = runtime.prepare(full_ids, "int8")
    raw = snapshot(rt, cache[LAYER-1], positions)
    (out / "observed").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "observed/target.npz", **raw,
                        known_prefix_ids=np.asarray(prefix), positions=np.asarray(positions))
    dump(out / "observed/candidate_pool.json", dict(token_ids=ids.tolist(), count=len(ids),
         rule="same full Base64-piece vocabulary filter as previous comparison"))
    dump(out / "private_eval/truth.json", dict(full_input_ids=full_ids,
         payload_ids=[full_ids[p] for p in positions], payload=payload, rendered_prompt=rendered))
    dump(out / "observed/metadata.json", dict(
        sample_id="0001", model=runtime.metadata(), layer_1based=LAYER,
        kv_storage="runtime KV8 affine, uint8 codes packed in uint32; BF16 scales/biases",
        kv_bits=8, group_size=GROUP, dimensions=1024, positions=len(positions),
        source_sha256=previous.digest(source), metric="sqrt(sum(((candidate_real-observed_real)/abs(observed_scale))**2))",
        no_division_by_sqrt_dimensions=True, no_coordinate_max_gate=True,
        stopping_radii=list(RADII), query_cap_600=False,
        source_reader="runtime cache API; Apple MLX GPU; not physical dump or NPU"))
    # Numerical replay diagnostic; these labels will not enter the recovery worker.
    known, _ = runtime.prepare(full_ids[:positions[0]], "int8")
    true_index = int(np.flatnonzero(ids==full_ids[positions[0]])[0])
    target = {key:a[0:1] for key,a in raw.items()}
    probes = []
    for batch in (1, 20, 64):
        others = np.random.default_rng(1000+batch).choice(
            np.delete(np.arange(len(ids)), true_index), batch-1, replace=False)
        measured = Oracle(rt, runtime, known, ids, batch).query(np.r_[true_index, others])
        d = lsb_l2({key:a[:1] for key,a in measured.items()}, target)[0]
        probes.append(dict(batch=batch, true_candidate_lsb_l2=float(d)))
    dump(out / "private_eval/replay_diagnostic.json", dict(
        known_correct_prefix=True, first_true_token_id=int(ids[true_index]), probes=probes,
        note="Numerical diagnostic only; not accessible to the recovery worker"))
    print(json.dumps(dict(target_tokens=len(positions), candidate_count=len(ids),
                          first_true_candidate_replays=probes), indent=2), flush=True)


if __name__ == "__main__":
    acquisition()
