"""Length extension: same search policies and raw L2<4, one 999-token ciphertext.

Uses stock native rotating caches beyond the 512-token window. Search and
selection functions are imported from the completed short comparison unchanged.
Append-only checkpoints avoid quadratic trace-rewrite overhead at long lengths.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SHORT = ROOT.parent / "ciphertext_hybrid_l2"
sys.path.insert(0, str(SHORT))
spec = importlib.util.spec_from_file_location("short_hybrid_runner", SHORT / "run.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
exp = shared.exp
shared.ROOT = ROOT
shared.OUT = ROOT / "results"
OUT = shared.OUT
SAMPLE = "0037"


def native_prepare(rt, runtime, ids):
    """Stock native model/cache path; old <512 assertion belongs to its short study."""
    runtime.base.layers = runtime.full_layers
    cache = runtime.make_cache("native")
    logits = runtime.model(rt.mx.array([ids]), cache=cache)
    rt.mx.eval(logits)
    return cache, logits[0, -1]


def fingerprint(rt, cache):
    digest = hashlib.sha256()
    for c in cache[:5]:
        digest.update(str((type(c).__name__, c.offset, getattr(c, "_idx", None))).encode())
        digest.update(np.array(c.keys.astype(rt.mx.float32)).tobytes())
        digest.update(np.array(c.values.astype(rt.mx.float32)).tobytes())
    return digest.hexdigest()


def acquire():
    if OUT.exists():
        raise FileExistsError("Observation already exists; preserve the completed acquisition")
    rt, prior, tokenizer, runtime = exp.load_runtime()
    import torch
    source = exp.KVIDENCE / f"datasets/shadow_chat/archive/pilot_100/prompts/encrypted/{SAMPLE}.txt"
    message = source.read_bytes().decode("utf-8")
    rendered, full_ids, prefix, positions, payload = exp.split_input(tokenizer, message)
    assert len(positions) == 999
    ids = torch.tensor(exp.candidate_pool(tokenizer), dtype=torch.long)
    actual = [full_ids[p] for p in positions]
    assert set(actual) <= set(ids.tolist())
    cache, _ = native_prepare(rt, runtime, full_ids)
    target = exp.vector(rt, cache, positions).numpy()
    (OUT / "observed").mkdir(parents=True)
    np.savez_compressed(OUT / "observed/target.npz", kv=target,
                        positions=np.asarray(positions), known_prefix_ids=np.asarray(prefix))
    exp.dump(OUT / "observed/candidate_pool.json", dict(
        token_ids=ids.tolist(), count=len(ids),
        rule="same complete Base64-piece filter as short experiment", selected_without_target_ids=True))
    exp.dump(OUT / "private_eval/truth.json", dict(
        sample_id=SAMPLE, payload_ids=actual, payload=payload,
        full_input_ids=full_ids, rendered_prompt=rendered, source_sha256=exp.digest(source)))
    exp.dump(OUT / "protocol.json", dict(
        sample_id=SAMPLE, sample_selection="closest-to-1000 standalone payload length in existing 100-sample long dataset; tie by ID; no attack-result selection",
        target_tokens=len(positions), rendered_chat_tokens=len(full_ids),
        user_tokens=len(tokenizer.encode(message, add_special_tokens=False)),
        known_prefix_tokens=len(prefix), source_sha256=exp.digest(source),
        epsilon=4.0, strict_less_than=True, metric="raw Euclidean L2 of concatenated K+V, 1024 dimensions",
        kv_dtype=str(cache[4].keys.dtype), layer_1based=5, model=runtime.metadata(),
        initial_hybrid=dict(probability_top=50, random_from_rest=150), seed=20261019,
        no_600_query_limit=True, no_coordinate_gate=True, no_lsb_normalization=True,
        no_rms_normalization=True, candidate_count=len(ids),
        native_long_prefill=True, native_sliding_window=512,
        base_search_script_sha256=exp.digest(SHORT / "run.py"),
        trace_checkpoints="append one JSONL record per position; final pretty JSON outside timed loop"))
    print("ACQUIRED", len(positions), "payload tokens;", len(full_ids), "chat tokens", flush=True)

    # Check candidate branches at, before and after ring wrap. Only the acquisition
    # diagnostic knows true tokens; the subsequent recovery process is isolated.
    probes = {int(positions[0]), 510, 511, 512, 513, 767, 768, int(positions[-1])}
    probes = sorted(p for p in probes if p in positions)
    seq, _ = native_prepare(rt, runtime, prefix)
    checks = []
    for j, position in enumerate(positions):
        assert seq[0].offset == position
        if position in probes:
            tid = full_ids[position]
            true_index = int(torch.where(ids==tid)[0][0])
            others = np.random.default_rng(20261019+position).choice(
                np.delete(np.arange(len(ids)), true_index), 3, replace=False)
            indices = np.r_[true_index, others]
            before = fingerprint(rt, seq)
            oracle = rt.Oracle(runtime, seq, ids, 5, 64)
            batch = oracle.query(indices)
            single = torch.stack([oracle.query([int(i)])[0] for i in indices])
            after = fingerprint(rt, seq)
            assert before == after
            correct_l2 = float(torch.linalg.vector_norm(batch[0]-torch.from_numpy(target[j])).item())
            assert correct_l2 < 4, (position, correct_l2)
            cross = torch.cdist(batch, single)
            assert cross.argmin(-1).tolist() == list(range(4))
            if position >= 512:
                assert seq[0].keys.shape[2] == 512
            checks.append(dict(absolute_position=position, payload_index=j,
                correct_candidate_l2=correct_l2, prefix_cache_unchanged=True,
                batched_vs_single_identification=True,
                sliding_cache_type=type(seq[0]).__name__,
                sliding_cache_capacity=seq[0].keys.shape[2],
                sliding_cache_offset=seq[0].offset, sliding_cache_ring_index=getattr(seq[0], "_idx", None)))
            print("CACHE_CHECK", position, "L2", round(correct_l2, 4), flush=True)
        runtime.base.layers = runtime.full_layers
        result = runtime.model(rt.mx.array([[full_ids[position]]]), cache=seq)
        rt.mx.eval(result)
    exp.dump(OUT / "cache_validation.json", dict(passed=True, probes=checks,
         scope="Independent acquisition diagnostic using correct prefix, not attack input"))


def run(args):
    assert json.loads((OUT / "cache_validation.json").read_text())["passed"]
    # Only checkpoint I/O changes; all proposal, distance and cache code is shared.
    original_dump = exp.dump
    open_counts = {}
    def append_checkpoint(path, value):
        if path.name == "trace.json" and isinstance(value, list):
            path.parent.mkdir(parents=True, exist_ok=True)
            dest = path.with_suffix(".jsonl")
            prior_count = open_counts.get(str(dest), 0)
            if prior_count == 0 and dest.exists():
                raise FileExistsError("Partial trace exists; preserve it before restarting")
            assert len(value) == prior_count+1
            with dest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value[-1], ensure_ascii=False, allow_nan=False)+"\n")
            open_counts[str(dest)] = len(value)
        else:
            original_dump(path, value)
    exp.dump = append_checkpoint
    shared.run(args)
    exp.dump = original_dump
    for method in args.methods:
        source = OUT / method / "trace.jsonl"
        with source.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        original_dump(source.with_suffix(".json"), rows)


def evaluate():
    shared.evaluate()
    path = OUT / "evaluation.json"
    result = json.loads(path.read_text())
    result["sample_id"] = SAMPLE
    result["cache_validation_passed"] = json.loads((OUT / "cache_validation.json").read_text())["passed"]
    exp.dump(path, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("acquire", "run", "evaluate"))
    p.add_argument("--methods", nargs="+", choices=shared.METHODS, default=list(shared.METHODS))
    args = p.parse_args()
    {"acquire":acquire, "run":lambda:run(args), "evaluate":evaluate}[args.stage]()
