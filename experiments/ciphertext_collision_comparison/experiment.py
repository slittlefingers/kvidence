"""One-ciphertext Gemma comparison; acquisition, recovery and evaluation are separate.

Collision+ is a Gemma/MLX port of the reference probability/CPA-threshold policy.
The KV-guided implementation imports the existing 200-query ridge ranker intact.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

ROOT = Path(__file__).resolve().parent
KVIDENCE = ROOT.parents[1]
OLD = ROOT.parent / "kv_guided_search_gemma4_int8"
DATASET = KVIDENCE / "datasets/shadow_chat/provenance/pilot_100_short"
BEFORE = "This is an encrypted message.\n\n"
AFTER = "\n\nHow do I save it?"
BUDGETS = (200, 220, 240, 300, 400, 600)
METHODS = ("collision_plus", "kv_guided_200")
LAYER = 5


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runtime():
    sys.path.insert(0, str(OLD))
    import runtime
    import search
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(OLD / "model", local_files_only=True)
    return runtime, search, tokenizer, runtime.Runtime()


def render(tokenizer, message):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def split_input(tokenizer, message):
    assert message.startswith(BEFORE) and message.endswith(AFTER)
    payload = message[len(BEFORE):-len(AFTER)]
    encoded_text = render(tokenizer, message)
    start = encoded_text.index(payload)
    end = start + len(payload)
    encoded = tokenizer(encoded_text, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                 if a < end and b > start]
    assert positions == list(range(positions[0], positions[-1] + 1))
    prefix = tokenizer.encode(encoded_text[:start], add_special_tokens=False)
    assert prefix == encoded["input_ids"][:positions[0]]
    return encoded_text, encoded["input_ids"], prefix, positions, payload


def vector(rt, cache, positions):
    mx = rt.mx
    c = cache[LAYER - 1]
    indices = mx.array(positions)
    k = c.keys[:, :, indices, :].transpose(0, 2, 1, 3).reshape(len(positions), -1)
    v = c.values[:, :, indices, :].transpose(0, 2, 1, 3).reshape(len(positions), -1)
    return rt.tensor(mx.concatenate([k, v], axis=-1))


def distance(measured, target):
    values = measured.numpy()
    target = target.numpy()
    width = values.shape[-1] // 2
    delta = values - target
    components = np.stack([np.linalg.norm(delta[:, :width], axis=-1),
                           np.linalg.norm(delta[:, width:], axis=-1)], axis=-1)
    nmse = np.mean(delta ** 2, axis=-1) / max(float(np.mean(target ** 2)), 1e-12)
    return components, nmse


def candidate_pool(tokenizer):
    special = set(tokenizer.all_special_ids)
    return [tid for tid in range(len(tokenizer)) if tid not in special and re.fullmatch(
        r"[A-Za-z0-9+/=]+", tokenizer.decode([tid], clean_up_tokenization_spaces=False))]


def calibrate(rt, runtime, tokenizer, ids, output):
    """Independent public chosen inputs, never the target ciphertext or its IDs.

    Reference CPA branch: if sampled true/false supports are separated, midpoint
    of true maximum and false minimum, independently for K and V. Otherwise use
    the same two-Gaussian P_T(tau)*(1-P_F(tau))**(n-1) objective, n=8.
    """
    import torch
    inverse = {int(t): i for i, t in enumerate(ids.tolist())}
    true_d, false_d, probes = [], [], []
    started = time.perf_counter()
    for case in range(3):
        raw = hashlib.shake_256(f"public independent calibration {case}".encode()).digest(108)
        payload = base64.b64encode(raw).decode("ascii")
        message = BEFORE + payload + AFTER
        _, full_ids, _, positions, _ = split_input(tokenizer, message)
        reference, _ = runtime.prepare(full_ids, "native")
        targets = vector(rt, reference, positions)
        for j in np.linspace(0, len(positions)-1, 4, dtype=int):
            prefix, _ = runtime.prepare(full_ids[:positions[j]], "native")
            actual = inverse[full_ids[positions[j]]]
            rng = np.random.default_rng(4000 + case * 100 + int(j))
            pool = np.delete(np.arange(len(ids)), actual)
            query = np.concatenate([[actual], rng.choice(pool, 199, replace=False)])
            oracle = rt.Oracle(runtime, prefix, ids, LAYER, 64)
            measured = oracle.query(query)
            d, _ = distance(measured, targets[j])
            # Include the individual replay shape as well as batched candidates.
            single = oracle.query([actual])
            one, _ = distance(single, targets[j])
            true_d.extend([d[0].tolist(), one[0].tolist()])
            false_d.extend(d[1:].tolist())
            probes.append(dict(case=case, payload_position=int(j), true_distances=d[0].tolist(),
                               individual_true_distances=one[0].tolist(), queries=201))
    truth, false = np.asarray(true_d), np.asarray(false_d)
    components, thresholds = [], []
    for component in range(2):
        t, f = truth[:, component], false[:, component]
        if t.max() < f.min():
            tau = float((t.max() + f.min()) / 2)
            policy = "reference separated-support midpoint"
        else:
            tm, ts, fm, fs = float(t.mean()), float(t.std()), float(f.mean()), float(f.std())
            def cdf(x, mean, std):
                return float(x >= mean) if std == 0 else .5*(1+math.erf((x-mean)/(std*math.sqrt(2))))
            def objective(x):
                return cdf(x, tm, ts) * (1-cdf(x, fm, fs))**7
            # Deterministic bounded numerical maximization of the paper objective.
            grid = np.linspace(min(tm, fm), max(tm, fm), 16385)
            tau = float(max(grid, key=objective))
            policy = "reference two-Gaussian objective, numerical grid, assumed rank 8"
        thresholds.append(tau)
        components.append(dict(component="K" if component == 0 else "V",
                               target_min=float(t.min()), target_max=float(t.max()),
                               target_mean=float(t.mean()), target_std=float(t.std()),
                               others_min=float(f.min()), others_mean=float(f.mean()),
                               others_std=float(f.std()), threshold=tau, policy=policy))
    record = dict(thresholds=thresholds, components=components, probes=probes,
                  target_ciphertext_used=False, calibration_queries=len(probes)*201,
                  source="3 public deterministic random-byte Base64 messages, 4 positions each",
                  false_candidates_per_probe=199, exhaustive_false_distribution=False,
                  reference_acceptance="K < 1.01*tau_K AND V < 1.01*tau_V",
                  seconds=time.perf_counter()-started)
    dump(output / "calibration.json", record)
    return record


def acquire(args):
    if (args.output / "observed/metadata.json").exists():
        raise FileExistsError("Acquisition already exists; choose another output")
    rt, prior, tokenizer, runtime = load_runtime()
    assert runtime.lm.args.layer_types[LAYER-1] == "full_attention"
    source = DATASET / f"prompts/encrypted/{args.sample_id}.txt"
    message = source.read_bytes().decode("utf-8")
    rendered, full_ids, prefix, positions, payload = split_input(tokenizer, message)
    pool = candidate_pool(tokenizer)
    import torch
    ids = torch.tensor(pool, dtype=torch.long)
    actual = [full_ids[i] for i in positions]
    assert set(actual) <= set(pool)
    target_cache, _ = runtime.prepare(full_ids, "native")
    target = vector(rt, target_cache, positions).numpy()
    public = args.output / "observed"
    public.mkdir(parents=True, exist_ok=True)
    np.savez(public / "target.npz", kv=target, known_prefix_ids=np.asarray(prefix),
             positions=np.asarray(positions))
    dump(public / "candidate_pool.json", dict(
        token_ids=pool, count=len(pool), full_vocabulary=len(tokenizer),
        rule="all nonspecial vocabulary IDs whose standalone decoded piece matches [A-Za-z0-9+/=]+",
        constructed_without_target_ids=True, same_for_both_methods=True))
    dump(args.output / "private_eval/truth.json", dict(
        source=str(source), source_sha256=digest(source), sample_id=args.sample_id,
        rendered_prompt=rendered, full_input_ids=full_ids, payload_ids=actual, payload=payload))
    meta = runtime.metadata()
    meta.update(dict(observed_layer_1based=LAYER, actual_observed_kv_dtype=str(target_cache[LAYER-1].keys.dtype),
                     observed_kv_dimensions=target.shape[-1], target_tokens=len(positions),
                     user_tokens=len(tokenizer.encode(message, add_special_tokens=False)),
                     rendered_chat_tokens=len(full_ids), public_prefix_tokens=len(prefix),
                     sample_id=args.sample_id, source_sha256=digest(source),
                     observation="runtime API cache, not physical memory dump or Pixel NPU",
                     payload_restricted_candidate_count=len(pool),
                     tokenizer_sha256=digest(OLD / "model/tokenizer.json"),
                     script_sha256=digest(Path(__file__)),
                     versions={key:importlib.metadata.version(key) for key in
                               ("mlx", "mlx-lm", "torch", "transformers", "numpy")}))
    dump(public / "metadata.json", meta)
    print("ACQUIRED", len(positions), "tokens; candidates", len(pool), flush=True)
    calibration = calibrate(rt, runtime, tokenizer, ids, args.output)
    print("CALIBRATED", calibration["thresholds"], flush=True)


def access_guard(output):
    forbidden = [(output / "private_eval").resolve(), DATASET.resolve()]
    def guard(event, details):
        if event != "open" or not isinstance(details[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(details[0])).resolve()
        if any(path.is_relative_to(parent) for parent in forbidden):
            raise PermissionError("Recovery cannot read ciphertext inputs, keys or evaluation truth")
    sys.addaudithook(guard)


def ordered_probabilities(rt, logits, ids):
    scores = rt.tensor(logits[rt.mx.array(ids.tolist())]).numpy()
    return np.argsort(-scores, kind="stable")


def search_position(rt, prior, runtime, cache, ids, embeddings, target,
                    method, probability_order, seed, thresholds, early_stop):
    import torch
    oracle = rt.Oracle(runtime, cache, ids, LAYER, 64)
    start = time.perf_counter()
    forward_seconds, rank_seconds = 0., 0.
    queried, measured = [], None
    selected, selection_queries = None, None
    stages = []
    rng = np.random.default_rng(seed)
    while len(queried) < 600:
        tick = time.perf_counter()
        if method == "collision_plus":
            chosen = probability_order[len(queried):len(queried)+20].tolist()
        elif not queried:
            chosen = rng.choice(len(ids), 200, replace=False).tolist()
        else:
            rank = prior.proxy_ranking(embeddings, torch.tensor(queried), measured, target)
            chosen = rank[:min(20, 600-len(queried))].tolist()
        rank_seconds += time.perf_counter() - tick
        assert not set(queried) & set(chosen) and len(chosen) == len(set(chosen))
        tick = time.perf_counter()
        new = oracle.query(chosen)
        forward_seconds += time.perf_counter() - tick
        measured = new if measured is None else torch.cat([measured, new])
        queried.extend(chosen)
        d, nmse = distance(measured, target)
        accepted = np.flatnonzero(np.all(d < np.asarray(thresholds)[None, :] * 1.01, axis=-1))
        if len(accepted) and selected is None:
            index = int(accepted[0] if method == "collision_plus" else
                        accepted[np.argmin(nmse[accepted])])
            selected, selection_queries = int(ids[queried[index]]), len(queried)
        stages.append(dict(queries=len(queried), best_nmse=float(nmse.min()),
                           acceptable_candidates=int(len(accepted))))
        if early_stop and selected is not None:
            break
    d, nmse = distance(measured, target)
    best = int(np.argmin(nmse))
    return dict(method=method, queried_token_ids=ids[torch.tensor(queried)].tolist(),
                normalized_mse=nmse.tolist(), component_l2=d.tolist(),
                best_token_id=int(ids[queried[best]]), best_nmse=float(nmse[best]),
                accepted_token_id=selected, first_acceptance_queries=selection_queries,
                queries=len(queried), query_forward_seconds=forward_seconds,
                fit_rank_seconds=rank_seconds, seconds=time.perf_counter()-start,
                early_stopping=early_stop, stages=stages)


def diagnostic(args):
    """Explicit known-correct-prefix control, never presented as a blind attack."""
    rt, prior, tokenizer, runtime = load_runtime()
    import torch
    public = args.output / "observed"
    obs = np.load(public / "target.npz")
    truth = json.loads((args.output / "private_eval/truth.json").read_text())
    ids = torch.tensor(json.loads((public / "candidate_pool.json").read_text())["token_ids"])
    embeddings = runtime.embeddings(ids.tolist())
    calibration = json.loads((args.output / "calibration.json").read_text())
    destination = args.output / "diagnostic"
    if (destination / "run.json").exists():
        raise FileExistsError("Diagnostic already finished")
    rows, orders = [], []
    started = time.perf_counter()
    for j, position in enumerate(obs["positions"]):
        target = torch.from_numpy(obs["kv"][j])
        tick = time.perf_counter()
        cache, logits = runtime.prepare(truth["full_input_ids"][:int(position)], "native")
        prefix_seconds = time.perf_counter()-tick
        order = ordered_probabilities(rt, logits, ids)
        orders.append(order.astype(np.int32))
        row = dict(index=j, position=int(position), known_correct_prefix=True,
                   prefix_full_model_seconds=prefix_seconds, methods={})
        # Alternate order to reduce a systematic first-run timing advantage.
        for method in (METHODS if j % 2 == 0 else METHODS[::-1]):
            row["methods"][method] = search_position(
                rt, prior, runtime, cache, ids, embeddings, target, method, order,
                args.seed+j, calibration["thresholds"], False)
        rows.append(row)
        dump(destination / "trace.json", rows)
        if j % 5 == 0 or j == len(obs["positions"])-1:
            print("DIAGNOSTIC", j+1, "/", len(obs["positions"]),
                  round(time.perf_counter()-started, 1), "sec", flush=True)
        rt.mx.clear_cache()
    np.savez_compressed(destination / "probability_orders.npz", orders=np.stack(orders))
    dump(destination / "run.json", dict(positions=len(rows), known_correct_prefix=True,
         candidate_budget_per_method_per_position=600, initial_random_queries=200,
         proposal_batch=20, ridge=.01, budgets=list(BUDGETS),
         seconds=time.perf_counter()-started, source="original 200-query ridge ranker imported unchanged",
         threshold_used_to_stop_search=False,
         note="Fixed-budget candidate-quality control; does not recover its own prefix"))


def recover(args):
    access_guard(args.output)
    rt, prior, tokenizer, runtime = load_runtime()
    import torch
    public = args.output / "observed"
    obs = np.load(public / "target.npz")
    ids = torch.tensor(json.loads((public / "candidate_pool.json").read_text())["token_ids"])
    embeddings = runtime.embeddings(ids.tolist())
    calibration = json.loads((args.output / "calibration.json").read_text())
    for method in args.methods:
        destination = args.output / "sequential" / method
        if (destination / "run.json").exists():
            raise FileExistsError("Sequential run already finished")
        tick = time.perf_counter()
        cache, logits = runtime.prepare(obs["known_prefix_ids"].tolist(), "native")
        setup_seconds = time.perf_counter()-tick
        rows, recovered = [], []
        started = time.perf_counter()
        advance_seconds = 0.
        for j, position in enumerate(obs["positions"]):
            assert cache[0].offset == int(position)
            target = torch.from_numpy(obs["kv"][j])
            order = ordered_probabilities(rt, logits, ids) if method == "collision_plus" else None
            row = search_position(rt, prior, runtime, cache, ids, embeddings, target,
                                  method, order, args.seed+j, calibration["thresholds"], True)
            row.update(index=j, position=int(position), known_correct_prefix=False)
            rows.append(row)
            dump(destination / "trace.json", rows)
            chosen = row["accepted_token_id"]
            print("SEQUENTIAL", method, j, "queries", row["queries"],
                  "accepted", chosen, "best nmse", round(row["best_nmse"], 7), flush=True)
            if chosen is None:
                break
            recovered.append(chosen)
            tick = time.perf_counter()
            if method == "collision_plus":
                runtime.base.layers = runtime.full_layers
                output = runtime.model(rt.mx.array([[chosen]]), cache=cache)
                rt.mx.eval(output)
                logits = output[0, -1]
            else:
                runtime.base.layers = runtime.full_layers[:LAYER]
                output = runtime.base(rt.mx.array([[chosen]]), cache=cache[:LAYER])
                rt.mx.eval(output)
            advance_seconds += time.perf_counter()-tick
        dump(destination / "run.json", dict(method=method, recovered_token_ids=recovered,
             recovered_text=tokenizer.decode(recovered, clean_up_tokenization_spaces=False),
             attempted_positions=len(rows), target_positions=len(obs["positions"]),
             stopped_unresolved=bool(rows and rows[-1]["accepted_token_id"] is None),
             total_queries=sum(r["queries"] for r in rows),
             setup_full_model_seconds=setup_seconds, advance_seconds=advance_seconds,
             seconds=time.perf_counter()-started,
             known_correct_prefix=False, private_file_access_guard=True,
             source_ciphertext_plaintext_keys_and_true_tokens_read=False,
             threshold_source="independent public chosen-input calibration",
             candidate_budget_per_position=600, initial_random_queries=200 if method==METHODS[1] else 0,
             candidate_logical_batch=20, oracle_microbatch=64,
             proposal_source="model full-depth logits" if method==METHODS[0] else
                             "existing main-embedding-to-KV standardized dual ridge, ridge=.01"))
        rt.mx.clear_cache()


def evaluate(args):
    truth = json.loads((args.output / "private_eval/truth.json").read_text())
    actual = truth["payload_ids"]
    report = dict(sample_id=truth["sample_id"], target_tokens=len(actual),
                  known_prefix_diagnostic={}, sequential={},
                  hardware="Apple MLX GPU; not Pixel NPU", layer_1based=LAYER)
    diag = args.output / "diagnostic"
    if (diag / "run.json").exists():
        rows = json.loads((diag / "trace.json").read_text())
        assert len(rows) == len(actual)
        ids = json.loads((args.output / "observed/candidate_pool.json").read_text())["token_ids"]
        orders = np.load(diag / "probability_orders.npz")["orders"]
        inverse = {tid:i for i,tid in enumerate(ids)}
        ranks = [int(np.flatnonzero(order==inverse[t])[0])+1 for order,t in zip(orders,actual)]
        report["model_probability_rank"] = dict(min=min(ranks), median=float(np.median(ranks)),
                                                 mean=float(np.mean(ranks)), max=max(ranks), ranks=ranks)
        for method in METHODS:
            curves = {}
            for budget in BUDGETS:
                hits, correct = [], []
                for row, tid in zip(rows, actual):
                    trace = row["methods"][method]
                    queries = trace["queried_token_ids"][:budget]
                    best = int(np.argmin(trace["normalized_mse"][:budget]))
                    hits.append(tid in queries)
                    correct.append(queries[best]==tid)
                curves[str(budget)] = dict(target_queried=sum(hits), nearest_kv_correct=sum(correct),
                                            total=len(actual), exact_rate=sum(correct)/len(actual))
            traces = [row["methods"][method] for row in rows]
            report["known_prefix_diagnostic"][method] = dict(
                budgets=curves, queries=sum(r["queries"] for r in traces),
                search_seconds=sum(r["seconds"] for r in traces),
                forward_seconds=sum(r["query_forward_seconds"] for r in traces),
                fit_rank_seconds=sum(r["fit_rank_seconds"] for r in traces),
                known_correct_prefix=True)
    sequential_cases = [("sequential", method) for method in METHODS]
    sequential_cases.append(("sequential_unpruned", "collision_plus"))
    for group, method in sequential_cases:
        path = args.output / group / method
        if not (path / "run.json").exists():
            continue
        run = json.loads((path / "run.json").read_text())
        trace = json.loads((path / "trace.json").read_text())
        recovered = run["recovered_token_ids"]
        matches = [a==b for a,b in zip(recovered,actual)]
        leading = 0
        for value in matches:
            if not value:break
            leading += 1
        run.update(correct_tokens=sum(matches), total_target_tokens=len(actual),
                   exact_leading_prefix_tokens=leading,
                   entire_payload_exact=recovered==actual,
                   correct_by_position=matches,
                   true_candidate_queried_by_attempt=[t in row["queried_token_ids"]
                                                      for t,row in zip(actual,trace)])
        if run["stopped_unresolved"]:
            j = len(recovered)
            run["unresolved_position_1based"] = j+1
            run["unresolved_true_token_id"] = actual[j]
            run["unresolved_true_token_was_queried"] = actual[j] in trace[j]["queried_token_ids"]
        if group == "sequential":
            report["sequential"][method] = run
        else:
            report["secondary_unpruned_probability"] = run
    dump(args.output / "evaluation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("acquire", "diagnostic", "recover", "evaluate"))
    p.add_argument("--sample-id", default="0001")
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = p.parse_args()
    args.output = args.output.resolve()
    {"acquire":acquire, "diagnostic":diagnostic, "recover":recover, "evaluate":evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
