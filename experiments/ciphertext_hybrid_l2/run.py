"""One sample, fixed raw KV L2 < 4, no 600-query cap.

Compare probability ordering with top-50 probability + 150 random initialization,
followed by the existing ridge-guided proposals. Original experiment is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from common import previous as exp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_raw_l2_4"
METHODS = ("collision_plus_l2", "hybrid_50_150")
EPSILON = 4.0
SEED = 20261019


def prepare():
    if OUT.exists():
        raise FileExistsError("Prepared output exists; use run/evaluate stages")
    source = exp.ROOT / "results"
    shutil.copytree(source / "observed", OUT / "observed")
    (OUT / "private_eval").mkdir()
    shutil.copy2(source / "private_eval/truth.json", OUT / "private_eval/truth.json")
    exp.dump(OUT / "protocol.json", dict(
        sample_id="0001", reused_observation=str(source / "observed/target.npz"),
        reused_observation_sha256=exp.digest(source / "observed/target.npz"),
        metric="Euclidean L2 of raw concatenated K+V, 1024 coordinates",
        formula="sqrt(sum((candidate_KV_j-observed_KV_j)**2)) < 4",
        epsilon=EPSILON, strict_less_than=True, kv_dtype="BF16", weights="8-bit affine",
        no_lsb_normalization=True, no_rms_normalization=True, no_coordinate_gate=True,
        no_CPA_threshold=True, no_600_query_limit=True,
        termination="first measured match, or complete finite candidate pool exhausted",
        initial_hybrid=dict(probability_top=50, uniform_without_replacement_from_rest=150),
        proposal_batch=20, seed=SEED, candidate_forward_layer_1based=5,
        model_probability_logits="full 35-layer original model, both methods",
        scope="One existing 109-token ciphertext; own recovered prefix; local MLX GPU"))
    print("Prepared identical BF16 observation and fixed raw L2 < 4 protocol", flush=True)


def install_guard():
    forbidden = [OUT / "private_eval", exp.ROOT / "results/private_eval",
                 exp.KVIDENCE / "datasets", ROOT / "results/private_eval"]
    forbidden = [p.resolve() for p in forbidden]
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, Path)):
            return
        import os
        p = Path(os.fsdecode(args[0])).resolve()
        if any(p.is_relative_to(parent) for parent in forbidden):
            raise PermissionError("Recovery cannot open source prompts, keys or evaluation labels")
    sys.addaudithook(guard)


def rank_candidates(rt, prior, embeddings, queried, measured, target):
    """Existing dual ridge; mathematically equivalent primal form if n > d."""
    import torch
    observed = torch.tensor(queried, dtype=torch.long)
    if len(queried) <= embeddings.shape[1]:
        return prior.proxy_ranking(embeddings, observed, measured, target).numpy()
    x = embeddings[observed]
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-4)
    x = (x-mean)/scale
    ym, ys = measured.mean(0), measured.std(0).clamp_min(1e-4)
    y = (measured-ym)/ys
    xm, ymean = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ymean
    matrix = xc.T @ xc + torch.eye(x.shape[1])*(.01*len(x))
    coef = torch.linalg.solve(matrix, xc.T @ yc)
    gpu = lambda t: rt.mx.array(t.contiguous().numpy())
    all_x = (gpu(embeddings)-gpu(mean))/gpu(scale)-gpu(xm)
    predicted = (all_x @ gpu(coef)+gpu(ymean))*gpu(ys)+gpu(ym)
    d = rt.mx.mean((predicted-gpu(target)).square(), axis=-1)
    d[rt.mx.array(queried)] = float("inf")
    return np.array(rt.mx.argsort(d))


def search(rt, prior, runtime, cache, ids, emb, target, order, method, index):
    import torch
    oracle = rt.Oracle(runtime, cache, ids, 5, 64)
    queried, measured = [], None
    all_l2 = []
    target64 = target.numpy().astype(np.float64)
    ticks = dict(forward=0., ranking=0.)
    selected, selected_vector, selected_distance = None, None, None
    stages = []
    started = time.perf_counter()
    while len(queried) < len(ids):
        tick = time.perf_counter()
        if method == "collision_plus_l2":
            chosen = order[len(queried):len(queried)+20].tolist()
        elif not queried:
            top = order[:50]
            remaining = np.setdiff1d(np.arange(len(ids)), top)
            random_part = np.random.default_rng(SEED+index).choice(remaining, 150, replace=False)
            chosen = np.concatenate([top, random_part]).tolist()
        else:
            ranked = rank_candidates(rt, prior, emb, queried, measured, target)
            chosen = ranked[:min(20, len(ids)-len(queried))].tolist()
        ticks["ranking"] += time.perf_counter()-tick
        assert chosen and not (set(chosen) & set(queried))
        tick = time.perf_counter()
        new = oracle.query(chosen)
        ticks["forward"] += time.perf_counter()-tick
        distances = np.linalg.norm(new.numpy().astype(np.float64)-target64, axis=-1)
        assert np.isfinite(distances).all()
        queried.extend(chosen)
        all_l2.extend(distances.tolist())
        measured = new if measured is None else torch.cat([measured, new])
        passing = np.flatnonzero(distances < EPSILON)
        stages.append(dict(queries=len(queried), batch_min_raw_l2=float(distances.min()),
                           passing_in_batch=int(len(passing))))
        if len(passing):
            if method == "collision_plus_l2":
                best = int(passing[0])
            else:
                best = int(passing[np.argmin(distances[passing])])
            selected = int(ids[int(chosen[best])])
            selected_vector = new[best].tolist()
            selected_distance = float(distances[best])
            break
    return dict(index=index, method=method, queries=len(queried),
        queried_token_ids=ids[torch.tensor(queried)].tolist(), raw_l2_distances=all_l2,
        accepted_token_id=selected, accepted_raw_l2=selected_distance,
        accepted_kv=selected_vector, stages=stages,
        ranking_seconds=ticks["ranking"], forward_seconds=ticks["forward"],
        search_seconds=time.perf_counter()-started,
        finite_pool_exhausted=selected is None and len(queried)==len(ids))


def run(args):
    install_guard()
    rt, prior, tokenizer, runtime = exp.load_runtime()
    import torch
    observation = np.load(OUT / "observed/target.npz")
    ids = torch.tensor(json.loads((OUT / "observed/candidate_pool.json").read_text())["token_ids"])
    embeddings = runtime.embeddings(ids.tolist())
    for method in args.methods:
        dest = OUT / method
        if (dest / "run.json").exists():
            raise FileExistsError(f"Completed run exists: {method}")
        cache, logits = runtime.prepare(observation["known_prefix_ids"].tolist(), "native")
        recovered, traces = [], []
        started = time.perf_counter()
        advance_seconds = 0.
        for j, position in enumerate(observation["positions"]):
            assert cache[0].offset == int(position)
            tick = time.perf_counter()
            order = exp.ordered_probabilities(rt, logits, ids)
            order_seconds = time.perf_counter()-tick
            row = search(rt, prior, runtime, cache, ids, embeddings,
                         torch.from_numpy(observation["kv"][j]), order, method, j)
            row.update(position=int(position), probability_ranking_seconds=order_seconds,
                       probability_top50_token_ids=ids[torch.tensor(order[:50])].tolist())
            traces.append(row)
            exp.dump(dest / "trace.json", traces)
            print(method, j+1, "/", len(observation["positions"]), "queries", row["queries"],
                  "accepted", row["accepted_token_id"], "L2", row["accepted_raw_l2"],
                  "elapsed", round(time.perf_counter()-started, 2), flush=True)
            if row["accepted_token_id"] is None:
                break
            chosen = row["accepted_token_id"]
            recovered.append(chosen)
            tick = time.perf_counter()
            # Hybrid needs the same full-depth probability logits at each position.
            runtime.base.layers = runtime.full_layers
            output = runtime.model(rt.mx.array([[chosen]]), cache=cache)
            rt.mx.eval(output)
            logits = output[0, -1]
            advance_seconds += time.perf_counter()-tick
        exp.dump(dest / "run.json", dict(
            method=method, recovered_token_ids=recovered,
            recovered_text=tokenizer.decode(recovered, clean_up_tokenization_spaces=False),
            target_tokens=len(observation["positions"]), attempted_positions=len(traces),
            epsilon=EPSILON, acceptance="raw K+V L2 < 4 only",
            query_limit_600=False, maximum_distinct_candidates=len(ids),
            total_queries=sum(t["queries"] for t in traces), seconds=time.perf_counter()-started,
            forward_seconds=sum(t["forward_seconds"] for t in traces),
            fitting_and_ranking_seconds=sum(t["ranking_seconds"]+t["probability_ranking_seconds"] for t in traces),
            advance_seconds=advance_seconds, known_correct_prefix=False,
            private_file_access_guard=True, true_previous_tokens_used=False,
            stopped_unresolved=bool(traces and traces[-1]["accepted_token_id"] is None)))
        rt.mx.clear_cache()


def evaluate():
    truth = json.loads((OUT / "private_eval/truth.json").read_text())
    actual = truth["payload_ids"]
    observation = np.load(OUT / "observed/target.npz")
    pool = np.asarray(json.loads((OUT / "observed/candidate_pool.json").read_text())["token_ids"])
    results = {}
    for method in METHODS:
        dest = OUT / method
        run = json.loads((dest / "run.json").read_text())
        traces = json.loads((dest / "trace.json").read_text())
        accepted = []
        for j, row in enumerate(traces):
            q = row["queried_token_ids"]
            assert len(q)==len(set(q))==row["queries"] <= len(pool)
            assert set(q) <= set(pool)
            if method == "hybrid_50_150":
                top = row["probability_top50_token_ids"]
                assert q[:50]==top
                remaining = np.setdiff1d(np.arange(len(pool)), np.flatnonzero(np.isin(pool, top)))
                random_part = np.random.default_rng(SEED+j).choice(remaining, 150, replace=False)
                assert q[50:200]==pool[random_part].tolist()
            if row["accepted_token_id"] is not None:
                d = float(np.linalg.norm(np.asarray(row["accepted_kv"], dtype=np.float64)
                                         -observation["kv"][j].astype(np.float64)))
                np.testing.assert_allclose(d, row["accepted_raw_l2"], rtol=1e-12, atol=1e-12)
                assert d < EPSILON
                assert row["accepted_token_id"] in q
                accepted.append(row["accepted_token_id"])
            else:
                assert row["finite_pool_exhausted"]
        assert accepted==run["recovered_token_ids"]
        matches = [a==b for a,b in zip(accepted, actual)]
        leading = next((i for i,v in enumerate(matches) if not v), len(matches))
        run.update(correct_tokens=sum(matches), exact_leading_prefix=leading,
                   all_exact=accepted==actual, incorrect_acceptances=len(matches)-sum(matches),
                   exact_ciphertext_text=run["recovered_text"]==truth["payload"],
                   accepted_l2_max=max((r["accepted_raw_l2"] for r in traces if r["accepted_token_id"] is not None), default=None),
                   maximum_queries_at_one_position=max(r["queries"] for r in traces))
        results[method]=run
    exp.dump(OUT / "evaluation.json", dict(sample_id="0001", epsilon=EPSILON,
        metric="raw KV L2, strict <, no LSB normalization or coordinate gate",
        target_tokens=len(actual), methods=results, validation_passed=True))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args=parser.parse_args()
    {"prepare":prepare, "run":lambda:run(args), "evaluate":evaluate}[args.stage]()
