"""Secondary sequential probability baseline without the primary 600-query cap.

Uses the same observations, candidate pool, calibration, model and read guard.
Primary experiment files and the original KV-guided method remain unchanged.
"""
import json
import time
import numpy as np
import experiment as exp

root = exp.ROOT / "results"
exp.access_guard(root)
rt, prior, tokenizer, runtime = exp.load_runtime()
import torch
obs = np.load(root / "observed/target.npz")
ids = torch.tensor(json.loads((root / "observed/candidate_pool.json").read_text())["token_ids"])
thresholds = np.asarray(json.loads((root / "calibration.json").read_text())["thresholds"]) * 1.01
destination = root / "sequential_unpruned/collision_plus"
if (destination / "run.json").exists():
    raise FileExistsError("Secondary baseline already finished")
tick = time.perf_counter()
cache, logits = runtime.prepare(obs["known_prefix_ids"].tolist(), "native")
setup_seconds = time.perf_counter()-tick
started = time.perf_counter()
rows, recovered = [], []
advance_seconds = 0.
for j, position in enumerate(obs["positions"]):
    assert cache[0].offset == int(position)
    target = torch.from_numpy(obs["kv"][j])
    tick = time.perf_counter()
    order = exp.ordered_probabilities(rt, logits, ids)
    rank_seconds = time.perf_counter()-tick
    oracle = rt.Oracle(runtime, cache, ids, exp.LAYER, 64)
    queried, pieces, component_pieces = [], [], []
    accepted = None
    forward_seconds = 0.
    tick_all = time.perf_counter()
    for start in range(0, len(ids), 20):
        selected = order[start:start+20]
        tick = time.perf_counter()
        measured = oracle.query(selected)
        forward_seconds += time.perf_counter()-tick
        d, nmse = exp.distance(measured, target)
        queried.extend(ids[torch.tensor(selected)].tolist())
        pieces.extend(nmse.tolist())
        component_pieces.extend(d.tolist())
        passing = np.flatnonzero(np.all(d < thresholds[None, :], axis=-1))
        if len(passing):
            accepted = int(ids[int(selected[int(passing[0])])])
            break
    best = int(np.argmin(pieces))
    row = dict(method="collision_plus", index=j, position=int(position),
               queried_token_ids=queried, normalized_mse=pieces,
               component_l2=component_pieces, queries=len(queried),
               best_token_id=queried[best], best_nmse=pieces[best],
               accepted_token_id=accepted, early_stopping=True,
               first_acceptance_queries=len(queried) if accepted is not None else None,
               known_correct_prefix=False, max_candidate_budget=len(ids),
               query_forward_seconds=forward_seconds, fit_rank_seconds=rank_seconds,
               seconds=time.perf_counter()-tick_all+rank_seconds)
    rows.append(row)
    exp.dump(destination / "trace.json", rows)
    print("UNPRUNED", j, "queries", len(queried), "accepted", accepted, flush=True)
    if accepted is None:
        break
    recovered.append(accepted)
    tick = time.perf_counter()
    runtime.base.layers = runtime.full_layers
    result = runtime.model(rt.mx.array([[accepted]]), cache=cache)
    rt.mx.eval(result)
    logits = result[0, -1]
    advance_seconds += time.perf_counter()-tick
exp.dump(destination / "run.json", dict(
    method="collision_plus", label="secondary full-pool sequential probability baseline",
    recovered_token_ids=recovered,
    recovered_text=tokenizer.decode(recovered, clean_up_tokenization_spaces=False),
    attempted_positions=len(rows), target_positions=len(obs["positions"]),
    stopped_unresolved=bool(rows and rows[-1]["accepted_token_id"] is None),
    total_queries=sum(r["queries"] for r in rows), seconds=time.perf_counter()-started,
    setup_full_model_seconds=setup_seconds, advance_seconds=advance_seconds,
    known_correct_prefix=False, private_file_access_guard=True,
    source_ciphertext_plaintext_keys_and_true_tokens_read=False,
    threshold_source="same independent public calibration as primary experiment",
    candidate_budget_per_position=len(ids), candidate_logical_batch=20,
    note="Added after observing primary 600-query failure; secondary higher-budget result, not equal-budget comparison"))
