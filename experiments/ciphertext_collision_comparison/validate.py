"""CPU-only audit of search budgets, candidate ordering, labels and read guards."""
import json
from pathlib import Path
import numpy as np
import experiment as exp

root = exp.ROOT / "results"
pool = json.loads((root / "observed/candidate_pool.json").read_text())["token_ids"]
pool_set = set(pool)
truth = json.loads((root / "private_eval/truth.json").read_text())
actual = truth["payload_ids"]
thresholds = np.asarray(json.loads((root / "calibration.json").read_text())["thresholds"]) * 1.01
rows = json.loads((root / "diagnostic/trace.json").read_text())
orders = np.load(root / "diagnostic/probability_orders.npz")["orders"]
assert len(rows) == len(actual) == 109
assert len(pool) == len(pool_set) == 53150
assert set(actual) <= pool_set


def audit_trace(trace, position, max_queries=600):
    q = trace["queried_token_ids"]
    assert len(q) == len(set(q)) == trace["queries"] <= max_queries
    assert set(q) <= pool_set
    distances = np.asarray(trace["normalized_mse"])
    components = np.asarray(trace["component_l2"])
    assert np.isfinite(distances).all() and np.isfinite(components).all()
    assert len(distances) == len(components) == len(q)
    assert trace["best_token_id"] == q[int(np.argmin(distances))]
    assert trace["best_nmse"] == float(distances.min())
    if trace["method"] == "kv_guided_200":
        rng = np.random.default_rng(20261019 + position)
        assert q[:200] == np.asarray(pool)[rng.choice(len(pool), 200, replace=False)].tolist()
    selected = trace["accepted_token_id"]
    if selected is not None:
        assert selected in q
        assert np.all(components[q.index(selected)] < thresholds)


for j, row in enumerate(rows):
    assert row["index"] == j and row["known_correct_prefix"]
    for method in exp.METHODS:
        trace = row["methods"][method]
        audit_trace(trace, j)
        assert trace["queries"] == 600 and not trace["early_stopping"]
    expected = np.asarray(pool)[orders[j, :600]].tolist()
    assert row["methods"]["collision_plus"]["queried_token_ids"] == expected

sequential = {}
for method in exp.METHODS:
    folder = root / "sequential" / method
    run = json.loads((folder / "run.json").read_text())
    traces = json.loads((folder / "trace.json").read_text())
    for j, trace in enumerate(traces):
        audit_trace(trace, j)
        assert trace["early_stopping"] and not trace["known_correct_prefix"]
    accepted = [t["accepted_token_id"] for t in traces if t["accepted_token_id"] is not None]
    assert accepted == run["recovered_token_ids"]
    assert sum(t["queries"] for t in traces) == run["total_queries"]
    assert run["private_file_access_guard"]
    assert not run["source_ciphertext_plaintext_keys_and_true_tokens_read"]
    if run["stopped_unresolved"]:
        assert traces[-1]["accepted_token_id"] is None
    sequential[method] = dict(attempted=len(traces), accepted=len(accepted),
                              total_queries=run["total_queries"])

secondary = root / "sequential_unpruned/collision_plus"
if (secondary / "run.json").exists():
    run = json.loads((secondary / "run.json").read_text())
    traces = json.loads((secondary / "trace.json").read_text())
    for j, trace in enumerate(traces):
        audit_trace(trace, j, len(pool))
        assert trace["early_stopping"] and not trace["known_correct_prefix"]
    assert [t["accepted_token_id"] for t in traces if t["accepted_token_id"] is not None] == run["recovered_token_ids"]
    assert sum(t["queries"] for t in traces) == run["total_queries"]
    sequential["secondary_unpruned_probability"] = dict(
        attempted=len(traces), accepted=len(run["recovered_token_ids"]),
        total_queries=run["total_queries"])

# Exercise the same guard in this fresh process; neither open may reach data.
exp.access_guard(root)
denied = 0
for path in (root / "private_eval/truth.json", exp.DATASET / "prompts/encrypted/0001.txt"):
    try:
        path.read_bytes()
    except PermissionError:
        denied += 1
    else:
        raise AssertionError("Private input was readable under recovery guard")
assert denied == 2
exp.dump(root / "validation.json", dict(
    passed=True, diagnostic_positions=len(rows), methods=list(exp.METHODS),
    same_candidate_pool=True, pool_contains_all_actual_tokens=True,
    all_queries_unique_and_within_budget=True,
    original_200_random_candidates_recomputed=True,
    probability_candidate_order_replayed=True,
    every_acceptance_passes_shared_independent_thresholds=True,
    private_source_guard_negative_checks=denied, sequential=sequential,
    note="This audit verifies recorded experiment integrity, not global uniqueness or NPU behavior"))
print("PASS: 109 paired positions, budgets, ordering, sequential traces and private-file guards")
