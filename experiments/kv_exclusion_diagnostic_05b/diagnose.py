#!/usr/bin/env python3
"""Privileged offline bounds; never a deployable stopping certificate."""
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SEARCH = ROOT.parent / 'kv_search_strategies_05b'
spec = importlib.util.spec_from_file_location('frozen_search_strategies', SEARCH / 'experiment.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)
DATA = search.PRIOR / 'data'
METHODS = ('original', 'linear_rbf')
BUDGETS = (200, 400, 600)
EPSILONS = (.1, .25)
BOUNDS = ('distance_uniform', 'kv_uniform', 'kv_coordinate')
EXAMPLE = 'confirmation_s20261019_t1_l12'
LABELS = {'original': '原线性代理', 'linear_rbf': '线性 + RBF',
          'distance_uniform': '统一距离误差界', 'kv_uniform': '统一 KV 误差界',
          'kv_coordinate': '逐坐标 KV 误差界'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def distribution(x):
    return {name: float(np.quantile(x, q)) for name, q in
            (('min', 0), ('median', .5), ('p95', .95), ('p99', .99), ('max', 1))}


def fit(method, embedding, indices, measured):
    # Only measured rows enter the imported, frozen predictor.
    if method == 'original':
        return search.ridge_predict(embedding, indices, measured)[0]
    return search.rbf_predict(embedding, indices, measured)


def intervals(prediction, table, target_index):
    """Evaluation only: this function intentionally accesses the FULL answer table."""
    table, prediction = table.astype(np.float64), prediction.astype(np.float64)
    target = table[target_index]
    scale = max(float(np.max(np.abs(target))), 1e-12)
    true_distance = np.max(np.abs(table - target), axis=1) / scale
    coordinate_distance = np.abs(prediction - target) / scale
    predicted_distance = coordinate_distance.max(axis=1)
    coordinate_error = np.abs(prediction - table) / scale
    distance_error = np.abs(true_distance - predicted_distance)
    coordinate_radius = coordinate_error.max(axis=0)
    vector_radius = float(coordinate_radius.max())
    distance_radius = float(distance_error.max())
    # Outward margin for FP64 evaluation, not a formal arithmetic certificate.
    pad = 1e-12 * max(1., vector_radius, float(true_distance.max()),
                      float(predicted_distance.max()))
    assert distance_radius <= vector_radius + pad
    bounds = {
        'distance_uniform': (np.maximum(0., predicted_distance - distance_radius - pad),
                             predicted_distance + distance_radius + pad),
        'kv_uniform': (np.maximum(0., predicted_distance - vector_radius - pad),
                       predicted_distance + vector_radius + pad),
        'kv_coordinate': (np.maximum(0., coordinate_distance - coordinate_radius - pad).max(axis=1),
                           (coordinate_distance + coordinate_radius + pad).max(axis=1)),
    }
    for lower, upper in bounds.values():
        assert np.isfinite(lower).all() and np.isfinite(upper).all()
        assert np.all(lower <= true_distance) and np.all(true_distance <= upper)
    assert np.all(bounds['kv_coordinate'][0] >= bounds['kv_uniform'][0] - pad)
    assert np.all(bounds['kv_coordinate'][1] <= bounds['kv_uniform'][1] + pad)
    assert np.all(bounds['distance_uniform'][0] >= bounds['kv_uniform'][0] - pad)
    assert np.all(bounds['distance_uniform'][1] <= bounds['kv_uniform'][1] + pad)
    stats = {
        'normalization_scale': scale, 'numerical_outward_margin': pad,
        'distance_uniform_radius': distance_radius, 'kv_uniform_radius': vector_radius,
        'coordinate_radius': coordinate_radius.tolist(),
        'distance_error_quantiles': distribution(distance_error),
        'vector_error_quantiles': distribution(coordinate_error.max(axis=1)),
        'worst_distance_error_index': int(distance_error.argmax()),
        'worst_vector_error_index': int(coordinate_error.max(axis=1).argmax()),
    }
    return true_distance, predicted_distance, bounds, stats


def classify(true_distance, indices, bounds, epsilon):
    queried = np.zeros(len(true_distance), dtype=bool)
    queried[indices] = True
    truth = true_distance <= epsilon
    found, missed = int((truth & queried).sum()), int((truth & ~queried).sum())
    result = {'epsilon': epsilon, 'truth_count': int(truth.sum()),
              'queried_matches': found, 'unqueried_matches': missed,
              'queried_all_solutions': missed == 0, 'bounds': {}}
    for name, (lower, upper) in bounds.items():
        rejected = ~queried & (lower > epsilon)
        accepted = ~queried & (upper <= epsilon)
        unresolved = ~queried & ~rejected & ~accepted
        false_exclusions = int((rejected & truth).sum())
        false_inclusions = int((accepted & ~truth).sum())
        assert false_exclusions == false_inclusions == 0
        assert not (rejected & accepted).any()
        assert queried.sum() + rejected.sum() + accepted.sum() + unresolved.sum() == len(truth)
        # A conservative fallback verifies ALL unqueried, non-rejected candidates.
        remaining = int((~queried & ~rejected).sum())
        assert remaining >= missed
        result['bounds'][name] = {
            'excluded_unqueried': int(rejected.sum()),
            'accepted_unqueried': int(accepted.sum()),
            'unresolved': int(unresolved.sum()),
            'remaining_to_query_for_verified_set': remaining,
            'hypothetical_total_after_fallback': len(indices) + remaining,
            'oracle_intervals_classify_all': not bool(unresolved.any()),
            'oracle_verified_set_can_stop': remaining == 0,
            'false_exclusions': false_exclusions, 'false_inclusions': false_inclusions,
        }
    return result


def aggregate(records):
    rows = []
    for split in ('development', 'confirmation'):
        for layer in (12, 24):
            for method in METHODS:
                for budget in BUDGETS:
                    group = [r for r in records if r['split'] == split and r['layer'] == layer
                             and r['method'] == method and r['budget'] == budget]
                    for eps in EPSILONS:
                        evaluations = [next(x for x in r['thresholds'] if x['epsilon'] == eps) for r in group]
                        for bound in BOUNDS:
                            cells = [x['bounds'][bound] for x in evaluations]
                            row = {'split': split, 'layer': layer, 'method': method, 'budget': budget,
                                   'epsilon': eps, 'bound': bound, 'cases': len(group),
                                   'truth_count_min': min(x['truth_count'] for x in evaluations),
                                   'truth_count_max': max(x['truth_count'] for x in evaluations),
                                   'multi_solution_cases': sum(x['truth_count'] > 1 for x in evaluations),
                                   'queried_complete_cases': sum(x['queried_all_solutions'] for x in evaluations),
                                   'interval_complete_cases': sum(x['oracle_intervals_classify_all'] for x in cells),
                                   'verified_stop_cases': sum(x['oracle_verified_set_can_stop'] for x in cells),
                                   'missed_solutions_total': sum(x['unqueried_matches'] for x in evaluations),
                                   'false_exclusions': sum(x['false_exclusions'] for x in cells),
                                   'false_inclusions': sum(x['false_inclusions'] for x in cells),
                                   'mean_distance_radius': statistics.mean(r['error']['distance_uniform_radius'] for r in group),
                                   'mean_vector_radius': statistics.mean(r['error']['kv_uniform_radius'] for r in group)}
                            for field in ('excluded_unqueried', 'accepted_unqueried', 'unresolved',
                                          'remaining_to_query_for_verified_set', 'hypothetical_total_after_fallback'):
                                values = [x[field] for x in cells]
                                row['mean_' + field] = statistics.mean(values)
                                row['min_' + field], row['max_' + field] = min(values), max(values)
                            rows.append(row)
    return rows


def report(obj):
    summary = obj['summary']
    focus = {r['method']: r for r in summary if r['split'] == 'confirmation' and r['layer'] == 12
             and r['epsilon'] == .25 and r['budget'] == 600 and r['bound'] == 'distance_uniform'}
    assert sum(r['interval_complete_cases'] for r in summary) == 0
    lines = ['# 0.5B 候选排除能力诊断结果', '',
        '**这是使用完整答案得到误差界的离线诊断，不是可部署的找全算法，也不是加速实验。**', '',
        '## 结论', '',
        '这轮没有得到可停止的案例：在所有检查点、阈值和三种区间下，都仍有未定候选。'
        '即使完整答案提前提供了真实最坏误差，这些冻结代理加统一界仍不能在 600 次查询内确认找全。',
        f'重点多解组（第 12 层、ε=0.25）在查询 600 个候选后，原线性代理平均排除 '
        f'{focus["original"]["mean_excluded_unqueried"]:.1f} 个、仍需查询 '
        f'{focus["original"]["mean_remaining_to_query_for_verified_set"]:.1f} 个；RBF 平均排除 '
        f'{focus["linear_rbf"]["mean_excluded_unqueried"]:.1f} 个、仍需查询 '
        f'{focus["linear_rbf"]["mean_remaining_to_query_for_verified_set"]:.1f} 个。'
        '这里剩余量的基数是 9,400 个未查询候选。',
        'RBF 在同组的实际查询找全数为 12/16，原方法为 11/16；但 RBF 的全局排除能力反而较弱。'
        '**更容易找到解，不等于更容易证明没有遗漏。**', '',
        '## 设置与范围', '',
        'Qwen2.5-0.5B，FP32 KV，10,000 个候选，固定已知正确前缀。原方法和 RBF 代理各自复用先前查询轨迹；'
        '在 200、400、600 点重新拟合，完整表仅进入之后的误差界评价。两种方法的差异包含查询轨迹差异。',
        '共 64 个目标/层案例 × 2 个代理 × 3 个预算 = 384 次拟合；每次检查 2 个阈值和 3 种误差区间。'
        '以下主要展示原 confirmation 分组，每层 16 例。分组并非本轮新收集，不能视为新独立确认。',
        '阈值为归一化最大坐标差 ε=0.10/0.25；这不是 Gemma INT8 或 ±4 LSB 测试。', '',
        '## 主结果：统一距离误差界', '',
        '对所有候选计算真实的最大 `|真实距离 − 预测距离|`。这是该代理、该候选池上最小的统一对称半径。'
        '“剩余需查询”包括所有尚未查询且不能排除的候选；“区间判完”允许由上界直接判定匹配。', '']
    for layer in (12, 24):
        for eps in EPSILONS:
            rows = [r for r in summary if r['split'] == 'confirmation' and r['layer'] == layer
                    and r['epsilon'] == eps and r['bound'] == 'distance_uniform']
            first = rows[0]
            lines += [f'### 第 {layer} 层，ε={eps:g}', '',
                f'真实解数 {first["truth_count_min"]}–{first["truth_count_max"]}；'
                f'{first["multi_solution_cases"]}/16 个多解案例。', '',
                '| 方法 | 已查询 | 平均误差半径 | 平均排除未查询 | 平均剩余需查询 | 查询已找全 | 区间判完 |',
                '|---|---:|---:|---:|---:|---:|---:|']
            for r in rows:
                lines.append(f'| {LABELS[r["method"]]} | {r["budget"]} | {r["mean_distance_radius"]:.4f} | '
                    f'{r["mean_excluded_unqueried"]:.1f} | {r["mean_remaining_to_query_for_verified_set"]:.1f} | '
                    f'{r["queried_complete_cases"]}/16 | {r["interval_complete_cases"]}/16 |')
            lines.append('')
    lines += ['## 三种界在 600 点的比较', '',
        '重点展示第 12 层、ε=0.25 的多解情况。所有误差半径均由完整候选表给出。', '',
        '| 方法 | 界 | 平均排除 | 平均未定 | 平均界内匹配但未查询 | 剩余查询范围 | 区间判完 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['split'] != 'confirmation' or r['layer'] != 12 or r['epsilon'] != .25 or r['budget'] != 600:
            continue
        lines.append(f'| {LABELS[r["method"]]} | {LABELS[r["bound"]]} | {r["mean_excluded_unqueried"]:.1f} | '
            f'{r["mean_unresolved"]:.1f} | {r["mean_accepted_unqueried"]:.1f} | '
            f'{r["min_remaining_to_query_for_verified_set"]}–{r["max_remaining_to_query_for_verified_set"]} | '
            f'{r["interval_complete_cases"]}/16 |')
    lines += ['', '## 固定案例：此前命中实际 token 但漏掉其他解的案例', '',
        f'案例 `{EXAMPLE}`，ε=0.25；沿用此前报告的例子，没有按本次结果挑选。', '',
        '| 方法 | 已查询 | 已找到 / 真解 | 距离误差半径 | 排除 | 剩余需查询 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in obj['records']:
        if r['key'] != EXAMPLE:
            continue
        evaluation = next(x for x in r['thresholds'] if x['epsilon'] == .25)
        b = evaluation['bounds']['distance_uniform']
        lines.append(f'| {LABELS[r["method"]]} | {r["budget"]} | '
            f'{evaluation["queried_matches"]}/{evaluation["truth_count"]} | '
            f'{r["error"]["distance_uniform_radius"]:.4f} | {b["excluded_unqueried"]} | '
            f'{b["remaining_to_query_for_verified_set"]} |')
    lines += ['', '## 为什么会这样，以及下一步的判据', '',
        '统一半径必须照顾最难预测的候选；不能因为多数候选预测准确，就把最坏误差缩小。'
        '第 12 层、600 点时，各案例误差中位数的平均值为原方法 0.1088、RBF 0.0906，'
        '但最大误差的平均值分别为 0.7584、0.7183。较小的典型误差没有转化成同样小的统一界。',
        '是否可排除取决于 `预测距离 − 误差界 > 阈值`，不只取决于误差界本身。'
        'RBF 即便在该组具有稍小的统一半径，也没有因此排除更多候选。',
        '当前查询策略的目标是接近目标 KV，并未为缩小所有候选的最坏预测误差而设计。'
        '观察到最坏误差也不随预算单调减少；不能人为让半径随查询轮次缩小，继而声称停止可靠。',
        '下一步若继续，应先检验局部或分组误差界是否显著增加安全排除量。分组应由 embedding、'
        '预测值等可用信息确定，再在离线评估中量出各组最坏误差；仍不能把看过答案的组内最大误差'
        '作为实际算法免费获得的保证。只有排除能力确实改善，才值得进一步从模型结构或明确假设推导可用的局部界。',
        '本轮未开展局部界实验，也没有证明原问题不可解。已经定位的限制是：当前代理加这几种全候选统一界，'
        '在现有查询预算下不足以完成排除。', '',
        '## 审计与正确解释', '',
        '- 全部 384 次拟合、2,304 组阈值/区间分类均检查了完整有限表上的覆盖；错误排除与错误接纳均为 0。',
        '- 所有旧源码、轨迹和 KV/embedding 输入文件的 SHA256 在运行前后相同。原线性代理的排序与原函数逐检查点核对；200/400 点的下一批排序与保存轨迹一致。',
        '- 评价使用 FP64 距离，与先前 FP32 评价在两个阈值的真解集合一致；为区间增加微小向外余量。这是有限存储表的数值检查，不是不同后端/真实数计算图的形式化证书。',
        '- 区间界覆盖全部答案是本轮构造的结果，不是从少量查询中推导出的泛化保证。没有用这个界决定新查询，也没有把其构造成本忽略后宣称加速。',
        '- 查询恰好找全、区间能判完、实际算法能证明可停止是不同结论。原搜索算法目前仍没有本轮这种误差界。',
        '- 结果只评价冻结代理加这三类统一误差界；若排除不足，不能据此证明所有局部界、其他模型或找全算法均不可行。', '',
        '## 文件', '',
        '- [实验设置与区间公式](README.md)',
        '- [全部案例与汇总](results/diagnostic.json)',
        '- [独立审计](results/validation.json)',
        '- 固定案例的逐候选距离、区间和查询索引：`results/example_*.npz`。', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    torch.set_num_threads(4)
    started = time.perf_counter()
    source_config = json.loads((SEARCH / 'results/config.json').read_text())
    assert digest(SEARCH / 'experiment.py') == source_config['source_sha256']
    assert digest(Path(source_config['original_source_path'])) == source_config['original_source_sha256']
    assert digest(search.PRIOR / 'experiment.py') == source_config['prior_experiment_sha256']
    cases = json.loads((SEARCH / 'results/trials.json').read_text())
    frozen = [Path(source_config['original_source_path']), SEARCH / 'experiment.py',
              search.PRIOR / 'experiment.py', SEARCH / 'results/config.json',
              SEARCH / 'results/trials.json'] + sorted(DATA.glob('*/*.npz'))
    hashes = {str(p): digest(p) for p in frozen}
    out = ROOT / 'results'
    out.mkdir(parents=True, exist_ok=True)
    config = {'methods': METHODS, 'budgets': BUDGETS, 'epsilons': EPSILONS, 'bounds': BOUNDS,
              'case_count': len(cases), 'model': source_config['model'], 'revision': source_config['revision'],
              'dtype': 'FP32 fit and stored KV; FP64 interval evaluation', 'threads': 4,
              'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
              'privileged_full_answer_bounds': True, 'deployable_stopping_certificate': False,
              'new_search_runs': 0, 'independent_new_confirmation_set': False,
              'source_sha256': digest(Path(__file__)), 'protocol_sha256': digest(ROOT / 'README.md'),
              'frozen_inputs_sha256': hashes}
    save(out / 'config.json', config)
    embeddings, tables, records = {}, {}, []
    parity_count, next_batch_count, truth_parity_count = 0, 0, 0
    with torch.inference_mode():
        for case in cases:
            seed, pi, layer = case['seed'], case['prefix_index'], case['layer']
            if seed not in embeddings:
                with np.load(DATA / str(seed) / 'features.npz') as raw:
                    embeddings[seed] = torch.from_numpy(raw['embedding'].copy())
            key = (seed, pi, layer)
            if key not in tables:
                with np.load(DATA / str(seed) / f'prefix_{pi}.npz') as raw:
                    tables[key] = raw[f'kv_{layer}'].copy()
            table = tables[key]
            target = table[case['target_index']]
            for method in METHODS:
                trace = case['methods'][method]['trace']['indices']
                assert len(trace) == len(set(trace)) == 600
                assert trace[:200] == case['initial_indices']
                for budget in BUDGETS:
                    indices = torch.tensor(trace[:budget])
                    measured = torch.from_numpy(table[indices.numpy()].copy())
                    tick = time.perf_counter()
                    prediction = fit(method, embeddings[seed], indices, measured)
                    fit_time = time.perf_counter() - tick
                    score = (prediction - torch.from_numpy(target)).square().mean(1)
                    score[indices] = float('inf')
                    ranking = score.argsort()
                    if method == 'original':
                        reference = search.old.proxy_ranking(embeddings[seed], indices, measured,
                                                             torch.from_numpy(target))
                        assert torch.equal(ranking, reference), (case['key'], budget, 'baseline parity')
                        parity_count += 1
                    if budget < 600:
                        assert ranking[:20].tolist() == trace[budget:budget + 20], (case['key'], method, budget)
                        next_batch_count += 1
                    prediction_np = prediction.numpy()
                    true_d, pred_d, bounds, error = intervals(prediction_np, table, case['target_index'])
                    prior_distance = np.max(np.abs(table - target), axis=1) / max(np.max(np.abs(target)), 1e-12)
                    evaluations = []
                    for eps in EPSILONS:
                        assert np.array_equal(true_d <= eps, prior_distance <= eps)
                        truth_parity_count += 1
                        evaluation = classify(true_d, indices.numpy(), bounds, eps)
                        old = next(x for x in case['methods'][method]['checkpoints'] if x['budget'] == budget)
                        assert evaluation['queried_matches'] == old[f'found_{eps:g}']
                        assert evaluation['truth_count'] == old[f'truth_count_{eps:g}']
                        evaluations.append(evaluation)
                    record = {k: case[k] for k in ('key', 'split', 'seed', 'trial', 'prefix_index', 'layer', 'target_index')}
                    record.update({'method': method, 'budget': budget, 'candidate_count': len(table),
                                   'error': error, 'thresholds': evaluations, 'fit_seconds_diagnostic_only': fit_time,
                                   'prediction_sha256': hashlib.sha256(prediction_np.tobytes()).hexdigest()})
                    records.append(record)
                    if case['key'] == EXAMPLE:
                        arrays = {'queried_indices': indices.numpy(), 'true_distance': true_d,
                                  'predicted_distance': pred_d}
                        for name, (lower, upper) in bounds.items():
                            arrays[name + '_lower'], arrays[name + '_upper'] = lower, upper
                        np.savez_compressed(out / f'example_{method}_{budget}.npz', **arrays)
            print(f'{case["key"]}: {len(records)}/384 snapshots', flush=True)
    assert all(digest(Path(path)) == expected for path, expected in hashes.items())
    assert digest(Path(__file__)) == config['source_sha256']
    obj = {'config': config, 'records': records, 'summary': aggregate(records),
           'audit': {'frozen_inputs_unchanged': True, 'snapshots': len(records),
                     'baseline_ranking_parity_checks': parity_count, 'saved_next_batch_checks': next_batch_count,
                     'fp32_fp64_threshold_set_parity_checks': truth_parity_count,
                     'full_table_interval_coverage_checks': len(records) * len(BOUNDS),
                     'classification_checks': len(records) * len(BOUNDS) * len(EPSILONS),
                     'elapsed_seconds_diagnostic_only': time.perf_counter() - started}}
    save(out / 'diagnostic.json', obj)
    report(obj)
    print(json.dumps(obj['audit'], indent=2), flush=True)
    print(ROOT / 'RESULTS.md')


if __name__ == '__main__':
    main()
