#!/usr/bin/env python3
"""Generate the research report from audited, unfiltered result files."""
import json
import statistics as st
from collections import defaultdict

import numpy as np
import torch

import experiment as e

LABELS = {'original': '原线性代理', 'linear_rbf': '线性 + RBF',
          'distance': '预测距离分组', 'embedding': 'PCA embedding 分组'}
COHORTS = {'development': '旧 development', 'confirmation': '旧 confirmation', 'fresh': '新目标'}


def mean(rows, field):
    return st.mean(r[field] for r in rows)


def aggregate_local(rows):
    fields = ('split', 'layer', 'method', 'budget', 'partition', 'groups', 'bound', 'epsilon')
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[f] for f in fields)].append(r)
    output = []
    for key, items in buckets.items():
        output.append({**dict(zip(fields, key)), 'cases': len(items),
            'mean_remaining': mean(items, 'remaining'), 'min_remaining': min(r['remaining'] for r in items),
            'max_remaining': max(r['remaining'] for r in items), 'mean_excluded': mean(items, 'excluded'),
            'mean_hypothetical_total': mean(items, 'hypothetical_total'),
            'queried_complete': sum(r['unqueried_matches'] == 0 for r in items),
            'verified_stops': sum(r['verified_can_stop'] for r in items),
            'truth_min': min(r['truth_count'] for r in items), 'truth_max': max(r['truth_count'] for r in items),
            'multi_solution_cases': sum(r['truth_count'] > 1 for r in items)})
    return output


def aggregate_empirical(rows):
    fields = ('split', 'layer', 'method', 'groups', 'calibration_count', 'epsilon')
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[f] for f in fields)].append(r)
    output = []
    for key, items in buckets.items():
        failures = {r['key'] for r in items if not r['complete']}
        entry = {**dict(zip(fields, key)), 'runs': len(items), 'targets': len({r['key'] for r in items}),
            'mean_queries': mean(items, 'queries'), 'min_queries': min(r['queries'] for r in items),
            'max_queries': max(r['queries'] for r in items), 'complete_runs': sum(r['complete'] for r in items),
            'missed_solutions_total': sum(r['missed'] for r in items), 'targets_with_any_omission': len(failures),
            'mean_lower_bound_violations': mean(items, 'lower_bound_violations_all_candidates'), 'audits': []}
        for delta in e.DELTAS:
            audits = [next(a for a in r['audits'] if a['delta'] == delta) for r in items]
            entry['audits'].append({'delta': delta, 'mean_queries': mean(audits, 'queries'),
                'min_queries': min(a['queries'] for a in audits), 'max_queries': max(a['queries'] for a in audits),
                'complete_runs': sum(a['complete'] for a in audits), 'fallbacks': sum(a['fallback_triggered'] for a in audits),
                'missed_total': sum(a['missed'] for a in audits),
                'mean_exact_conditional_failure_probability': mean(audits, 'exact_conditional_failure_probability'),
                'max_exact_conditional_failure_probability': max(a['exact_conditional_failure_probability'] for a in audits)})
        output.append(entry)
    return output


def write_example(empirical):
    torch.set_num_threads(4)
    row = next(r for r in empirical if r['split'] == 'fresh' and r['layer'] == 12 and r['method'] == 'original'
               and r['groups'] == 32 and r['calibration_count'] == 600 and r['repeat'] == 0
               and r['epsilon'] == .25 and not r['complete'])
    cases = {c['key']: c for c in json.loads((e.ROOT / 'fresh_results/cases.json').read_text())}
    c = cases[row['key']]
    with np.load(e.d.DATA / str(c['seed']) / 'features.npz') as raw:
        embedding = torch.from_numpy(raw['embedding'].copy())
        ids = raw['ids'].copy()
    with np.load(e.d.DATA / str(c['seed']) / f'prefix_{c["prefix_index"]}.npz') as raw:
        table = raw[f'kv_{c["layer"]}'].copy()
    texts = json.loads((e.d.DATA / str(c['seed']) / 'candidates.json').read_text())
    target = table[c['target_index']].astype(np.float64)
    scale = max(float(np.abs(target).max()), 1e-12)
    true = np.abs(table.astype(np.float64) - target).max(axis=1) / scale
    initial = np.array(c['methods']['original']['trace']['indices'], dtype=np.int64)
    with torch.inference_mode():
        prediction = e.d.fit('original', embedding, torch.from_numpy(initial), torch.from_numpy(table[initial].copy())).numpy()
    predicted = np.abs(prediction.astype(np.float64) - target).max(axis=1) / scale
    labels = e.grouped_labels(predicted, 'distance')[32]
    oracle = e.ReplayOracle(true)
    excluded, radius, lower, cal = e.base_search(oracle, initial, predicted, labels, 600, row['calibration_seed'], .25)
    assert len(oracle.indices) == row['queries']
    lines = ['# 一个实际漏解案例', '',
        '选择规则：按保存顺序，取新目标组中原方法、32 组、600 个校准点、ε=0.25、重复 0 的第一个失败案例。', '',
        f'- 案例：`{c["key"]}`；目标 token：`{c["target_text"]}`，ID {c["target_token_id"]}。',
        f'- 固定前缀：`{c["prefix"]}`。',
        f'- 原搜索在 {c["methods"]["original"]["first_hit_batch"]} 次查询时命中实际目标。',
        f'- 真正相容的候选有 {row["truth_count"]} 个；经验局部界方案总共查询 {row["queries"]} 个，'
        f'只返回 {row["found"]} 个，漏掉 {row["missed"]} 个。', '',
        '下面按候选索引列出前 5 个被错误排除的匹配项。距离和半径均为本实验归一化单位，不是 LSB。', '',
        '| 候选索引 | token 文本 | token ID | 真实距离 | 预测距离 | 校准组半径 | 声称的下界 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    details = []
    for index in row['missed_indices'][:5]:
        assert true[index] <= .25 < lower[index]
        details.append({'index': index, 'token_id': int(ids[index]), 'text': texts[index]['text'],
            'true_distance': float(true[index]), 'predicted_distance': float(predicted[index]),
            'group_radius': float(radius[labels[index]]), 'invalid_lower_bound': float(lower[index])})
        lines.append(f'| {index} | `{texts[index]["text"]}` | {int(ids[index])} | {true[index]:.6f} | '
            f'{predicted[index]:.6f} | {radius[labels[index]]:.6f} | {lower[index]:.6f} |')
    lines += ['', '问题不是阈值比较代码出错，而是校准样本没有覆盖组内最坏过预测残差，'
        '导致所谓下界高于真实距离。把这些未查询候选排除后，“所有剩余项都处理完了”的停止条件就失去依据。', '',
        '| 审计 δ | 总查询 | 是否触发完整回退 | 剩余漏解 |', '|---:|---:|---|---:|']
    for a in row['audits']:
        lines.append(f'| {a["delta"]} | {a["queries"]} | {a["fallback_triggered"]} | {a["missed"]} |')
    lines += ['', '审计能补上有明确概率含义的停止规则，但本例触发回退后会读完全部候选。', '']
    (e.ROOT / 'EXAMPLE.md').write_text('\n'.join(lines))
    e.d.save(e.ROOT / 'results/example.json', {'case': c['key'], 'row': row, 'first_five_omissions': details})


def main():
    local, empirical, audits = [], [], []
    for name in ('results', 'fresh_results'):
        folder = e.ROOT / name
        validation = json.loads((folder / 'validation.json').read_text())
        assert validation['passed']
        assert e.d.digest(folder / 'local.json') == validation['local_sha256']
        assert e.d.digest(folder / 'empirical.json') == validation['empirical_sha256']
        local.extend(json.loads((folder / 'local.json').read_text()))
        empirical.extend(json.loads((folder / 'empirical.json').read_text()))
        audits.append(validation)
    ls, es = aggregate_local(local), aggregate_empirical(empirical)
    e.d.save(e.ROOT / 'results/summary.json', {'local': ls, 'empirical': es})
    lines = ['# KVidence 找全与可信停止：完整研究报告', '',
        '## 核心结论', '',
        '**局部界有明显排除潜力，但本轮没有获得一个既大幅减少查询、又能对真实深层 KV 保证找全的算法。** '
        '这个结论经过旧案例诊断、冻结后新目标检验、独立审计以及停止条件证明。', '',
        '1. 按预测距离分组并使用真实最坏单侧误差，明显优于全域统一界；这证明局部化值得关注，但构造这个 oracle 界已经需要完整答案。',
        '2. 用少量额外校准样本替代真实最坏误差，能很快给出一个集合，但实际出现遗漏；不能据此宣布“已经找全”。',
        '3. 独立随机审计加完整回退具有可证明的单次漏解概率上界，不过查询代价接近 10,000 候选穷举。',
        '4. 已证明的黑盒查询障碍不适用于所有已知参数模型。利用实际网络结构构造足够紧、低成本的局部证书，仍是尚未解决的部分。', '',
        '原 KVidence 方法、参数、旧查询轨迹和数据未修改。新增代码与所有结果均在本目录。', '',
        '## 1. 研究问题与设计', '',
        '目标是在固定正确前缀、模型、有限候选词表和距离阈值下，返回全部相容 token。'
        '这包含命中目标之外的其他匹配候选；多个相容候选并不等于能够唯一认定原输入。',
        '模型为 Qwen2.5-0.5B，10,000 候选，FP32 KV，第 12/24 层，四个短前缀。'
        '比较原线性代理与已冻结的线性 + RBF 代理，两者各自原查询轨迹不变。主检查点为 600 个查询。',
        '旧组：64 个目标/层案例，含此前 development 和 confirmation。新目标：另选 16 个 token、'
        '两层共 32 例，其实际词表 ID 与之前所有目标不重叠。新目标协议在生成目标和运行之前固定。'
        '新目标仍共享模型、候选池和前缀，不能称为跨模型或新语料验证。',
        '所有距离阈值 ε=0.10/0.25 均为归一化最大坐标差。没有把它们解释为 INT8 的 ±4 LSB。'
        '全部新实验都是对现有真实模型 KV 表的计数 oracle 回放；没有把回放耗时当作实际 NPU 推理耗时。', '',
        '## 2. 已知真实最坏误差时，局部化有没有用？', '',
        '下表是第 12 层、ε=0.25、已查询 600 个候选后，平均还需要实际检查多少候选。'
        '统一单侧界也报告在内，以区分“只管过预测”与“分组”各自带来的变化。', '',
        '| 目标组 | 方法 | 统一对称界 | 统一单侧界 | 距离 8 组 | 距离 32 组 | 距离 128 组 | embedding 32 组 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort in ('confirmation', 'fresh'):
        for method in e.d.METHODS:
            def value(groups, partition='distance', bound='one_sided'):
                return next(r['mean_remaining'] for r in ls if r['split'] == cohort and r['layer'] == 12
                    and r['method'] == method and r['budget'] == 600 and r['epsilon'] == .25
                    and r['groups'] == groups and r['partition'] == partition and r['bound'] == bound)
            vals = [value(1, bound='symmetric'), value(1), value(8), value(32), value(128), value(32, 'embedding')]
            lines.append(f'| {COHORTS[cohort]} | {LABELS[method]} | ' + ' | '.join(f'{v:.1f}' for v in vals) + ' |')
    lines += ['', '这些数是 oracle 诊断，不是可部署查询预算。分组由可用特征决定，但组内最坏误差仍由完整答案决定。'
        '新目标更难，不能只引用旧组较好的数字：新目标第 12 层在 ε=0.25 下的相容集合大小可达 192。',
        '第 24 层在本次局部 oracle 界下明显更容易；新目标 ε=0.25 时真实集合大小为 1–2。'
        '这不支持“所有深层都具有同样难度”的结论。', '',
        '一个理论解释是：若组内预测距离为同一常数且高于阈值，精确单侧残差界是否能排除整组，'
        '等价于该组最小真实距离是否超过阈值。因此最紧的 oracle 界本身可能已经包含“组内有没有解”的答案。'
        '详细恒等式见 [THEORY.md 第 3.1 节](THEORY.md#31-为什么分组有潜力)。', '',
        '## 3. 不看完整答案，改用校准样本，会怎样？', '',
        '固定 600 点代理，再从未查询候选中随机抽取额外 600 个校准点。以组内最大观测过预测残差估计半径，'
        '查询全部未被排除的候选并返回已验证匹配。下表每行是 16 个目标 × 5 次校准随机重复，共 80 次运行。'
        '重复共享目标，不是 80 个独立语料样本。', '',
        '| 目标组 | 方法 | 分组数 | 平均总查询 | 找全运行 | 出现遗漏的目标数 | 漏项总数（含重复） |',
        '|---|---|---:|---:|---:|---:|---:|']
    for cohort in ('confirmation', 'fresh'):
        for method in e.d.METHODS:
            for groups in (1, 32):
                r = next(r for r in es if r['split'] == cohort and r['layer'] == 12 and r['method'] == method
                         and r['epsilon'] == .25 and r['groups'] == groups and r['calibration_count'] == 600)
                lines.append(f'| {COHORTS[cohort]} | {LABELS[method]} | {groups} | {r["mean_queries"]:.1f} | '
                    f'{r["complete_runs"]}/{r["runs"]} | {r["targets_with_any_omission"]}/{r["targets"]} | '
                    f'{r["missed_solutions_total"]} |')
    lines += ['', '局部经验半径过窄会制造“已经全部处理”的假象。它能大幅减少查询，但遗漏真实匹配候选。'
        '样本中偶尔全部找全也不能推出全候选覆盖保证。', '',
        '### 额外校准量的预设敏感性分析：新目标、多解主组', '',
        '| 方法 | 校准样本 | 分组数 | 平均总查询 | 找全运行 | 漏项总数（含重复） |',
        '|---|---:|---:|---:|---:|---:|']
    for method in e.d.METHODS:
        for cal in e.CALIBRATIONS:
            for groups in (1, 32):
                r = next(r for r in es if r['split'] == 'fresh' and r['layer'] == 12 and r['method'] == method
                         and r['epsilon'] == .25 and r['groups'] == groups and r['calibration_count'] == cal)
                lines.append(f'| {LABELS[method]} | {cal} | {groups} | {r["mean_queries"]:.1f} | '
                    f'{r["complete_runs"]}/{r["runs"]} | {r["missed_solutions_total"]} |')
    lines += ['', '## 4. 加上有证明的审计之后，代价是多少？', '',
        '对经验方案拟排除的集合做一次独立无放回审计。若发现匹配就补查全部剩余项；否则停止。'
        '这是单次运行至少 1−δ 正确的程序，不是绝对零失败，也不是观察到零命中后的后验概率。', '',
        '| 目标组 | 方法 | δ | 平均总查询 | 找全运行 | 触发完整回退 |',
        '|---|---|---:|---:|---:|---:|']
    for cohort in ('confirmation', 'fresh'):
        for method in e.d.METHODS:
            r = next(r for r in es if r['split'] == cohort and r['layer'] == 12 and r['method'] == method
                     and r['epsilon'] == .25 and r['groups'] == 32 and r['calibration_count'] == 600)
            for a in r['audits']:
                lines.append(f'| {COHORTS[cohort]} | {LABELS[method]} | {a["delta"]} | {a["mean_queries"]:.1f} | '
                    f'{a["complete_runs"]}/{r["runs"]} | {a["fallbacks"]}/{r["runs"]} |')
    lines += ['', '旧组存在审计后仍漏解的运行，已原样保留。它们并不违反概率保证；δ>0 本来就允许小概率失败。'
        '新目标表中没有出现失败也不意味着升级为确定性保证。',
        f'汇总所有预设配置，而不只看主表，旧组共有 {audits[0]["audit_omissions_preserved_in_results"]} 次、'
        f'新目标共有 {audits[1]["audit_omissions_preserved_in_results"]} 次审计后遗漏；原始记录均保留。',
        '基准使用固定伪随机种子复现样本；概率定理针对排除提案固定之后的独立均匀审计抽样，'
        '不把某个固定种子的确定结果解释成随机保证。实际使用须满足该随机源条件。',
        '若要求确定性零遗漏，补查全部被排除候选，总查询量为 10,000。该精确回退在每个经验运行上都执行并核对。', '',
        '## 5. 已完成的理论与没有完成的理论', '',
        '| 命题 | 状态 | 适用范围 |', '|---|---|---|',
        '| 有效下界排除，补查其余候选，必然找全 | 已证明并测试 | 固定前缀、有限词表、确定性查询分类 |',
        '| 局部单侧残差界可用于安全排除 | 已证明 | 必须已有有效的组内误差上界 |',
        '| 用认证的模型/代理 Lipschitz 常数与已查锚点构造局部界 | 条件性公式已证明 | 真实模型所需紧常数未取得 |',
        '| 独立审计 + 完整回退，漏解概率至多 δ | 已证明、组合枚举核对并运行 | 残余标签固定、独立单次审计 |',
        '| 无结构黑盒、只返回验证匹配时的最坏期望查询下界 | 已证明 | 不是固定已知神经网络上的不可能性结论 |',
        '| 少量校准样本自动提供全候选误差界 | 不成立 | 本轮存在明确的实际反例 |',
        '| 实际深层 KV 可低成本认证找全并大幅加速 | 未证明 | 需要紧的结构证书或进一步明确假设 |', '',
        '完整证明、式 (7a) 的 oracle 解释、有限总体概率和单 token/全文边界见 [THEORY.md](THEORY.md)。'
        '不能把 LinFACT 的置信半径直接套到任意非线性 KV 代理；其 misspecification 分析有明确偏差与间隔条件。'
        '[原论文第 4–5 节](https://arxiv.org/html/2510.00073v1)', '',
        '## 6. 一个可以逐项核对的失败实例', '',
        '已单独保存[新目标中的第一个经验漏解案例](EXAMPLE.md)。它列出真实匹配项的距离、预测距离、校准半径与错误下界。'
        '这能直接看到：实际目标早已命中，其他符合阈值的 token 仍被错误排除；问题在误差界的有效性，而不是是否能找到一个好候选。', '',
        '## 7. 审计、数据范围与复现', '',
        f'- 独立重新拟合 {sum(a["refitted_snapshots"] for a in audits)} 个检查点，并与冻结预测/原排序核对。',
        f'- 独立检查 {sum(a["independently_checked_local_decisions"] for a in audits)} 组局部 oracle 决策、'
        f'{sum(a["independently_checked_empirical_runs"] for a in audits)} 次经验运行、'
        f'{sum(a["independently_checked_audits"] for a in audits)} 次随机审计。',
        '- oracle 有效界没有错误排除；经验方案和概率审计中的实际遗漏全部保留，不筛掉失败运行。',
        '- 小有限总体上枚举全部审计子集，与超几何组合概率逐项核对；单个隐藏匹配项的 5% 最坏情形也有精确测试。',
        '- 原始源码、数据、旧查询轨迹 SHA256 不变；新目标按真实 token ID 与旧目标去重。',
        '- 现有候选池只覆盖特定英语/数字子集；实验不构成全词表、未知前缀、长文本、Gemma INT8 或 Pixel NPU 的保证。',
        '- 本轮没有评估神经网络形式化验证器的实际成本；条件性 Lipschitz 构造未被标为已经运行成功。', '',
        '完整原始结果与所有预设配置：', '',
        '- [研究协议](README.md)、[新目标协议](NEW_TARGET_PROTOCOL.md)',
        '- [全部汇总，包括 development、两个层、两个阈值和所有预设配置](results/summary.json)',
        '- [旧案例局部界](results/local.json)、[旧案例经验方案与审计](results/empirical.json)',
        '- [新目标与完整搜索轨迹](fresh_results/cases.json)、[新目标局部界](fresh_results/local.json)、[新目标经验方案与审计](fresh_results/empirical.json)',
        '- [旧案例独立审计](results/validation.json)、[新目标独立审计](fresh_results/validation.json)', '',
        '追加新目标与报告的复现命令：', '',
        '```bash',
        'PYTHONDONTWRITEBYTECODE=1 kvidence/experiments/kv_inverse_tiny/.venv/bin/python kvidence/experiments/kv_complete_search_study_05b/fresh_targets.py',
        'PYTHONDONTWRITEBYTECODE=1 kvidence/experiments/kv_inverse_tiny/.venv/bin/python kvidence/experiments/kv_complete_search_study_05b/validate.py --fresh',
        'PYTHONDONTWRITEBYTECODE=1 kvidence/experiments/kv_inverse_tiny/.venv/bin/python kvidence/experiments/kv_complete_search_study_05b/report.py',
        '```', '',
        '## 8. 本轮研究最终判断', '',
        '本轮完成的是一条可复现的研究链：统一界诊断 → 局部单侧界诊断 → 无答案校准对照 → 新目标检验 → '
        '带证明的随机审计与精确回退 → 黑盒限制与结构条件。',
        '**目前不能声称 KVidence 已经获得“少量查询即可保证找全”的方法。** '
        '已经得到的正面结果是局部化显著提高 oracle 排除能力；已经确认的缺口是如何低成本获得有效的局部误差证书。'
        '少量校准最大值不能填补这个缺口；无结构的随机审计可以提供概率保证，但节省很有限。',
        '这项研究支持将“候选发现效率”和“完整集合证书成本”分开评价，保留原检索方法作为基线。'
        '若论文需要宣称实际模型上的可靠加速，仍须补充由网络结构支持的证书；不能把本轮条件定理、oracle 数字或未出现失败的测试代替该证明。', '',
        '## English abstract', '',
        'We evaluate complete compatible-token recovery from deep KV observations separately from target-token retrieval. '
        'Using frozen Qwen2.5-0.5B search trajectories and additional disjoint token targets, we find that prediction-distance '
        'partitions with exact one-sided residual envelopes can substantially reduce the unresolved candidate set. '
        'These privileged envelopes use the full answer table. Replacing them with finite-sample calibration maxima produces '
        'actual omissions, including after the original target has been recovered. We derive and implement a one-shot '
        'finite-population audit with exhaustive fallback, providing a per-run omission probability of at most delta without '
        'assuming proxy correctness. The singleton-omission case forces near-exhaustive audit costs. We provide conditional '
        'structural certificate formulas and a restricted black-box query lower bound, without claiming an impossibility '
        'result for fixed known neural networks. Efficient, valid local certificates for the actual deep-KV mapping remain unresolved.', '']
    (e.ROOT / 'REPORT.md').write_text('\n'.join(lines))
    write_example(empirical)
    e.d.save(e.ROOT / 'results/report_manifest.json', {
        'report_sha256': e.d.digest(e.ROOT / 'REPORT.md'), 'theory_sha256': e.d.digest(e.ROOT / 'THEORY.md'),
        'example_sha256': e.d.digest(e.ROOT / 'EXAMPLE.md'), 'report_script_sha256': e.d.digest(e.ROOT / 'report.py'),
        'all_validations_passed': True})
    print(e.ROOT / 'REPORT.md')


if __name__ == '__main__':
    main()
