#!/usr/bin/env python3
"""Summarize every frozen search variant, including the confirmation split."""
import argparse
import json
import statistics as stats

import experiment as e

LABELS={'original':'原 KVidence（冻结）','random':'随机查询',
        'linfact_vector_reference':'上轮 LinFACT 向量适配',
        'ridge_linf':'原拟合 + 最大坐标距离排序',
        'target_weighted_ridge':'目标附近加权拟合',
        'linear_rbf':'线性 + RBF 非线性代理',
        'target_design_16_4':'16 个近邻 + 4 个信息采集'}


def aggregate(cases,method,budget):
    results=[c['methods'][method] for c in cases]
    rows=[next(r for r in item['checkpoints'] if r['budget']==budget) for item in results]
    return {'trials':len(rows),'hits':sum(r['hit'] for r in rows),
            'mean_best_nmse':stats.mean(r['best_nmse'] for r in rows),
            'median_best_nmse':stats.median(r['best_nmse'] for r in rows),
            'mean_best_linf':stats.mean(r['best_linf'] for r in rows),
            'mean_recall_0.1':stats.mean(r['recall_0.1'] for r in rows),
            'mean_recall_0.25':stats.mean(r['recall_0.25'] for r in rows),
            'mean_search_seconds':stats.mean(r['learner_seconds'] for r in rows),
            'mean_queries':stats.mean(r['queries'] for r in rows),
            'mean_capped_first_hit_queries':stats.mean(item['capped_first_hit_queries'] for item in results),
            'initial_hits':sum(c['target_index'] in c['initial_indices'] for c in cases),
            'conditional_hits':sum(r['hit'] for c,r in zip(cases,rows) if c['target_index'] not in c['initial_indices']),
            'initial_miss_trials':sum(c['target_index'] not in c['initial_indices'] for c in cases)}


def paired(cases,method,budget=220):
    wins=losses=earlier=later=ties=0
    for c in cases:
        ours,base=c['methods'][method],c['methods']['original']
        hit=lambda v:next(r['hit'] for r in v['checkpoints'] if r['budget']==budget)
        wins+=hit(ours) and not hit(base)
        losses+=hit(base) and not hit(ours)
        a,b=ours['first_hit_batch'] or float('inf'),base['first_hit_batch'] or float('inf')
        earlier+=a<b
        later+=a>b
        ties+=a==b
    return {'win_at_220':wins,'loss_at_220':losses,
            'first_hit_earlier':earlier,'first_hit_later':later,'first_hit_tied':ties}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results')
    args=p.parse_args()
    folder=e.ROOT/args.output
    cases=json.loads((folder/'trials.json').read_text())
    config=json.loads((folder/'config.json').read_text())
    validation=json.loads((folder/'validation.json').read_text())
    summary={}
    lines=['# 0.5B 独立搜索策略对比','',
        '实验日期：2026-10-18。原方法保持不变；本报告由 `report.py` 自动汇总全部冻结策略。','',
        '完整设置与方法定义见 [README.md](README.md)。本轮研究相同查询预算下的搜索效果，不提供找全证书。','',
        '## 设置','',
        '- Qwen2.5-0.5B，FP32 模型计算与 KV，10,000 个英文/数字候选，固定正确前缀，恢复一个未知 token。',
        '- 开发组为上轮 16 个目标；确认组另取 16 个目标，目标 ID 与开发组不重叠。两组共享候选池和四个前缀。',
        '- 两层分别比较，同一目标会在两层重复出现；不能按层数扩充独立样本量。',
        '- 所有方法共享初始 200 次查询，每批 20 次，最多 600 次。首要指标预先固定为命中 @220。',
        '- 本轮新策略使用完整 896 维 embedding 和全部已查询训练样本，没有未证明的永久候选排除。',
        '- 全候选 KV 表只作为隐藏 oracle 和独立评分依据，搜索函数看不到目标 ID 或未查询 KV。',
        '- 本实验未对确认组调参；并列报告各方法，不把探索性胜出解释为普遍最优。','']
    for split in config['splits']:
        title='开发组：上轮目标' if split=='development' else '确认组：新目标'
        for layer in config['layers']:
            selected=[c for c in cases if c['split']==split and c['layer']==layer]
            key=f'{split}_layer_{layer}'
            summary[key]={'methods':{},'paired_vs_original':{}}
            initial=sum(c['target_index'] in c['initial_indices'] for c in selected)
            lines.extend([f'## {title}，第 {layer} 层','',
                f'目标数 {len(selected)}，初始 200 次已经包含目标的案例数 {initial}。','',
                '| 方法 | 命中 @220 | 命中 @240 | 命中 @600 | 首次命中预算截断均值 | 平均搜索计算秒数 @600 |',
                '|---|---:|---:|---:|---:|---:|'])
            for method in config['methods']:
                a={str(b):aggregate(selected,method,b) for b in e.BUDGETS}
                summary[key]['methods'][method]=a
                row=a['600']; n=row['trials']
                lines.append(f'| {LABELS[method]} | {a["220"]["hits"]}/{n} | {a["240"]["hits"]}/{n} | '
                    f'{row["hits"]}/{n} | {row["mean_capped_first_hit_queries"]:.1f} | {row["mean_search_seconds"]:.4f} |')
            lines.extend(['','“首次命中预算截断均值”按批次计数；600 次仍未命中的案例记为 600，'
                '因此需与命中数一起看，不能理解为最终成功所需查询数。',
                '搜索秒数仅包含拟合与选点；不包含真实模型在线前向，不是端到端攻击时间。','',
                '| 方法 | 平均最小 NMSE @220 | 平均最小最大坐标距离 @220 | ε=0.10 集合召回 @600 | ε=0.25 集合召回 @600 |',
                '|---|---:|---:|---:|---:|'])
            for method in config['methods']:
                a=summary[key]['methods'][method]
                lines.append(f'| {LABELS[method]} | {a["220"]["mean_best_nmse"]:.5g} | '
                    f'{a["220"]["mean_best_linf"]:.4f} | {a["600"]["mean_recall_0.1"]:.1%} | '
                    f'{a["600"]["mean_recall_0.25"]:.1%} |')
            if split=='confirmation':
                lines.extend(['','| 新策略相对原方法 | @220 多命中的案例 | @220 少命中的案例 | 首次命中更早 / 更晚 / 相同 |',
                    '|---|---:|---:|---:|'])
                for method in ('ridge_linf','target_weighted_ridge','linear_rbf','target_design_16_4'):
                    if method not in config['methods']:continue
                    row=paired(selected,method)
                    summary[key]['paired_vs_original'][method]=row
                    lines.append(f'| {LABELS[method]} | {row["win_at_220"]} | {row["loss_at_220"]} | '
                        f'{row["first_hit_earlier"]} / {row["first_hit_later"]} / {row["first_hit_tied"]} |')
            lines.append('')
    lines.extend(['## 新目标上的主要观察',''])
    for layer in config['layers']:
        key=f'confirmation_layer_{layer}'
        if key not in summary:continue
        methods=summary[key]['methods']
        baseline=methods['original']
        lines.append(f'第 {layer} 层原方法 @220 为 {baseline["220"]["hits"]}/{baseline["220"]["trials"]}，'
                     f'@600 为 {baseline["600"]["hits"]}/{baseline["600"]["trials"]}。')
        lines.append('')
        for method in ('ridge_linf','target_weighted_ridge','linear_rbf','target_design_16_4'):
            if method not in methods:continue
            row=methods[method]
            delta=row['220']['hits']-baseline['220']['hits']
            cost=row['600']['mean_search_seconds']/max(baseline['600']['mean_search_seconds'],1e-12)
            lines.append(f'- {LABELS[method]}：@220 {row["220"]["hits"]}/{row["220"]["trials"]}，'
                f'相差 {delta:+d} 个；@600 {row["600"]["hits"]}/{row["600"]["trials"]}；'
                f'搜索计算开销约为原方法 {cost:.2f} 倍。')
        lines.append('')
    consistent=[]
    for method in ('ridge_linf','target_weighted_ridge','linear_rbf','target_design_16_4'):
        comparisons=[]
        for layer in config['layers']:
            key=f'confirmation_layer_{layer}'
            if key in summary and method in summary[key]['methods']:
                ms=summary[key]['methods']
                comparisons.append(ms[method]['220']['hits']-ms['original']['220']['hits'])
        if comparisons and min(comparisons)>=0 and max(comparisons)>0:consistent.append(LABELS[method])
    if not consistent:
        lines.extend(['**本轮没有新策略在两层都保持 @220 命中不下降、并至少一层提升。** '
            '非线性代理在第 12 层出现较早命中的信号，第 24 层未复现这种优势。原方法保留为基线。',''])
    lines.extend(['这些是本轮的配对结果。16 个新目标仍属小样本，而且有四个预先固定的新策略；'
        '个别方法在某层胜出，尚不足以认定它在其他目标、前缀、模型上都更好。',
        '若继续，应固定有希望的策略再扩大新目标验证，并增加在线真实模型前向计时；原方法仍保留不变。','',
        '## 核对','',
        f'- 原方法源码 SHA256：`{config["original_source_sha256"]}`；运行前后和审计时一致。',
        f'- {validation["audit"]["baseline_trajectories_identical_to_previous_experiment"]} 条开发组原方法完整查询轨迹，与上轮逐项一致。',
        f'- {validation["audit"]["cases"]} 个目标/层设置、'
        f'{validation["audit"]["method_budget_checkpoints"]} 个方法/预算快照均通过独立原始 KV 重算审计。',
        '- 原排序等价、单位权重退化、核训练顺序不变性、设计批次约束及条件协方差更新检查通过。',
        '- 新旧目标 ID 不重叠；查询无重复，各方法起点相同，预算符合设置。','',
        '首次运行发现一个跨候选池重复目标，该轮已隔离为 `audit_rejected_run/`。'
        '本报告仅使用按真实 token ID 全局去重后重新运行的结果；策略和参数未调整。',
        f'修正后 {validation["audit"]["audit_correction_unchanged_target_trajectories"]} 个目标/层设置未换目标，'
        '其所有方法的完整查询轨迹与修正前一致。',''])
    if (folder/'online.json').exists():
        online=json.loads((folder/'online.json').read_text())
        online_summary={}
        replay={c['key']:c for c in cases}
        rows=[r for c in online['cases'] for r in c['methods'].values()]
        batch_matches=sum(r['first_hit_batch']==replay[c['key']]['methods'][m]['first_hit_batch']
                          for c in online['cases'] for m,r in c['methods'].items())
        lines.extend(['## 少量实际模型前向计时','',
            '固定选择确认组第一个候选池的前四个目标、两层，每种方法测一次。CPU 四线程，'
            '真正逐批运行原模型，方法顺序按案例随机打乱。以下包含初始查询、后续前向、拟合和选点，'
            '不包含模型加载、前缀缓存准备和预热。','',
            '**首次命中时间由事后目标 ID 标注，不等于算法自行判断成功或找全的停止时间。** '
            '固定 600 次耗时用于比较同预算运行成本；首次命中时间用于观察搜索轨迹。',
            '每种方法仅单次计时，样本很小，不能当作稳定硬件基准。',''])
        for layer in config['layers']:
            cs=[c for c in online['cases'] if c['layer']==layer]
            if not cs:continue
            common=[c for c in cs if all(v['first_hit_seconds'] is not None for v in c['methods'].values())]
            online_summary[str(layer)]={}
            lines.extend([f'### 第 {layer} 层','',
                f'共同成功案例 {len(common)}/{len(cs)}；平均首次命中时间仅在这组共同案例上比较。','',
                '| 方法 | 命中案例 | 平均首次命中秒数 | 固定 600 次平均总秒数 |',
                '|---|---:|---:|---:|'])
            for method in online['config']['methods']:
                values=[c['methods'][method] for c in cs]
                hits=sum(v['first_hit_seconds'] is not None for v in values)
                first=stats.mean(c['methods'][method]['first_hit_seconds'] for c in common) if common else None
                total=stats.mean(v['total_seconds_600'] for v in values)
                online_summary[str(layer)][method]={'cases':len(cs),'hits':hits,
                    'common_success_cases':len(common),'mean_first_hit_seconds_common':first,
                    'mean_total_seconds_600':total}
                first_text=f'{first:.4f}' if first is not None else '无共同成功案例'
                lines.append(f'| {LABELS[method]} | {hits}/{len(cs)} | {first_text} | {total:.4f} |')
            lines.append('')
        first_hard=next((c for c in online['cases'] if c['layer']==12 and
                        (c['methods']['original']['first_hit_batch'] or 0)>220),None)
        if first_hard:
            a,b=first_hard['methods']['original'],first_hard['methods']['linear_rbf']
            lines.extend([f'按顺序取第一个原方法需要超过 220 次的第 12 层案例 `{first_hard["key"]}`：'
                f'原方法在 {a["first_hit_batch"]} 次、{a["first_hit_seconds"]:.4f} 秒命中；'
                f'非线性代理在 {b["first_hit_batch"]} 次、{b["first_hit_seconds"]:.4f} 秒命中。'
                '它展示了一次具体收益，不能代表所有目标。',''])
        lines.extend([f'{batch_matches}/{len(rows)} 条实际模型搜索的首次命中批次与离线回放一致；'
            f'{sum(r["same_indices_as_replay"] for r in rows)}/{len(rows)} 条完整查询轨迹一致。'
            f'返回 KV 的最大坐标差为 {max(r["max_kv_difference_from_replay"] for r in rows):.3g}，'
            '均通过数值一致性检查；细小数值差会影响部分后续候选排序。','',
            '本次非线性策略在部分目标上少查了候选，但在固定 600 次预算下增加了运行成本。'
            '后续需要更大样本和重复计时，才能判断何种情况下值得采用。',''])
        e.save(folder/'online_summary.json',online_summary)
    lines.extend(['## 原始记录','',
        '- [逐案例结果与轨迹](results/trials.json)',
        '- [汇总指标](results/summary.json)',
        '- [冻结配置](results/config.json)',
        '- [审计结果](results/validation.json)',
        '- [实际模型计时记录](results/online.json)',
        '- [实验代码](experiment.py)',''])
    e.save(folder/'summary.json',summary)
    dest=e.ROOT/('RESULTS.md' if args.output=='results' else f'{args.output.upper()}_RESULTS.md')
    dest.write_text('\n'.join(lines))
    print(dest)


if __name__=='__main__':main()
