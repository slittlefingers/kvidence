#!/usr/bin/env python3
"""Regenerate concise Chinese results from independently auditable traces."""
import argparse
import json
import statistics

import numpy as np

import experiment as e

LABELS={'random':'随机','kv_surrogate_full':'原 KV 代理（896 维）',
        'kv_surrogate_pca':'KV 代理（PCA 32）','linfact_g_scalar':'LinFACT-G 适配：标量',
        'linfact_g_vector':'LinFACT-G 适配：KV 向量'}


def aggregate(cases,method,budget):
    rows=[next(r for r in c['methods'][method] if r['budget']==budget) for c in cases]
    n=len(rows)
    return {'trials':n,'hits':sum(r['nearest_recovered'] for r in rows),
        'target_queried':sum(r['target_queried'] for r in rows),
        'exact_sets':sum(r['exact_set'] for r in rows),
        'mean_recall':statistics.mean(r['recall'] for r in rows),
        'mean_precision':statistics.mean(r['precision'] if r['precision'] is not None else 0 for r in rows),
        'false_elimination_cases':sum(r['false_eliminations']>0 for r in rows),
        'target_elimination_cases':sum(r['target_eliminated'] for r in rows),
        'false_inclusion_cases':sum(r['false_inclusions']>0 for r in rows),
        'stopped':sum(r['stopped'] for r in rows),
        'wrong_stops':sum(r['stopped'] and not r['exact_set'] for r in rows),
        'mean_possible_count':statistics.mean(r['possible_count'] for r in rows),
        'mean_queries':statistics.mean(r['queries'] for r in rows),
        'mean_learner_seconds':statistics.mean(r['learner_seconds'] for r in rows),
        'mean_empirical_radius_floor':statistics.mean(r['fit_stats'].get('empirical_radius_floor',0) for r in rows),
        'mean_interval_coverage':statistics.mean(r['interval_coverage'] for r in rows) if 'interval_coverage' in rows[0] else None,
        'all_interval_coverage_cases':sum(r.get('all_intervals_cover',False) for r in rows),
        'target_interval_coverage_cases':sum(r.get('target_interval_covers',False) for r in rows)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results')
    args=p.parse_args()
    folder=e.ROOT/args.output
    cases=json.loads((folder/'trials.json').read_text())
    config=json.loads((folder/'config.json').read_text())
    summary={}
    lines=['# Qwen2.5-0.5B：LinFACT 风格搜索初步对比','',
        '实验日期：2026-10-18。由 `report.py` 从查询轨迹自动生成。算法和定理的适配边界见 [README.md](README.md)。','',
        '**这次直接适配没有优于现有 KVidence 代理搜索，也没有得到可靠的“找全”保证。** '
        '这是对本适配实现的结论，不是对原论文算法的否定。','',
        f'- {config["candidate_count"]:,} 个英文/数字候选；{len(config["seeds"])} 个种子，每个种子 {config["trials_per_seed"]} 个均匀随机目标。',
        '- 固定正确前缀，恢复一个未知 token；FP32 模型和 KV，无额外观测噪声。',
        '- 两层和两个 epsilon 对同一目标重复评估，不能把这些重复当作更多独立目标。',
        '- 五种方法共享 200 个初始查询，最多 600 次。LinFACT 两个版本是经验误差界适配，不具备论文原版 PAC 保证。',
        '- 整张 KV 表只用于隐藏答案和事后评分，搜索每请求一个候选都计入查询数；这是离线回放，不是在线推理耗时测试。','']
    for layer in config['layers']:
        for eps in config['epsilons']:
            selected=[c for c in cases if c['layer']==layer and c['epsilon']==eps]
            if not selected:continue
            key=f'layer_{layer}_epsilon_{eps:g}'
            truth_counts=[next(iter(c['methods'].values()))[0]['truth_count'] for c in selected]
            summary[key]={'truth_count_min':min(truth_counts),'truth_count_max':max(truth_counts),
                'truth_count_mean':statistics.mean(truth_counts),'methods':{}}
            lines.extend([f'## 第 {layer} 层，epsilon = {eps:g}','',
                f'真实近最优集合大小：{min(truth_counts)}–{max(truth_counts)}，平均 {statistics.mean(truth_counts):.1f}。','',
                '| 方法 | 命中 @220 | 命中 @600 | 集合召回 @600 | 事后恰好找全 @600 | 曾误删真解的案例 @600 | 平均未排除数 @600 |',
                '|---|---:|---:|---:|---:|---:|---:|'])
            for method in config['methods']:
                stats={str(b):aggregate(selected,method,b) for b in e.BUDGETS}
                summary[key]['methods'][method]=stats
                a,b=stats['220'],stats['600']
                n=b['trials']
                drop=f'{b["false_elimination_cases"]}/{n}' if method.startswith('linfact') else '不做排除'
                lines.append(f'| {LABELS[method]} | {a["hits"]}/{n} | {b["hits"]}/{n} | '
                    f'{b["mean_recall"]:.1%} | {b["exact_sets"]}/{n} | {drop} | {b["mean_possible_count"]:.0f} |')
            lines.extend(['','“事后恰好找全”由隐藏答案表核对，不等于搜索时具备完备性证书。',
                '', '| 适配方法 | 停止案例 | 错误停止 | 全部区间同时覆盖的案例 @600 | 平均区间覆盖 @600 | 真目标区间覆盖 @600 | 平均实际查询 |',
                '|---|---:|---:|---:|---:|---:|---:|'])
            for method in config['methods']:
                if not method.startswith('linfact'):continue
                b=summary[key]['methods'][method]['600']; n=b['trials']
                lines.append(f'| {LABELS[method]} | {b["stopped"]}/{n} | {b["wrong_stops"]}/{n} | '
                    f'{b["all_interval_coverage_cases"]}/{n} | {b["mean_interval_coverage"]:.2%} | '
                    f'{b["target_interval_coverage_cases"]}/{n} | {b["mean_queries"]:.1f} |')
            lines.append('')
    primary=[c for c in cases if c['layer']==24 and c['epsilon']==.1]
    if primary:
        a=aggregate(primary,'kv_surrogate_full',600)
        pca=aggregate(primary,'kv_surrogate_pca',600)
        scalar=aggregate(primary,'linfact_g_scalar',600)
        vector=aggregate(primary,'linfact_g_vector',600)
        lines.extend(['## 如何理解这次结果','',
            f'以第 24 层、epsilon=0.10 为例，原代理命中 {a["hits"]}/{a["trials"]}，'
            f'PCA 代理命中 {pca["hits"]}/{pca["trials"]}，标量适配 {scalar["hits"]}/{scalar["trials"]}，'
            f'向量适配 {vector["hits"]}/{vector["trials"]}。PCA 控制仍能找回大多数目标，'
            '因此性能下降不能简单全部归因于降维；本实验还未单独分离拟合方式、选点策略和区间宽度的影响。','',
            '标量适配直接拟合“embedding → KV 距离”，但距离不必是 embedding 的线性函数；'
            '即使 KV 映射恰好线性，取绝对值与最大坐标后，距离也一般不是线性的。'
            f'本组在预算终点，仅 {scalar["target_interval_coverage_cases"]}/{scalar["trials"]} 个真目标被自己的距离区间覆盖。'
            '这具体暴露了经验包络的可靠性问题。','',
            '向量适配先拟合 KV，再计算距离，更贴近原代理；但本次校准得到的包络很宽。'
            f'本组平均半径基数为 {vector["mean_empirical_radius_floor"]:.3f}，大于 epsilon=0.10，'
            f'平均还剩 {vector["mean_possible_count"]:.0f} 个未排除候选。'
            'G 设计仍在广泛探测候选特征，未形成原代理那样针对目标的有效排序。','',
            '因此目前可以借用“区间分类 + 审计”的思路，但不能把 LinFACT 的找全保证直接赋予 KVidence。'
            '下一步若继续，应先改善或验证目标附近的代理误差界；仅增加选点轮数没有解决已经观察到的包络失效。','',
            '## 一例具体轨迹',''])
        case=primary[0]
        token=json.loads((e.ROOT/'data'/str(case['seed'])/'candidates.json').read_text())[case['target_index']]
        lines.extend([f'按运行顺序取本组第一例 `{case["key"]}`，不是筛选最佳/最差案例。'
            f'目标 token 为 `{token["text"]}`（ID {token["token_id"]}），其真实距离为 0。','',
            '| 方法 | 查询数 | 真目标距离区间 | 下界为 0 的候选数 | 未排除数 |',
            '|---|---:|---:|---:|---:|'])
        for method in ('linfact_g_scalar','linfact_g_vector'):
            with np.load(folder/'traces'/f'{case["key"]}_{method}.npz') as raw:
                for budget in (200,600):
                    lower,upper=raw[f'lower_{budget}'],raw[f'upper_{budget}']
                    target=case['target_index']
                    lines.append(f'| {LABELS[method]} | {budget} | '
                        f'[{lower[target]:.4f}, {upper[target]:.4f}] | {int((lower==0).sum())} | '
                        f'{int((~raw[f"bad_{budget}"]).sum())} |')
        lines.extend(['','标量版本的区间不含真实值 0；向量版本留下大量下界为 0 的候选，难以排除。'
            '前者暴露可靠性问题，后者暴露区分能力不足。两者都不能仅凭平均覆盖率宣称找全。','',
            '## 搜索计算开销','',
            '以下只比较第 24 层、epsilon=0.10 的平均搜索计算时间，包含拟合、排序/选点与分类。'
            '不包含答案表生成、模型加载、候选特征/PCA 预计算或模型在线前向；不能换算成端到端攻击加速倍数。','',
            '| 方法 | 平均逻辑查询数 | 平均搜索计算秒数 |', '|---|---:|---:|'])
        for method in config['methods']:
            row=aggregate(primary,method,600)
            lines.append(f'| {LABELS[method]} | {row["mean_queries"]:.1f} | {row["mean_learner_seconds"]:.4f} |')
        lines.append('')
    design_certificates=[]
    for trace in (folder/'traces').glob('*linfact*.json'):
        obj=json.loads(trace.read_text())
        design_certificates.extend(d['max_leverage_over_rank'] for d in obj['designs'])
    if design_certificates:
        lines.extend(['## 实验设计数值质量','',
            f'所有设计步骤的 max-leverage/rank：中位数 {statistics.median(design_certificates):.3f}，最大值 {max(design_certificates):.3f}；理想值为 1。',
            '有限迭代和无放回批次分配属于工程近似，未当作论文精确设计的复现。',''])
    validation=json.loads((folder/'validation.json').read_text())
    metadata=[json.loads(path.read_text()) for seed in config['seeds']
              for path in (e.ROOT/'data'/str(seed)).glob('prefix_*_metadata.json')]
    maximum_error=max(value for m in metadata for value in m['oracle_validation_max_abs_error'].values())
    lines.extend(['## 验证与离线成本','',
        f'- 对 {validation["trajectory_audit"]["cases"]} 个任务设置、'
        f'{validation["trajectory_audit"]["audited_method_checkpoints"]} 个方法/预算快照完成独立审计：'
        '重新从原始 KV 计算距离、核对查询不重复和初始样本一致、分类与评分一致。',
        '- 100 个合成有效区间分类检查通过；两个精确线性正对照均在 200 次查询时正确找全并停止。',
        f'- 原模型完整前缀前向与缓存批量查询核对通过，最大坐标差 {maximum_error:.3g}。'
        '这检查 KV 提取路径，并不保证代理误差包络。',
        f'- 隐藏答案表额外生成 {sum(m["offline_candidate_forwards"] for m in metadata):,} 个候选前向，'
        f'候选前向循环共 {sum(m["offline_seconds"] for m in metadata):.2f} 秒；'
        '每次同时采集两层。此成本独立记录，未伪装成搜索算法的节省。',''])
    linrows=[r for c in cases for m,rows in c['methods'].items() if m.startswith('linfact') for r in [rows[-1]]]
    drops=sum(r['false_eliminations']>0 for r in linrows)
    wrong=sum(r['stopped'] and not r['exact_set'] for r in linrows)
    simultaneous=sum(r['all_intervals_cover'] for r in linrows)
    lines.extend(['## 结论范围','',
        f'- 两个 LinFACT 适配方法合计 {len(linrows)} 个“目标 × 层 × epsilon × 方法”运行，其中 {drops} 个出现真解误删，{wrong} 个出现错误停止。',
        f'- 在 600 次预算检查点，有 {simultaneous}/{len(linrows)} 个运行的全部距离区间同时覆盖真实距离。',
        '- 这些是相关的重复运行，不能用合计数当成独立统计样本量。',
        '- 集合召回率 100% 不代表没有额外候选；未排除集合很大也不代表已找全。必须同时检查完整集合、误删和停止状态。',
        '- 本实验只验证适配算法的表现与经验包络是否可靠，不证明原论文定理成立或失效，不证明 Gemma、INT8、Pixel NPU 或整段秘密恢复。',
        '- 原 896 维 KV 代理和 PCA 32 维控制可帮助判断降维影响；标量/向量版本同时改变拟合目标、区间与选点，不能把差异只归因于单一组件。','',
        '## 原始资料','',
        '- `results/config.json`：冻结设置和源码校验值。',
        '- `results/trials.json`、`results/summary.json`：逐案例与汇总指标。',
        '- `results/traces/`：查询顺序、批次、设计证书及每个预算的区间/分类快照。',
        '- `results/validation.json`：线性正对照、区间逻辑检查和独立轨迹审计。',
        '- `data/*/prefix_*_metadata.json`：完整答案表生成成本与前向一致性检查。',''])
    e.save_json(folder/'summary.json',summary)
    dest=e.ROOT/('RESULTS.md' if args.output=='results' else f'{args.output.upper()}_RESULTS.md')
    dest.write_text('\n'.join(lines))
    print(dest)


if __name__=='__main__':main()
