#!/usr/bin/env python3
"""Post-hoc all-solutions evaluation. Does not change or rerun any search."""
import json
import statistics
from pathlib import Path

import numpy as np

import experiment as e
from report import LABELS


def main():
    folder=e.ROOT/'results'
    cases=json.loads((folder/'trials.json').read_text())
    config=json.loads((folder/'config.json').read_text())
    assert e.digest(e.ROOT/'experiment.py')==config['source_sha256']
    assert e.digest(Path(config['original_source_path']))==config['original_source_sha256']
    tables={}
    records=[]
    for c in cases:
        key=(c['seed'],c['prefix_index'],c['layer'])
        if key not in tables:
            with np.load(e.PRIOR/'data'/str(key[0])/f'prefix_{key[1]}.npz') as raw:
                tables[key]=raw[f'kv_{key[2]}'].copy()
        table=tables[key]
        target=table[c['target_index']]
        distance=np.max(np.abs(table-target),axis=1)/max(np.max(np.abs(target)),1e-12)
        for eps in (.1,.25):
            truth=distance<=eps
            record={k:c[k] for k in ('key','split','seed','trial','layer','target_index')}
            record.update({'epsilon':eps,'truth_count':int(truth.sum()),'methods':{}})
            for method,result in c['methods'].items():
                trace=result['trace']
                indices=trace['indices']
                assert len(indices)==len(set(indices))
                found=0
                total=0
                first=None
                for batch in trace['batches']:
                    total+=len(batch)
                    found+=int(truth[batch].sum())
                    if first is None and found==truth.sum():first=total
                checkpoints=[]
                for row in result['checkpoints']:
                    budget=row['budget']
                    nfound=int(truth[indices[:budget]].sum())
                    assert nfound==row[f'found_{eps:g}']
                    assert truth.sum()==row[f'truth_count_{eps:g}']
                    checkpoints.append({'budget':budget,'found':nfound,'missed':int(truth.sum())-nfound,
                        'complete':bool(nfound==truth.sum()),'recall':nfound/int(truth.sum())})
                record['methods'][method]={'checkpoints':checkpoints,'first_full_set_batch':first,
                    'first_target_batch':result['first_hit_batch'],'certified_stopping_rule':False}
            records.append(record)
    e.save(folder/'all_solutions.json',{'post_hoc':True,'new_search_runs':0,
        'definition':'all candidate IDs with normalized coordinatewise max distance <= epsilon',
        'set_counting':'only candidates verified by an actual oracle query',
        'original_source_unchanged':True,'search_source_unchanged':True,'cases':records})
    lines=['# 找全候选：事后重新评价','',
        '本附录在用户重申“最终目标是找全”后补充。它重新评价已经保存的轨迹，'
        '没有重跑、修改或调节任何搜索方法。原报告中预先固定的单目标命中指标保留，'
        '**不把本次事后评价冒充为此前预先设计的主实验。**','',
        '## 最终任务','',
        '在给定正确前缀、候选词表和距离容差下，恢复所有与观测 KV 相容的候选 token。',
        '记 `Cε = {v : max(abs(KV(v)-target))/max(abs(target)) <= ε}`。'
        '本次有 10,000 个候选，完整答案表独立定义 Cε。',
        '所有方法在这里统一按“实际查询、验证到的匹配候选”计数；不把未经查询的代理猜测算作已验证候选。'
        'LinFACT 参考版本的经验分类集合不是本附录评价对象，上一轮单独审计过它的区间和分类。','',
        '需要区分两个层次：','',
        '1. **实验上恰好找全**：利用隐藏答案事后检查，输出集合与 Cε 相等。',
        '2. **搜索中知道可以停止**：算法有可靠依据确认，所有尚未查询候选都不可能属于 Cε。',
        '第二点尚未解决；第一点也不能用“命中实际输入 token”代替。','',
        '## 确认组：每种方法最多 600 次查询','']
    for layer in (12,24):
        for eps in (.1,.25):
            selected=[r for r in records if r['split']=='confirmation' and r['layer']==layer and r['epsilon']==eps]
            multiple=[r for r in selected if r['truth_count']>1]
            sizes=[r['truth_count'] for r in selected]
            lines.extend([f'### 第 {layer} 层，ε={eps:g}','',
                f'共 {len(selected)} 个目标，真实集合大小 {min(sizes)}–{max(sizes)}；'
                f'其中 {len(multiple)} 个目标具有多个容差内候选。','',
                '| 方法 | 找全案例 @220 | 找全案例 @600 | 多解案例中找全 @600 | 漏解总数 @600 | 平均集合召回 @600 |',
                '|---|---:|---:|---:|---:|---:|'])
            for method in config['methods']:
                at=lambda r,b:next(x for x in r['methods'][method]['checkpoints'] if x['budget']==b)
                early=sum(at(r,220)['complete'] for r in selected)
                late=[at(r,600) for r in selected]
                multi=f'{sum(at(r,600)["complete"] for r in multiple)}/{len(multiple)}' if multiple else '无多解案例'
                lines.append(f'| {LABELS[method]} | {early}/{len(selected)} | '
                    f'{sum(x["complete"] for x in late)}/{len(selected)} | {multi} | '
                    f'{sum(x["missed"] for x in late)} | {statistics.mean(x["recall"] for x in late):.1%} |')
            if not multiple:
                lines.extend(['','该组全为单解；这里找全与命中一个目标在评分上重合，无法检验多解枚举能力。'])
            lines.append('')
    example=next(r for r in records if r['split']=='confirmation' and r['layer']==12 and r['epsilon']==.25
        and r['methods']['original']['first_target_batch'] is not None
        and not r['methods']['original']['checkpoints'][-1]['complete'])
    original=example['methods']['original']
    last=original['checkpoints'][-1]
    lines.extend(['## 命中一个目标为什么不够','',
        f'按顺序取第一例“命中原 token、但仍未找全”的确认案例 `{example["key"]}`（ε=0.25）：'
        f'原方法在 {original["first_target_batch"]} 次查询时命中实际输入 token；'
        f'真正相容的候选共有 {example["truth_count"]} 个，600 次时验证到 {last["found"]} 个，仍漏 {last["missed"]} 个。','',
        '## 对当前研究的判断','',
        '本轮第 12 层、ε=0.25 的多解情形更贴近“找全”问题；原方法和新策略仍有漏解。'
        '个别策略提高平均召回或找全案例数，不等于获得普遍完备性。',
        '后续应以完整集合恢复率、漏解数、首次找全预算和可信停止条件为主要评价。'
        '单目标命中率与搜索方向只作为辅助指标，不能替代目标。',
        '本附录阈值属于 FP32 归一化距离，不能解释为 INT8 的 ±4 LSB；'
        '保证范围也仅限固定正确前缀和给定 10,000 个候选。','',
        '- [逐案例找全检查点及首次找全预算](results/all_solutions.json)',
        '- [事后评分代码](evaluate_all_solutions.py)',
        '- [原始对照报告](RESULTS.md)',''])
    path=e.ROOT/'ALL_SOLUTIONS.md'
    path.write_text('\n'.join(lines))
    print(path)


if __name__=='__main__':main()
