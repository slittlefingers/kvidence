#!/usr/bin/env python3
"""Pool seeds and report paired effects of runtime KV quantization."""
import json
from runtime import ROOT
from summary import summarize, BUDGETS, METHODS

runs={}
for seed in (20261017,20261018):
    for mode in ('native','int8'):
        path=ROOT/'results'/f'seed_{seed}_{mode}'
        metrics=json.loads((path/'metrics.json').read_text())
        trials=json.loads((path/'trials.json').read_text())
        runs[(seed,mode)]=(metrics,trials,summarize(trials))
combined={'model_id':runs[(20261017,'native')][0]['model_id'],
    'model_revision':runs[(20261017,'native')][0]['model_revision'],
    'seeds':[20261017,20261018],'candidate_count':10000,
    'initial_queries':200,'budgets':list(BUDGETS),'modes':{},'paired_outcomes':{}}
names={'random':'随机','language_probability':'语言概率','kv_surrogate':'KV 方向','shuffled_embedding_control':'打乱 embedding'}
lines=['# Gemma 4 W8：BF16 KV 与运行时 KV8 结果','',
       '每种 KV 模式、每种目标分布、每层合计 32 个目标（两个种子各 16 个）。两层观察同一目标，不能按 64 个独立目标统计。下表为最终真实 KV 最近邻精确恢复数。','']
for mode,title in [('native','W8 + BF16 KV'),('int8','W8 + KV8')]:
    combined['modes'][mode]={}
    lines += ['## '+title,'']
    for scenario,caption in [('uniform_target','均匀目标'),('model_probability_target','模型概率目标（两组相同目标）')]:
        lines += ['### '+caption,'','| 层 | 方法 | 200 | 220 | 240 | 300 | 400 | 600 |','|---|---|---|---|---|---|---|---|']
        for layer in ['10','15']:
            combined['modes'][mode].setdefault(layer,{})[scenario]={}
            for method in METHODS:
                result={}
                for budget in BUDGETS:
                    entries=[runs[(s,mode)][2][layer][scenario][method][str(budget)] for s in combined['seeds']]
                    n=sum(x['trials'] for x in entries)
                    exact=sum(x['exact_recoveries'] for x in entries)
                    hits=sum(x['hits'] for x in entries)
                    result[str(budget)]={'trials':n,'exact_recoveries':exact,'hits':hits,
                        'exact_recovery_rate':exact/n,'target_queried_but_nearest_kv_wrong':hits-exact}
                combined['modes'][mode][layer][scenario][method]=result
                cells=[f"{result[str(b)]['exact_recoveries']}/{result[str(b)]['trials']}" for b in BUDGETS]
                lines += ['| '+' | '.join([layer,names[method]]+cells)+' |']
        lines += ['']
lines += ['## 配对比较：KV 方向','',
          '按相同种子、目标、前缀与初始候选配对。差异包括 KV8 存储与量化注意力路径。','',
          '| 分布 | 层 | 预算 | 两组都成功 | 仅 BF16 成功 | 仅 KV8 成功 | 两组都失败 |',
          '|---|---|---|---|---|---|---|']
for scenario,caption in [('uniform_target','均匀'),('model_probability_target','概率')]:
    for layer in [10,15]:
        by_budget={}
        for budget in BUDGETS:
            counts={'both_success':0,'native_only_success':0,'int8_only_success':0,'both_fail':0}
            for seed in combined['seeds']:
                native=[r for r in runs[(seed,'native')][1] if r['scenario']==scenario and r['layer']==layer]
                int8=[r for r in runs[(seed,'int8')][1] if r['scenario']==scenario and r['layer']==layer]
                for a,b in zip(native,int8):
                    assert a['target_token_id']==b['target_token_id'] and a['initial_indices']==b['initial_indices']
                    def recovered(r):
                        t=r['methods']['kv_surrogate']
                        best=min(range(budget),key=lambda i:t['normalized_mse'][i])
                        return t['indices'][best]==r['target_index']
                    good_a,good_b=recovered(a),recovered(b)
                    key='both_success' if good_a and good_b else 'native_only_success' if good_a else 'int8_only_success' if good_b else 'both_fail'
                    counts[key]+=1
            by_budget[str(budget)]=counts
            if budget in [220,600]:
                lines += ['| '+' | '.join(map(str,[caption,layer,budget]+list(counts.values())))+' |']
        combined['paired_outcomes'].setdefault(str(layer),{})[scenario]=by_budget
(ROOT/'combined_metrics.json').write_text(json.dumps(combined,indent=2)+'\n')
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
