#!/usr/bin/env python3
"""Pool the two completed runs, keeping layers and target distributions separate."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
runs = [json.loads((root / d / 'metrics.json').read_text()) for d in ('results', 'replication')]
combined = {'model_id': runs[0]['model_id'], 'model_revision': runs[0]['model_revision'],
            'seeds': [r['seed'] for r in runs], 'candidate_count': runs[0]['candidate_count'],
            'embedding_table_shape': runs[0]['embedding_table_shape'], 'summary': {}}
for r in runs:
    assert r['candidate_count'] == combined['candidate_count']
    assert r['embedding_table_shape'] == combined['embedding_table_shape']
    assert r['budgets'] == runs[0]['budgets'] and r['layers'] == runs[0]['layers']
for layer in runs[0]['summary']:
    combined['summary'][layer] = {}
    for scenario in runs[0]['summary'][layer]:
        combined['summary'][layer][scenario] = {}
        for method in runs[0]['summary'][layer][scenario]:
            combined['summary'][layer][scenario][method] = {}
            for budget in runs[0]['summary'][layer][scenario][method]:
                records = [r['summary'][layer][scenario][method][budget] for r in runs]
                n = sum(x['trials'] for x in records)
                hits = sum(x['exact_recoveries'] for x in records)
                combined['summary'][layer][scenario][method][budget] = {
                    'trials': n, 'exact_recoveries': hits, 'exact_recovery_rate': hits / n}
(root / 'combined_metrics.json').write_text(json.dumps(combined, indent=2) + '\n')
names = {'random': '随机', 'language_probability': '语言概率',
         'kv_surrogate': 'KV 方向', 'shuffled_embedding_control': '打乱 embedding 对照'}
lines = ['## 正式结果（两个种子合并）', '',
         '每种目标分布、每个观察层合计 32 个目标；同一目标的两个层分别统计。以下均为精确 token 恢复数，不是仅降低 KV 距离。', '']
for scenario, title in [('uniform_target', '均匀随机目标'),
                        ('model_probability_target', '按模型语言概率抽样的目标')]:
    lines += ['### ' + title, '', '| 层 | 方法 | 200 | 220 | 240 | 300 | 400 | 600 |',
              '|---|---|---|---|---|---|---|---|']
    for layer, scenarios in combined['summary'].items():
        for method, budgets in scenarios[scenario].items():
            values = [f"{budgets[str(b)]['exact_recoveries']}/{budgets[str(b)]['trials']}" for b in runs[0]['budgets']]
            lines.append('| ' + ' | '.join([layer, names[method]] + values) + ' |')
    lines += ['']
(root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
