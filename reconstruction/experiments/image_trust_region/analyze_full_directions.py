"""Matched 45,001-query comparison: 64 directions vs complete 768 input basis."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump


def load(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    report=json.loads((folder/'result.json').read_text())
    trace=json.loads((folder/'optimization.json').read_text())
    return protocol,report,trace


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[17,29])
    p.add_argument('--directory',type=Path,default=Path(__file__).parent/'full_directions_20261021')
    args=p.parse_args();records=[]
    for seed in args.seeds:
        root=args.directory.parent/'results'
        old=load(root/f'0001_geometry300_seed{seed}_20261021')
        full=load(root/f'0001_geometry768_seed{seed}_20261021')
        for key in ('initial_sha256','observation','seed','lr','perturbation','geometry_source_sha256'):
            if old[0][key]!=full[0][key]:raise ValueError('Unmatched protocol: '+key)
        if old[1]['initial_rmse']!=full[1]['initial_rmse'] or old[1]['queries']!=full[1]['queries']:
            raise ValueError('Unmatched initial measurement or budget')
        if not full[1]['initial_exact_reference'] or not full[1]['repeat_exact']:raise ValueError('Replay/repeatability failed')
        for label,data in (('64',old),('768',full)):
            proto,report,trace=data
            records.append(dict(seed=seed,directions=int(label),initial_rmse=report['initial_rmse'],
                best_rmse=report['best_rmse'],queries=report['queries'],rounds=report['rounds'],
                seconds=report['optimizer_seconds'],oracle_seconds=report['oracle_seconds'],
                passes_query=report['passes_query'],repeat_exact=report['repeat_exact'],
                first_reaches_old_best=next((dict(queries=r['queries'],seconds=r['elapsed_seconds']) for r in trace if r['best_rmse']<=old[1]['best_rmse']),None),
                direction_counts=[r.get('directions',proto['directions']) for r in trace],
                checkpoints=[dict(queries=r['queries'],rmse=r['best_rmse'],seconds=r['elapsed_seconds']) for r in trace]))
    args.directory.mkdir(parents=True,exist_ok=True)
    dump(args.directory/'summary.json',dict(records=records,note='Single sample, first image slot; equal search-candidate budget, query-progress schedules.'))
    lines=['# 完整 768 方向与 64 方向搜索对照', '',
        '材料为 image_chat_100/0001，第 5 层，第一个图片位置（position 5）。',
        '搜索变量仍为 768 维 float32 pre-adapter 特征；使用真实手机 vision adapter 和 NPU。',
        '完整版本每轮用 768 个正交方向，测量正负 1536 个探针，另测 22 个更新候选。',
        '候选分批执行，每个物理批次最多 128 条，全部保持相同前缀和位置。', '',
        '| Seed | 方向数 | 初始 RMSE | 最佳 RMSE | 查询数 | 更新轮数 | 秒 | 通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in records:lines.append(f"| {r['seed']} | {r['directions']} | {r['initial_rmse']:.5f} | {r['best_rmse']:.5f} | {r['queries']} | {r['rounds']} | {r['seconds']:.2f} | {r['passes_query']} |")
    lines += ['', '## 本次结果', '']
    for seed in args.seeds:
        a,b=[r for r in records if r['seed']==seed]
        change=100*(a['best_rmse']-b['best_rmse'])/a['best_rmse']
        lines.append(f'- Seed {seed}：同预算 RMSE 降低 {change:.2f}%。')
        if b['first_reaches_old_best']:
            lines.append(f"  首次达到旧版最终误差需要 {b['first_reaches_old_best']['queries']} 次查询。")
    lines += ['', '## 比较口径', '',
        '两种方法使用相同观测、初始输入、初始 KV、扰动初值、学习率初值与几何求解器。',
        '64 方向使用已完成的 300 轮实验；其日程等价于按候选预算进度推进。',
        '768 方向通过 --max-queries 45001 同样按查询进度推进日程，而非把 300 轮日程直接套在 29 轮上。',
        '在 45,001 次预算内，768 版本完成 28 个完整 768 方向轮次和一个 677 方向的尾轮。',
        '64 版本完成 300 个完整轮次。因此比较的是同查询预算的恢复效果，不是同轮数的效果。',
        '随机种子和初始输入相同，但维度不同使 QR 抽样和搜索轨迹不同。', '',
        '完整探针增加同一中心的方向覆盖，但更新机会减少，也不会消除非线性或量化误差。',
        '不能把完整 Jacobian 的方向覆盖直接等同于更低的最终恢复误差。', '',
        '32 次最佳输入重复性测量和 worker 启动不计入搜索耗时；最佳 KV 重放完全一致。',
        '验收仍严格要求 LSB RMSE < 0.125；未通过时不提交、不推进后续媒体位置。',
        '只覆盖一张图片的首个位置，不能推断整张图片恢复或一般速度优势。', '',
        '验证：111 项测试，109 通过，2 项可选 MLX 测试跳过；新增覆盖完整 768 方向的批处理、',
        '尾轮缩减方向、精确查询预算与失败不提交。', '',
        '## 复现', '', '```bash',
        'kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \\',
        '  kvidence/reconstruction/experiments/image_trust_region/run_spsa_joint.py \\',
        '  --reference-run kvidence/reconstruction/experiments/image_trust_region/results/0001_rounds300_seed17_20261020 \\',
        '  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \\',
        '  --output /tmp/new_full768_seed17 \\',
        '  --geometry --directions 768 --steps 300 --max-queries 45001', '```', '',
        '完整协议、代码哈希、每轮输入及实测 KV、最佳输入和轨迹保留在 results/0001_geometry768_seed*_20261021/。', '']
    (args.directory/'REPORT.md').write_text('\n'.join(lines))
    for r in records:print({k:v for k,v in r.items() if k not in ('checkpoints','direction_counts')})

if __name__=='__main__':main()
