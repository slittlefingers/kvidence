"""Summarize complete-basis continuation and its explicit stagnation criterion."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path(__file__).parent/'plateau_20261021')
    args=p.parse_args();records=[]
    for seed in (17,29):
        folder=args.directory.parent/'results'/f'0001_geometry768_plateau_seed{seed}_20261021'
        protocol=json.loads((folder/'protocol.json').read_text())
        result=json.loads((folder/'result.json').read_text())
        trace=json.loads((folder/'optimization.json').read_text())
        if result['status'] not in ('plateau','threshold'):raise ValueError('Run has not reached a stopping criterion')
        if not result['repeat_exact'] or not result['initial_exact_reference']:raise ValueError('Replay check failed')
        if result['plateau_detected']:
            window=trace[-protocol['plateau_rounds']:]
            if len(window)!=protocol['plateau_rounds'] or any(r['directions']!=768 or r['improvement']!=0 for r in window):
                raise ValueError('Plateau window does not match actual trace')
        previous=Path(protocol['resume_run']);old=json.loads((previous/'result.json').read_text())
        old_trace=json.loads((previous/'optimization.json').read_text())
        improvements=[r for r in trace if r['improvement']>0]
        if improvements:
            last=improvements[-1];last_round=protocol['adam_offset']+last['round']
            last_queries=old['total_queries_including_warmup']+last['queries']
        else:
            improving_old=[r for i,r in enumerate(old_trace) if r['best_rmse']<(old_trace[i-1]['best_rmse'] if i else old['initial_rmse'])]
            last=improving_old[-1] if improving_old else None
            last_round=last['round'] if last else 0;last_queries=last['queries'] if last else 1
        records.append(dict(seed=seed,status=result['status'],before_rmse=result['initial_rmse'],best_rmse=result['best_rmse'],
            added_rounds=result['rounds'],added_queries=result['queries'],added_seconds=result['optimizer_seconds'],
            restore_seconds=result['state_restore_seconds'],total_queries=result['total_queries_including_warmup'],
            total_seconds=result['total_seconds_including_warmup'],total_rounds=protocol['adam_offset']+result['rounds'],
            last_improvement_round=last_round,last_improvement_total_queries=last_queries,
            plateau_rounds=protocol['plateau_rounds'],stale_rounds=result['stale_rounds'],repeat_exact=result['repeat_exact'],
            fixed_lr=protocol['fixed_tail_lr'],fixed_perturbation=protocol['fixed_tail_perturbation'],passes_query=result['passes_query'],
            path=str(folder),protocol_sha256=digest(folder/'protocol.json'),trace_sha256=digest(folder/'optimization.json')))
    args.directory.mkdir(parents=True,exist_ok=True)
    dump(args.directory/'summary.json',dict(criterion='20 consecutive full 768-direction rounds without ANY measured best RMSE improvement; continuation-local',
        checkpoint_source_sha256=digest(ROOT/'kvrecon/media/image_checkpoint.py'),records=records))
    lines=['# 768 方向搜索的平台期实测', '',
        '沿用 image_chat_100/0001、第 5 层、首个图片位置，分别从两次 45,001 查询运行的最佳输入继续。',
        '每轮 768 个正交方向，1,536 个正负探针加 22 个更新候选，全部实际 NPU 测量。',
        '平台判据：续跑阶段连续 20 个完整轮次没有任何实测最佳 RMSE 改善；有任何改善便重新计数。', '',
        '| Seed | 续跑前 RMSE | 最佳 RMSE | 新增轮数 | 累计轮数 | 最后改善轮次（累计） | 累计查询数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['seed']} | {r['before_rmse']:.6f} | {r['best_rmse']:.6f} | {r['added_rounds']} | {r['total_rounds']} | {r['last_improvement_round']} | {r['total_queries']} |")
    lines += ['', '## 续跑设置与验证', '',
        '旧运行没有 optimizer checkpoint，因此从其公开测量探针逐轮重建 Adam 一二阶矩和随机数发生器状态。',
        '重建时核对每轮正负输入差与生成方向一致；Adam 偏差修正继续使用累计步数。',
        '新运行每轮保存 RNG/Adam checkpoint，原始运行和数据保留。没有换随机起点或重置优化器。',
        '原有学习率日程已到尾端，本阶段固定在上一运行末轮的学习率和扰动幅度。',
        '此为明确的尾值延长阶段，不是继续按原有限预算日程衰减。', '',
        '| Seed | 固定学习率 | 固定扰动 RMS | 新搜索秒数 | 状态重建秒数 | 累计搜索秒数 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['seed']} | {r['fixed_lr']:.8f} | {r['fixed_perturbation']:.8f} | {r['added_seconds']:.2f} | {r['restore_seconds']:.2f} | {r['total_seconds']:.2f} |")
    lines += ['', '累计轮数包含此前 28 个完整 768 方向轮次和 1 个 677 方向尾轮，续跑所有轮次均为完整 768 方向。',
        '累计成本含原运行、续跑和状态重建；初始复核查询计入。worker 启动与额外 32 次重复测量不计入搜索。',
        '停止后最佳输入各重复测量 32 次，KV 完全一致；初始 KV 也与上一运行最佳 KV 完全相同。',
        '未读取原图、原始特征或隐藏文字。真实后端验收仍严格要求 LSB RMSE < 0.125；未通过不提交。', '',
        '本报告只表明这组参数、首个图片位置和对应初始化下的经验平台，不能证明全局最优或不可再降低。',
        '也不代表整张图片特征已恢复。另一个尺度、目标或优化器仍可能继续改善。', '',
        '测试：112 项，110 通过，2 项可选 MLX 测试跳过；覆盖状态重建、RNG/Adam 恢复、',
        '平台自动停止、累计成本、批量查询与失败不提交。', '',
        '复现参数：run_spsa_joint.py --geometry --directions 768 --resume-run PREVIOUS_RUN --continue-tail --plateau-rounds 20 --steps 300。',
        '需要已有的 --reference-run、--observation，并为 --output 指定新目录。300 轮是单次保护上限，',
        '若到上限仍有改善则只记 iteration_limit，不能称为已达平台。', '']
    (args.directory/'REPORT.md').write_text('\n'.join(lines))
    for r in records:print(r)

if __name__=='__main__':main()
