"""Report bounded adaptive expansion, including rejected trials and real costs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[17,29])
    p.add_argument('--directory',type=Path,default=Path(__file__).parent/'immediate_expansion_20261021')
    args=p.parse_args();records=[]
    for seed in args.seeds:
        folder=args.directory.parent/'results'/f'0001_immediate_expansion_seed{seed}_20261021'
        proto=json.loads((folder/'protocol.json').read_text());report=json.loads((folder/'result.json').read_text())
        trace=json.loads((folder/'optimization.json').read_text())
        if not report['repeat_exact'] or not report['initial_exact_reference']:raise ValueError('Replay/repeatability failed')
        events=[dict(round=r['round'],best_rmse=r['best_rmse'],queries=r['queries'],event=r['expansion_event'])
            for r in trace if r['expansion_event']['action'] not in ('continue','improve','trial')]
        records.append(dict(seed=seed,status=report['status'],before_rmse=report['initial_rmse'],best_rmse=report['best_rmse'],
            improvement_percent=100*(report['initial_rmse']-report['best_rmse'])/report['initial_rmse'],
            added_queries=report['queries'],added_seconds=report['optimizer_seconds'],rounds=report['rounds'],
            total_queries=report['total_queries_including_warmup'],total_seconds=report['total_seconds_including_warmup'],
            passes_query=report['passes_query'],expansion_state=report['expansion_state'],events=events,
            last_improvement=next((dict(round=r['round'],rmse=r['best_rmse'],selected=r['selected']) for r in reversed(trace) if r['improvement']>0),None),
            path=str(folder),protocol_sha256=digest(folder/'protocol.json'),trace_sha256=digest(folder/'optimization.json')))
    args.directory.mkdir(parents=True,exist_ok=True);dump(args.directory/'summary.json',dict(records=records))
    lines=['# 单轮无改善即放大搜索尺度', '',
        '从多尺度五轮实验的两个最佳点继续；仍为同一样本、第 5 层、第一个图片位置。',
        '所有候选均使用实际手机 vision adapter 和 NPU，前缀不推进；只有真实 KV 误差改善才更新最佳点。', '',
        '## 规则', '',
        '- 每轮使用完整 768 方向，三个探针尺度；完整测量正负探针和宽范围步长候选。',
        '- 一轮没有实测最佳值改善，下一轮就将主要探针幅度及岭回归位移半径扩大 2 倍。',
        '- 扩大后只试一轮；有实测改善就保留新尺度。',
        '- 试验没有改善则拒绝该档，保留最佳输入和最近有效半径，再试下一档。',
        '- 主要半径最多为起始值的 4 倍；两档均失败或最大档下一轮没有改善则停止。',
        '- 最小探针幅度始终固定为起始半径的 1/4，避免扩大时丢失原来的精细探索。',
        '- 每轮最大探针幅度为当前主要半径的 4 倍，更新倍率仍为 1/64 到 4。', '',
        '学习率沿用前次尾值，RNG 与 Adam 状态继续恢复。任何尺度的实测改善都会接受试验档，',
        '因此接受新档本身不证明改善由大尺度造成；逐轮记录保留实际获胜候选及来源。', '',
        '| Seed | 续跑前 RMSE | 最佳 RMSE | 降低 | 新增轮数 | 新增查询 | 新搜索秒数 | 状态 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in records:lines.append(f"| {r['seed']} | {r['before_rmse']:.6f} | {r['best_rmse']:.6f} | {r['improvement_percent']:.2f}% | {r['rounds']} | {r['added_queries']} | {r['added_seconds']:.2f} | {r['status']} |")
    lines += ['', '## 放大与回退事件', '']
    for r in records:
        lines += [f"### Seed {r['seed']}", '']
        for event in r['events']:lines.append(f"- 第 {event['round']} 轮，RMSE {event['best_rmse']:.6f}：{event['event']}")
        if not r['events']:lines.append('本次尚未触发扩大。')
        lines += ['',f"最后改善：{r['last_improvement']}",'',f"最终状态：{r['expansion_state']}",'']
    lines += ['## 验证与限制', '',
        '113 项已有测试加 3 项策略测试：116 项，114 通过、2 项可选 MLX 测试跳过。',
        '策略测试覆盖有改善重置平台、放大接受、失败回退、半径上限、状态恢复及拒绝无改善档。',
        '集成测试核对实际批次限制、精确查询计数、小尺度锚点和失败不提交。',
        '每次初始输入重放均与上一最佳 KV 一致；停止后的最佳输入各重复 32 次 KV 完全一致。',
        '新增耗时不含 worker 启动和重复性诊断；累计耗时和查询包含此前全部搜索。',
        'seed 29 的旧 20 轮等待策略按用户新指令中断；其完整 9 轮与第 10 轮已执行的 4608 个探针查询均计入成本。',
        '该中断阶段耗时采用包含启动与恢复间隔的保守上界，因此 seed 29 累计耗时也是上界；本阶段耗时为实际计时。',
        '从最后完整检查点恢复。中断后完整保存的两个探针块均未优于已有最佳值，未使用不完整的第三个块。',
        '每轮成本为 4729 次候选测量，加阶段初始复核；这是带额外预算的续跑，不是独立同预算胜率比较。',
        '真实验收仍严格要求 LSB RMSE < 0.125；未通过不提交、不推进后续位置。',
        '没有读取原图或真实特征，没有恢复整张图片。单轮放大失败不是平台期证明，也不证明全局最优。',
        'round_budget 只表示单次保护预算耗尽，不代表尺度探索结束。', '',
        '代码：kvrecon/media/image_expansion.py；入口 run_full_multiscale.py --adaptive。',
        '每轮保存 expansion_state.json 和 optimizer_state.npz，可从检查点继续。',
        '实际运行参数：--rounds 100 --adaptive --plateau-rounds 1 --expansion-trial-rounds 1 --max-radius-multiplier 4。', '']
    (args.directory/'REPORT.md').write_text('\n'.join(lines))
    for r in records:print(r)

if __name__=='__main__':main()
