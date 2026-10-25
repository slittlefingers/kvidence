"""Report full-basis multiscale / broad line-search trials at two plateau points."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path(__file__).parent/'full_multiscale_20261021')
    args=p.parse_args();records=[]
    for seed in (17,29):
        folder=args.directory.parent/'results'/f'0001_fullmultiscale5_seed{seed}_20261021'
        protocol=json.loads((folder/'protocol.json').read_text());report=json.loads((folder/'result.json').read_text())
        trace=json.loads((folder/'optimization.json').read_text())
        if not report['initial_exact_reference'] or not report['repeat_exact']:raise ValueError('Replay/repeatability failed')
        record=dict(seed=seed,path=str(folder),before_rmse=report['initial_rmse'],after_rmse=report['best_rmse'],
            relative_reduction_percent=100*(report['initial_rmse']-report['best_rmse'])/report['initial_rmse'],
            added_queries=report['queries'],added_seconds=report['optimizer_seconds'],
            total_queries=report['total_queries_including_warmup'],total_seconds=report['total_seconds_including_warmup'],
            rounds=report['rounds'],status=report['status'],passes_query=report['passes_query'],
            first_round_probe_comparison=trace[0]['scale_probes'],first_round_step_comparison=trace[0]['scale_updates'],
            improvements=[dict(round=r['round'],best_rmse=r['best_rmse'],improvement=r['improvement'],selected=r['selected']) for r in trace if r['improvement']>0],
            source_sha256=protocol['source_sha256'],trace_sha256=digest(folder/'optimization.json'))
        records.append(record)
    args.directory.mkdir(parents=True,exist_ok=True);dump(args.directory/'summary.json',dict(records=records))
    lines=['# 平台点的多尺度探针与扩展步长实验', '',
        '材料为 image_chat_100/0001、第 5 层、首个图片位置；从两个已满足连续 20 轮不改善的平台点开始。',
        '保持 768 个完整正交方向。每轮在同一中心、同一方向基上测量 h/4、h、4h 三种探针幅度。',
        'h=0.0871384031 RMS，三种幅度约为 0.02178、0.08714、0.34855。',
        '每个尺度求 4 种阻尼的岭回归方向，分别测量 1/64、1/32、1/16、1/8、1/4、1/2、1、2、4 倍步长。',
        '同时保留每尺度 4 个 Adam 候选及中心重放，共 121 个更新候选。',
        '每轮 3×1536+121=4729 个实际查询候选，5 轮加初始查询共 23,646 个。', '',
        '| Seed | 原平台 RMSE | 新最佳 RMSE | 降低 | 新查询数 | 新搜索秒数 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['seed']} | {r['before_rmse']:.6f} | {r['after_rmse']:.6f} | {r['relative_reduction_percent']:.2f}% | {r['added_queries']} | {r['added_seconds']:.2f} |")
    lines += ['', '## 实测改善来自哪里', '']
    for r in records:
        lines += [f"### Seed {r['seed']}", '']
        if not r['improvements']:lines.append('本次预算内没有刷新原平台最佳值。')
        for item in r['improvements']:
            a=item['selected'];lines.append(f"- 第 {item['round']} 轮：RMSE {item['best_rmse']:.6f}，来源 {a}。")
        lines += ['', '第一轮的更新候选均使用同一个原平台中心，因而可以直接查看尺度和步长对候选误差的影响：', '',
            '| 探针倍率 | 原有步长子集最佳 RMSE | 扩展步长最佳 RMSE |', '|---|---:|---:|']
        for a in r['first_round_step_comparison']:lines.append(f"| {a['scale']} | {a['best_original_step_subset_rmse']:.6f} | {a['best_broad_step_rmse']:.6f} |")
        lines.append('')
    lines += ['## 口径与限制', '',
        '原有步长子集为岭回归倍率 0.25、0.5、1、2 和该尺度的四个 Adam 候选。',
        '所有探针和更新候选均由真实 NPU 测量，最佳值包括探针；没有按预测值接受候选。',
        '初始 KV 与此前最佳 KV 完全一致，停止后最佳输入重复测量 32 次 KV 完全一致。',
        '恢复了此前 RNG 和 Adam 状态，未随机重启；各尺度使用相同的原 Adam 状态计算候选。',
        '本轮实测胜出候选所属尺度的梯度用来更新 Adam 状态；未改善时使用原尺度梯度。',
        '本实验同时扩大探针和更新尺度，后续轨迹不同，不能仅据最终误差把作用归因到单一参数。',
        '三尺度开销高于单尺度；这不是与单尺度同预算的独立速度优越性试验。',
        '本次仅各 5 轮，状态 round_budget 表示预算用完，不代表发现新平台或全局最优。',
        '真实验收仍为 LSB RMSE < 0.125；未通过不提交，也没有恢复整张图片或后续文本。',
        '未读取原图、真实特征或隐藏文字；累计成本包含此前搜索，详见 summary.json。',
        '新增搜索耗时不含 worker 启动和额外 32 次诊断；状态读取成本计入累计耗时。', '',
        '测试：113 项，111 通过，2 项可选 MLX 测试跳过；新增测试覆盖三尺度的真实查询预算、',
        '128 条物理批次限制、步长范围、保留最佳值和失败不提交。', '',
        '## 复现', '', '```bash',
        'kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \\',
        '  kvidence/reconstruction/experiments/image_trust_region/run_full_multiscale.py \\',
        '  --warm-run kvidence/reconstruction/experiments/image_trust_region/results/0001_geometry768_plateau_seed17_20261021 \\',
        '  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \\',
        '  --output /tmp/new_full_multiscale17 --rounds 5', '```', '',
        '各尺度探针、更新候选及其实际位移 RMS、实测 KV、来源、逐轮轨迹和协议/代码哈希均保留在对应 results/ 目录。', '']
    (args.directory/'REPORT.md').write_text('\n'.join(lines))
    for r in records:print(r)

if __name__=='__main__':main()
