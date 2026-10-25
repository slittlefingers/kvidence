"""Compare saved baseline and geometry runs at matched candidate budgets."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Output must be new')
    runs=[]
    for seed in (17,29):
        pair=[]
        for kind,date in (('joint','20261020'),('geometry','20261021')):
            folder=args.results/f'0001_{kind}300_seed{seed}_{date}'
            protocol=json.loads((folder/'protocol.json').read_text())
            report=json.loads((folder/'result.json').read_text())
            if not report['initial_exact_reference'] or not report['repeat_exact']:
                raise ValueError('Initial replay or repeatability verification failed')
            trace=json.loads((folder/'optimization.json').read_text())
            pair.append((protocol,report))
            runs.append(dict(seed=seed,method=kind,path=str(folder),
                initial_rmse=report['initial_rmse'],best_rmse=report['best_rmse'],
                queries=report['queries'],seconds=report['optimizer_seconds'],
                passes_query=report['passes_query'],repeat_exact=report['repeat_exact'],
                initial_exact_reference=report['initial_exact_reference'],
                checkpoints=[dict(round=r['round'],queries=r['queries'],rmse=r['best_rmse'],seconds=r['elapsed_seconds'])
                    for r in trace if r['round'] in (25,50,100,150,200,300)]))
        a,b=pair
        for key in ('initial_sha256','seed','steps','directions','lr','perturbation','epsilon'):
            if a[0][key]!=b[0][key]:raise ValueError('Protocol mismatch: '+key)
        if a[0]['observation']!=b[0]['observation']:raise ValueError('Different observations')
        if a[1]['queries']!=b[1]['queries']:raise ValueError('Different search budgets')
        if a[1]['initial_rmse']!=b[1]['initial_rmse']:raise ValueError('Different initial measurements')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    dump(args.output,dict(note='Single sample, first image position, two initializations. Same candidate budget; sequential device timings.',runs=runs))
    lines=['# 图片特征搜索：正交探针＋真实中心残差', '',
        '对象为 image_chat_100/0001，第 5 层，第一个图片位置（position 5）。',
        '全部候选使用手机 CPU vision adapter 和真实 NPU 测量；没有读取原图、真实特征或隐藏文本。', '',
        '| Seed | 方法 | 初始 RMSE | 最佳 RMSE | 候选数 | 秒 | 通过 |',
        '|---|---|---:|---:|---:|---:|---|']
    for row in runs:
        lines.append(f"| {row['seed']} | {row['method']} | {row['initial_rmse']:.5f} | {row['best_rmse']:.5f} | {row['queries']} | {row['seconds']:.2f} | {row['passes_query']} |")
    lines += ['', '## 变化与结果', '',
        '新版是 opt-in 的 `run_spsa_joint.py --geometry`。使用 RMS 为 1 的正交探针；',
        '使用真实中心 KV 而非正负探针的平均 KV 构造残差；岭回归步长先限制到当前探针 RMS 半径。',
        '每轮仍为 128 个探针和 22 个更新候选，初始输入、学习率日程及验收阈值保持一致。',
        '三项变化一起测试，不能单独归因；未启用参数时继续使用旧算法。', '']
    for seed in (17,29):
        old,new=[r for r in runs if r['seed']==seed]
        change=100*(old['best_rmse']-new['best_rmse'])/old['best_rmse']
        lines.append(f"- Seed {seed}：同预算最佳 RMSE 降低 {change:.2f}%。")
    lines += ['', '每次初始 KV 均与对应旧实验完全相同；最佳输入的 32 次重复测量均完全一致。',
        '32 次诊断查询与启动耗时不计入搜索成本，口径与旧实验一致。耗时来自顺序执行，不能视为一般速度优势。', '',
        '验收仍严格要求 LSB RMSE < 0.125。本次结果未通过，因此没有提交候选或推进后续位置。',
        '只验证一个样本的首个图片位置、两个初始化；不能称为整张图片特征恢复完成、原始特征恢复或像素重建。', '',
        '## 诊断与验证', '',
        '`heldout.json` 使用旧运行正负探针拟合局部模型，单独测得的更新候选作为预测检查。',
        '这些是同一搜索轨迹上的相关数据；结果不支持仅凭中心残差修正就保证改善。',
        '在检查的轮次，正负探针中约 85%–95% 的 KV 分量改变，因此没有证据称完全量化平台是唯一原因。',
        '新版的改进以真实 NPU 测量为准；没有以代理预测接受候选。', '',
        '常规测试：108 项，106 通过、2 项可选真实 MLX 测试跳过。',
        '另补充新版 runner 的同预算与失败不提交检查，相关 2 项测试通过。', '',
        '数值模块：`kvrecon/media/image_geometry.py`。重现命令与数学定义见同级 `../README.md`。',
        '协议、代码哈希、每次候选输入与实测 KV、优化轨迹及最佳输入均保留在各运行目录。',
        '运行 `analyze_geometry_comparison.py --output NEW_JSON` 可重建比较表。', '']
    args.output.with_suffix('.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
