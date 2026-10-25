"""Summarize trained-MLP latency, held-out prediction and real-NPU proposals."""
import argparse,json,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path(__file__).parent/'mlp_comparison_20261021')
    args=p.parse_args();directory=args.directory
    offline=json.loads((directory/'offline.json').read_text());records=[]
    for seed in (17,29):
        folder=directory.parent/'results'/f'0001_mlp5_seed{seed}_20261021'
        proto=json.loads((folder/'protocol.json').read_text());report=json.loads((folder/'result.json').read_text())
        trace=json.loads((folder/'optimization.json').read_text())
        if not report['repeat_exact'] or not report['initial_exact_reference']:raise ValueError('Replay failed')
        rows=[r['mlp'] for r in trace]
        records.append(dict(seed=seed,before_rmse=report['initial_rmse'],after_rmse=report['best_rmse'],
            added_queries=report['queries'],added_seconds=report['optimizer_seconds'],rounds=report['rounds'],
            initial_fit_ms=1000*rows[0]['mlp_fit_seconds'],warm_fit_median_ms=1000*statistics.median(r['mlp_fit_seconds'] for r in rows[1:]),
            mlp_proposal_median_ms=1000*statistics.median(r['mlp_proposal_seconds'] for r in rows),
            linear_proposal_median_ms=1000*statistics.median(r['linear_proposal_seconds'] for r in rows),
            linear_fit_median_ms=1000*statistics.median(r['linear_fit_seconds'] for r in rows),
            mlp_extra_improvement_rounds=[r['round'] for r in trace if r['mlp']['mlp_proposal_best_rmse']<min(r['mlp']['standard_best_rmse'],r['mlp']['linear_proposal_best_rmse'])],
            linear_extra_improvement_rounds=[r['round'] for r in trace if r['mlp']['linear_proposal_best_rmse']<min(r['mlp']['standard_best_rmse'],r['mlp']['mlp_proposal_best_rmse'])],
            selected=[dict(round=r['round'],rmse=r['best_rmse'],selected=r['selected'],stats=r['mlp']) for r in trace],
            passes_query=report['passes_query'],torch_setup_seconds=proto['mlp_import_seconds'],
            trace_sha256=digest(folder/'optimization.json'),path=str(folder)))
    groups={k:[r for r in offline['records'] if r['model']==k] for k in ('linear','fixed_tanh','mlp100','mlp300')}
    summary=dict(offline={k:dict(mean_prediction_rmse=statistics.mean(r['prediction_rmse'] for r in rows),
        median_fit_ms=1000*statistics.median(r['fit_seconds'] for r in rows),
        median_predict_ms=1000*statistics.median(r['predict_seconds'] for r in rows)) for k,rows in groups.items()},online=records)
    dump(directory/'summary.json',summary)
    lines=['# 小型可训练 MLP：速度、预测与 NPU 搜索', '',
        'MindEye 官方 BrainNetwork 使用可训练残差 MLP，默认隐藏宽度 4096、4 个残差块，',
        '用于 fMRI 到 CLIP 表示映射，另有 projector 等模块。官方源码：',
        '[MedARC-AI/models.py](https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/models.py)。',
        '本实验借鉴可训练非线性表示映射，未复制其完整模型、损失或训练流程，也不对比其训练吞吐。', '',
        '## 当前小模型', '',
        '非线性支路为 768 → 128 → 64 → 1024，SiLU 激活，172,224 个可训练参数。',
        '输入为归一化的局部特征位移，输出为 KV 变化的非线性修正；另保留冻结的局部线性岭回归支路。',
        '因此总表示还包括线性支路，172,224 只统计 MLP 参数，不包含岭回归系数。',
        '两条支路都锚定在实测中心 KV；不训练实际 Gemma/adapter，不读取原图或真实特征。',
        'CPU 单线程 PyTorch AdamW，学习率 0.003、权重衰减 1e-4，batch 64；首次 100 步，后续热更新 20 步。', '',
        '## 离线保留集预测与耗时', '',
        '同一批旧 geometry 查询：两个起点、第 25/100/300 轮，用当前和上一轮共 256 条探针训练。',
        '更新候选作为保留集；排除中心和与训练重复的输入，不用测试误差早停或选择权重。',
        '100/300 步属于探索性配置比较，不能把这些相关轨迹数据当独立泛化测试。', '',
        '| 模型 | 平均 KV 预测 RMSE | 拟合中位数 ms | 预测约 20 候选中位数 ms |',
        '|---|---:|---:|---:|']
    for k,row in summary['offline'].items():lines.append(f"| {k} | {row['mean_prediction_rmse']:.5f} | {row['median_fit_ms']:.2f} | {row['median_predict_ms']:.3f} |")
    warm=[r['warm_fit_seconds'] for r in groups['mlp100']]
    lines += ['',f"100 步模型的首次拟合（含 optimizer 初始化）为 {groups['mlp100'][0]['fit_seconds']*1000:.2f} ms；",
        f"后续相同训练集 20 步热更新中位数为 {statistics.median(warm)*1000:.2f} ms。该热更新时间仅是速度检查，未计入前表预测结果。",
        f"框架导入另计 {offline['torch_import_seconds']:.3f} s。没有使用 GPU/MPS。", '',
        '## 真实 NPU 五轮试验', '',
        '从最新最佳点继续，保留完整 768 方向、三个探针尺度及原有 121 个更新候选。',
        '主探针 RMS 固定为 0.0871384031，本阶段单独检验模型，不执行尺度扩张策略。',
        '每尺度从实测探针随机抽取 42 对方向，共 252 条，线性和 MLP 使用完全相同训练数据。',
        '从实测原有候选选取相同 16 个起点，各模型独立优化 12 步，位移 RMS 上限 8h。',
        '新增的 16+16 个候选全部由实际 NPU 测量，每轮共 4761 次，五轮加初始重放为 23,806 次。',
        '训练只是辅助候选生成，实际搜索没有缩减为 64 个方向。', '',
        '| Seed | 运行前 RMSE | 运行后 RMSE | MLP 初次拟合 ms | 热更新中位数 ms | MLP 独有改善轮次 | 线性独有改善轮次 |',
        '|---|---:|---:|---:|---:|---|---|']
    for r in records:lines.append(f"| {r['seed']} | {r['before_rmse']:.6f} | {r['after_rmse']:.6f} | {r['initial_fit_ms']:.2f} | {r['warm_fit_median_ms']:.2f} | {r['mlp_extra_improvement_rounds']} | {r['linear_extra_improvement_rounds']} |")
    lines += ['', '独有改善指该模型候选的实测误差同时低于本轮全部标准候选（含探针与原最佳）和另一个模型候选。',
        '两类额外候选相同预算、相同训练集和起点，且保留标准搜索为逐轮对照。',
        '后续轮次共享实测最佳轨迹，因此不是两个独立优化器的同预算长程胜率比较。', '',
        '## 限制与验证', '',
        '离线预测更准不能替代真实候选改善。零个独有改善轮次不代表所有 MLP 配置均无效。',
        '仅一张图片的首个位置、两个初始化、各五轮，不能称整张图片特征恢复。',
        '真实验收仍严格要求 LSB RMSE < 0.125；未通过不提交、不推进后续位置。',
        '初始 KV 与前次最佳值一致，最终最佳输入各重复 32 次 KV 完全一致。',
        '五轮预算用完不是平台证明。没有复制 MindEye 的图像解码或 diffusion prior。',
        '测试 118 项：116 通过、2 项可选 MLX 测试跳过。包括可训练非线性泛化、隐藏层更新、',
        '输入梯度有限差分核对、候选半径、实测查询计数与失败不提交。', '',
        '实现：kvrecon/media/image_mlp.py；离线 benchmark_mlp.py；NPU 入口 run_full_multiscale.py --mlp-test。',
        '实际运行参数：--rounds 5 --mlp-test --probe-radius 0.08713840313369006。',
        '网络每轮权重、训练配置、实测候选及 KV、训练与优化耗时、候选来源和代码哈希均保留在运行目录。', '']
    (directory/'REPORT.md').write_text('\n'.join(lines))
    for r in records:print(r)

if __name__=='__main__':main()
