"""Summarize offline prediction and real-NPU warm-start control results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kvrecon.common import dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path(__file__).parent/'surrogate_comparison_20261021')
    args=p.parse_args();directory=args.directory
    offline=json.loads((directory/'heldout.json').read_text())
    rows=[r for run in offline['runs'] for r in run['rows']]
    groups={a:[r for r in rows if r['nonlinear']==a] for a in (0.,.1,1.,10.)}
    records=[]
    for kind,run_name in (('linear','linear100'),('tanh','tanh100'),('tanh10','tanh10_100')):
        folder=directory.parent/'results'/f'0001_{run_name}_seed17_20261021'
        protocol=json.loads((folder/'protocol.json').read_text());report=json.loads((folder/'result.json').read_text())
        trace=json.loads((folder/'optimization.json').read_text())
        records.append(dict(method=kind,path=str(folder),initial_rmse=report['initial_rmse'],best_rmse=report['best_rmse'],
            queries=report['queries'],total_queries=report['total_queries_including_warmup'],seconds=report['optimizer_seconds'],
            total_seconds=report['total_seconds_including_warmup'],passes_query=report['passes_query'],repeat_exact=report['repeat_exact'],
            fit_median_ms=1000*float(np.median([r['surrogate_fit_seconds'] for r in trace])),
            surrogate_median_ms=1000*float(np.median([r['surrogate_seconds'] for r in trace])),
            protocol=protocol))
    for other in records[1:]:
        for key in ('initial_sha256','resume_run','seed','steps','directions','lr','perturbation','hidden','surrogate_ridge','observation','surrogate_source_sha256','source_sha256'):
            if records[0]['protocol'][key]!=other['protocol'][key]:raise ValueError('Unmatched protocols: '+key)
        if records[0]['queries']!=other['queries'] or records[0]['initial_rmse']!=other['initial_rmse']:
            raise ValueError('Unmatched budget or start')
    for r in records:
        if not r['repeat_exact']:raise ValueError('Repeatability failed')
    summary=dict(offline_mean_rmse={str(a):float(np.mean([r['kv_prediction_rmse'] for r in group])) for a,group in groups.items()},online=records)
    dump(directory/'summary.json',summary)
    lines=['# 小型非线性 KV 代理模型', '',
        '结构：64 个固定随机 tanh 神经元，加完整线性支路；仅用实测查询拟合输出。',
        '训练最多使用当前和上一轮的 256 条探针，不需要深度学习框架。',
        '输入为 768 维 pre-adapter 特征，预测 1024 维 KV codes；训练求解最多 256×256 的带正则线性系统。',
        '中心输入和中心 KV 均为实际测量。非线性支路在中心严格为零。', '',
        '## 离线预测', '',
        '用两个 geometry 起点的第 25、100、300 轮；只用当前和上一轮探针训练。',
        '更新候选作为保留集，移除与训练输入重复的行及两个中心候选。',
        '这些是同一搜索轨迹的相关保留数据；不同非线性权重属于探索性比较。', '',
        '| 非线性权重 | 六个检查点平均 KV 预测 RMSE |', '|---|---:|']
    for a,group in groups.items():lines.append(f"| {a} | {np.mean([r['kv_prediction_rmse'] for r in group]):.5f} |")
    wins=sum(b['kv_prediction_rmse']<a['kv_prediction_rmse'] for a,b in zip(groups[0.],groups[.1]))
    lines += ['',f'权重 0.1 的小模型在 {wins}/6 个检查点上优于同数据、同正则的线性模型。', '',
        '## 真实 NPU 搜索', '',
        '从 geometry seed 17 的最佳输入（RMSE 10.61686）重新开始，三种配置均使用相同初始输入和 RNG seed。',
        '各跑 100 轮，每轮 128 个真实探针、22 个真实更新候选，加初始查询，共 15,001 次新搜索查询。',
        '16 个候选由小模型优化，其余沿用 Adam 与中心候选。非线性权重分别为 0、0.1 和 10。',
        '先运行 0.1，再运行线性对照，观察结果后追加离线表现较好的权重 10，属于探索性调参。',
        '代理优化 12 步，输入 RMS 位移限制为当前探针半径的 2 倍。所有候选仍必须由真实 NPU 测量。', '',
        '| 模型 | 最佳实测 RMSE | 新查询数 | 新搜索秒数 | 拟合中位数 ms | 全部代理计算中位数 ms |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['method']} | {r['best_rmse']:.5f} | {r['queries']} | {r['seconds']:.2f} | {r['fit_median_ms']:.2f} | {r['surrogate_median_ms']:.2f} |")
    winner=min(records,key=lambda r:r['best_rmse'])
    lines += ['',f"本次同预算最佳配置为 {winner['method']}，实测 RMSE {winner['best_rmse']:.5f}。"]
    if winner['method']=='linear':
        lines += ['固定 tanh 支路的离线预测更准，但本次真实搜索未优于同结构的线性对照。',
            '因此保留为可选实验，不能把它描述为已验证更好的非线性搜索方法；现有默认搜索保持不变。']
    lines += ['', '上述耗时不含 worker 启动与额外 32 次重复性诊断。三种配置的最佳输入均重复测量一致。',
        '总成本必须计入此前 45,001 次 geometry 搜索，具体数值保留在 summary.json。',
        '只测一个样本的第一个图片位置；顺序执行耗时不能推断一般速度优势。', '',
        '较好的 KV 预测不保证更好的反向搜索。验收仍为真实 LSB RMSE < 0.125；',
        '结果未达到阈值时不提交候选，也不推进后续位置。没有读取原图或真实特征。', '',
        '测试：111 项，109 通过，2 项可选真实 MLX 测试跳过。新增检查包括非线性保留集、',
        '解析梯度的有限差分核对、候选 RMS 限制、实际查询预算和失败不提交。', '',
        '代码：kvrecon/media/image_surrogate.py。入口：run_spsa_joint.py --geometry --surrogate。',
        '指定 --nonlinear 0 运行同结构的线性对照；--resume-run 从已有联合搜索结果继续。',
        '逐查询输入和实测 KV、协议/代码哈希、拟合耗时及预测误差均保留在各运行目录。', '']
    (directory/'REPORT.md').write_text('\n'.join(lines))
    for r in records:print({k:v for k,v in r.items() if k!='protocol'})

if __name__=='__main__':main()
