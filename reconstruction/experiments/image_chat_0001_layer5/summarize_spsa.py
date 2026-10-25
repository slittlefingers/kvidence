"""Summarize completed runs without opening any private media/text labels."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    args=parser.parse_args();p=args.run
    result=json.loads((p/'result.json').read_text())
    protocol=json.loads((p/'protocol.json').read_text())
    lines=['# 手机原生 SPSA + Adam 结果','',
        '仅调用手机 adapter 与 NPU 前向计算；不加载电脑模型，不读取原图或原文。','',
        '| 目标 / 初始化 | 初始 L2 | 最佳查询 L2 | 单独提交 L2 | 轮数 | 逻辑候选数 | 用时 | 达标 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    derived=[]
    for r in result['runs']:
        passing=r['passes_query'] and r['passes_commit']
        lines.append(f"| {r['kind']} | {r['initial_l2']:.4f} | {r['best_l2']:.4f} | {r['commit_l2']:.4f} | {r['steps']} | {r['queries']:,} | {r['seconds']:.1f} s | {'是' if passing else '否'} |")
        trace=[json.loads(x) for x in (p/r['kind']/'trace.jsonl').read_text().splitlines()]
        within=[x for x in trace if x['queries']<=23318]
        derived.append(dict(kind=r['kind'],best_l2_within_23318_queries=within[-1]['best_l2'],
            actual_queries_at_that_checkpoint=within[-1]['queries'],
            mean_zero_slope_fraction=statistics.mean(x['zero_slope_fraction'] for x in trace[1:]) if len(trace)>1 else None))
    lines+=['','## 范围与结论','',
        f"已知存在解的合成目标：{result['synthetic_passes']}/{result['synthetic_runs']} 个初始化同时通过查询与提交的原始 K+V L2 < 4。",
        '两个初始化共用一个目标，不能解读为两张图片或整体恢复率。','']
    if result['real_first_position_attempted']:
        r=next(x for x in result['runs'] if x['kind']=='real_image_position_5')
        lines.append('真实 `image_chat_100/0001` 的首个图片位置'+('通过阈值。' if r['passes_query'] and r['passes_commit'] else '没有通过阈值。'))
    else:
        lines.append('按预设规则，合成控制未全部通过，因此没有启动真实图片首位置测试。')
    lines+=['','本次没有运行整张图片恢复或后续文字恢复，也没有声称恢复原图像素。',
        '达到迭代上限不证明没有解；达到距离阈值也不证明 latent 唯一。','',
        '## 配置与开销','',
        f"每轮 {protocol['config']['directions']} 个随机方向，每个方向正负两次探测；之后用 Adam 更新并额外测一个更新点。",
        f"初始学习率 {protocol['config']['lr']}，初始扰动 {protocol['config']['perturbation']}，最多 {protocol['config']['steps']} 轮。",
        '每轮 129 个逻辑候选，分两次固定 128-slot NPU 前向。候选数不是 NPU 调用次数。',
        '优化损失使用均方差缩放；验收使用逐位置未经缩放的原始 K+V L2 < 4。',
        '完整方法与数据隔离约定见上级 `SPSA_PROTOCOL.md`。','',
        '## 较早预算点','',
        '为便于观察查询效率，列出不超过 23,318 次逻辑候选时的检查点；这不是额外的停止条件。','',
        '| 运行 | 检查点候选数 | 当时最佳 L2 | 正负损失相同方向的平均比例 |',
        '|---|---:|---:|---:|']
    for d in derived:
        lines.append(f"| {d['kind']} | {d['actual_queries_at_that_checkpoint']:,} | {d['best_l2_within_23318_queries']:.4f} | {d['mean_zero_slope_fraction']:.4f} |")
    lines+=['','证据：`protocol.json`、`runs.json`、各运行的 `trace.jsonl` / `result.json`，以及 `device/` 中的设备身份和 RPC 日志。','']
    (p/'REPORT.md').write_text('\n'.join(lines))
    (p/'analysis.json').write_text(json.dumps(derived,indent=2)+'\n')
    print(p/'REPORT.md')

if __name__=='__main__':main()
