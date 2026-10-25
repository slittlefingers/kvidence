"""Summarize measured real-image results without reading private evaluation data."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output
    protocol = json.loads((out/'protocol.json').read_text())
    runs = json.loads((out/'runs.json').read_text()) if (out/'runs.json').exists() else []
    final = json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
    passed = sum(r['passes_query'] and r['passes_commit'] for r in runs)
    lines = [
        '# 真实图片：手机原生 SPSA + Adam，原始 KV L2 < 5', '',
        '样本：`image_chat_100/0001`；目标：第 5 层，260 个图片位置。', '',
        '用户在合成控制完成后明确将图片阈值从原始 K+V L2 < 4 放宽为 < 5。'
        '这里是未经维度归一化、未除以量化步长的原始数值距离，不是 5 LSB。'
        '原来的合成实验记录及文字恢复阈值不变。', '',
        '实际前向使用手机模型的 CPU vision adapter 和 NPU decoder。电脑只做'
        '随机扰动、距离计算与 Adam 更新。恢复不读取原图、原文或真实 encoder 输出，'
        '不把目标 KV 写入前缀；后续位置使用已经恢复并提交的表示。', '',
        f'已完成搜索的位置数：{len(runs)}；查询和提交同时通过的位置数：{passed}/260。', '',
        '| 绝对位置 | 初始 L2 | 最佳查询 L2 | 实际提交 L2 | 轮数 | 逻辑候选数 | 搜索时间 | 通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for r in runs:
        good = r['passes_query'] and r['passes_commit']
        lines.append(f"| {r['position']} | {r['initial_l2']:.5f} | {r['best_l2']:.5f} | "
            f"{r['commit_l2']:.5f} | {r['steps']} | {r['queries']:,} | {r['seconds']:.1f} s | {'是' if good else '否'} |")
    lines += ['', f"已完成位置的搜索时间合计：{sum(r['seconds'] for r in runs):.1f} s。"
        '时间包括候选传输和手机前向，不包括模型初始化与最终提交检查。', '', '## 结果边界', '']
    if final is None:
        lines += ['实验尚未生成最终结果；本表仅汇总已完成的位置，不代表整张图片已恢复。']
    elif final['status'] == 'interrupted_for_user_prioritized_content_diagnostic':
        lines += ['用户再次明确优先检查图片内容后，本次逐位置搜索被主动中断，'
            '释放设备用于已有完整图片候选的语义诊断。已通过的位置保留，'
            '正在搜索的位置没有完成，不能记为搜索失败，也不能记为通过。', '',
            '本次 SPSA 尚无完整图片表示；整图语义诊断使用的是此前的完整候选，'
            '来源和结果独立记录在 `../full_proposal_phone_content/`，不能冒充本次 SPSA 结果。']
    elif not final['whole_image_representation_available']:
        lines += ['在未达标的位置停止，未用失败候选继续构造图片前缀。'
            '没有完整的已验收图片表示，因此没有生成恢复图片的内容描述，也没有进行原图对照。', '',
            '达到 1,000 轮上限只能说明这次搜索未找到达标候选，不能证明不存在解。'
            '本次结果也不能单独区分优化器收敛不足、前缀近似误差与完整媒体路径差异。']
    else:
        lines += ['全部图片位置通过距离阈值。这仍不证明恢复表示唯一或语义正确。']
    caption_path = out/'image_description.json'
    if caption_path.exists():
        caption = json.loads(caption_path.read_text())
        lines += ['', '## 恢复表示生成的描述（尚需独立对照原图）', '', caption['text'], '']
    lines += ['', '## 可复查记录', '',
        '- `protocol.json`：阈值变更、配置、模型和源码哈希。',
        '- `runs.json` 与各位置的 `result.json` / `trace.jsonl`：实际收敛轨迹和提交验证。',
        '- `accepted_features.npy`：已通过位置的表示；不足 260 行时不能当作完整图片。',
        '- `device/`：设备身份、模型哈希与实际调用日志。', '',
        f"初始化 seed：{protocol['config']['seed']}；选择依据是之前的合成控制表现，未按原图筛选。", '']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
