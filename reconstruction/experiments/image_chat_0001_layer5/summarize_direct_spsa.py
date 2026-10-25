"""Compare sequential post-adapter pilot with preserved pre-adapter baseline."""
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'direct_spsa_l2_5'
    result=json.loads((out/'result.json').read_text())
    old=json.loads((HERE/'real_spsa_l2_5/runs.json').read_text())
    by_position={r['position']:r for r in old}
    rows=[]
    for current in result['runs']:
        before=by_position[current['position']]
        rows.append(dict(position=current['position'],baseline=before,direct=current,
            time_ratio_baseline_over_direct=before['seconds']/current['seconds'],
            candidate_ratio_baseline_over_direct=before['queries']/current['queries'],
            identical_prefix=current['position']==5,
            both_pass=bool(current['passes_query'] and current['passes_commit'] and
                           before['passes_query'] and before['passes_commit'])))
    paired_pass=all(r['both_pass'] for r in rows)
    old_seconds=sum(r['baseline']['seconds'] for r in rows)
    new_seconds=sum(r['direct']['seconds'] for r in rows)
    old_queries=sum(r['baseline']['queries'] for r in rows)
    new_queries=sum(r['direct']['queries'] for r in rows)
    comparison=dict(rows=rows,all_attempted_pass_in_both=paired_pass,
        baseline_search_seconds=old_seconds,direct_search_seconds=new_seconds,
        baseline_candidates=old_queries,direct_candidates=new_queries,
        elapsed_ratio=old_seconds/new_seconds if paired_pass else None,
        candidate_ratio=old_queries/new_queries if paired_pass else None,
        note='Historical single-seed comparison; first position identical initial candidates and prefix. Later recovered prefixes differ.')
    timing={}
    for name,op in [('real_spsa_l2_5','vision_query'),('direct_spsa_l2_5','query_embeddings')]:
        calls=[json.loads(s) for s in (HERE/name/'device/rpc.jsonl').read_text().splitlines()]
        batches=[r for r in calls if r['op']==op and r.get('position')==5 and r.get('payload_bytes')==128*1024]
        timing[name]=dict(full_batches=len(batches),
            median_host_seconds=statistics.median(r['host_seconds'] for r in batches),
            median_decoder_seconds=statistics.median(r['decoder_seconds'] for r in batches))
    comparison['first_position_full_batch_timing']=timing
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    lines=['# 逐位置直接恢复 adapter 后向量：手机实验','',
        f"结果：计划 {result['planned_positions']} 个位置，实际尝试 {result['attempted_positions']} 个，"
        f"通过 {result['accepted_positions']} 个。状态 `{result['status']}`。",'',
        '## 方法与隔离','',
        '原方法优化每位置 768 维 pre-adapter latent，每次经过手机 CPU adapter，再由 NPU 评分。'
        '新方法直接优化 1536 维 post-adapter 向量，每次仅由 NPU 评分。两者都用 64 个随机方向的'
        '双边 SPSA 梯度估计和 Adam；学习率 0.05，扰动 0.1，最多 1,000 轮。', '',
        '**严格一个位置一个位置恢复。** 一批 128 个向量是同一位置的不同候选，'
        '共享一个固定前缀，不是同时恢复 128 个位置。查询及实际提交均须原始 K+V L2 < 5，'
        '通过才进入下一个位置。后续前缀只由已恢复向量计算，不导入观测 KV。', '',
        '新方法初始化时把与原方法 seed 29 相同的 128 个随机 latent 通过手机 adapter 一次，'
        '得到完全相同的初始候选池；迭代阶段不再调用 adapter。'
        '以初始候选的整体 RMS 作一个标量单位转换，之后向量自由更新，不执行单位球面投影。'
        '这比较的是两种搜索参数化，不是单纯删除一个算子的消融。', '',
        '## 实测结果','',
        '| 图片位置（绝对位置） | 原方法查询数 | 直接向量查询数 | 原方法时间 | 直接向量时间 | 直接向量提交 L2 | 通过 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        a,b=r['baseline'],r['direct']
        lines.append(f"| {r['position']-4}（{r['position']}） | {a['queries']:,} | {b['queries']:,} | "
            f"{a['seconds']:.2f} s | {b['seconds']:.2f} s | {b['commit_l2']:.6f} | {'是' if r['both_pass'] else '否'} |")
    lines+=['',f'原方法这些位置搜索时间合计 {old_seconds:.2f} s；新方法 {new_seconds:.2f} s。',
        f"新方法一次性初始化投影另耗时 {result['initialization_seconds']:.4f} s；模型/设备初始化 "
        f"{result['device_setup_seconds']:.2f} s 未计入搜索时间。",'']
    if paired_pass:
        lines.append(f'同范围实测时间比（原方法 / 新方法）为 {old_seconds/new_seconds:.3f}，'
                     f'逻辑候选数比为 {old_queries/new_queries:.3f}。大于 1 才表示新方法更快或查询更少。')
    else:
        lines.append('存在未通过的位置，因此不把少运行的位置或未完成的搜索当成成功加速。')
    lines+=['','第一位置 128 候选满批调用的中位数：','',
        '| 路径 | 主机观测调用时间 | 其中 NPU decoder |',
        '|---|---:|---:|']
    for name,description in [('real_spsa_l2_5','经 adapter'),('direct_spsa_l2_5','直接向量')]:
        t=timing[name]
        lines.append(f"| {description} | {t['median_host_seconds']*1000:.2f} ms | {t['median_decoder_seconds']*1000:.2f} ms |")
    lines+=['','## 解释范围与验证','',
        '- 第一位置的公开前缀及初始候选相同；后续位置各自使用本方法恢复的前缀，不能视为完全相同条件的逐位置配对。',
        '- 原方法是保留的历史运行，未随机交错重跑；单样本、单 seed 不足以建立普遍性能结论。',
        '- 时间包含电脑到手机的通信、输入准备及 NPU 前向；不是纯 NPU 耗时。直接向量每候选传 1536 个 float，原路径为 768 个，通信量更大。',
        '- 每轮 128 个方向探测候选加 1 个更新点，仍是两个固定 128-slot NPU 调用。',
        '- 本轮只检查前几个位置，不验证整图语义、像素重建或后续文字恢复。',
        '- 手机控制：adapter 路径与直接路径 KV 一致、重排一致、查询不推进位置、查询与提交一致；见 `controls.json`。',
        '- 框架测试 76 项：74 通过、2 项可选测试跳过；独立直接优化器测试 2 项通过。',
        '- 新接口只加入测试 APK；JSON 帧上限从 1 MiB 调至 2 MiB，以容纳 128×1536 float32 的十六进制输入。原文本搜索和应用推理库未改。',
        '', '证据：`protocol.json`、`comparison.json`、各位置的 `trace.jsonl` / `result.json`、`device/`。','']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
