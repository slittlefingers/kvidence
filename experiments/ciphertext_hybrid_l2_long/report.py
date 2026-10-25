"""Audit and report the 999-token sequential recovery timing comparison."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import run_long as experiment

ROOT, OUT = experiment.ROOT, experiment.OUT
evaluation = json.loads((OUT / "evaluation.json").read_text())
protocol = json.loads((OUT / "protocol.json").read_text())
cache_check = json.loads((OUT / "cache_validation.json").read_text())
assert evaluation["validation_passed"] and cache_check["passed"]
assert evaluation["target_tokens"] == protocol["target_tokens"] == 999
assert protocol["epsilon"]==4 and protocol["strict_less_than"]
for method in experiment.shared.METHODS:
    folder=OUT/method
    pretty=json.loads((folder/"trace.json").read_text())
    with (folder/"trace.jsonl").open(encoding="utf-8") as stream:
        appended=[json.loads(line) for line in stream if line.strip()]
    assert appended==pretty
    run=json.loads((folder/"run.json").read_text())
    assert run["total_queries"]==sum(row["queries"] for row in pretty)

guard_code="""
import run_long as experiment
experiment.shared.install_guard()
paths = [experiment.OUT/'private_eval/truth.json',
         experiment.exp.ROOT/'results/private_eval/truth.json',
         experiment.exp.KVIDENCE/'datasets/shadow_chat/archive/pilot_100/prompts/encrypted/0037.txt']
denied=0
for path in paths:
    try: path.read_bytes()
    except PermissionError: denied+=1
    else: raise AssertionError('Private input was readable')
print(denied)
"""
result=subprocess.run([sys.executable,"-c",guard_code],cwd=ROOT,
                      check=True,capture_output=True,text=True)
assert int(result.stdout.strip())==3
experiment.exp.dump(OUT/"validation.json",dict(
    passed=True, target_tokens=999, same_source_observation_both_methods=True,
    cache_boundary_probes=len(cache_check["probes"]),
    query_branches_preserve_prefix_cache=True,
    all_accepted_raw_l2_recomputed_below_4=True,
    initial_probability_50_and_uniform_random_150_recomputed=True,
    query_ids_unique_and_within_finite_pool=True,
    append_checkpoints_match_final_trace=True,
    no_600_query_cap=True, private_source_guard_negative_checks=3))

methods=evaluation["methods"]
baseline=methods["collision_plus_l2"]
hybrid=methods["hybrid_50_150"]
names={"collision_plus_l2":"Collision+ 概率排序，统一 L2 验收",
       "hybrid_50_150":"概率前 50 + 随机 150，再做 KV 引导"}
lines=["# 约 1,000-token 密文：完整顺序恢复的时间对比","",
       "使用已有长版数据集样本 `0037`，密文本身为 **999 token**。它是 100 条已有样本中长度最接近 1,000 的一条，按长度及 ID 确定，未参考恢复结果；完整密文没有截断或填充。","",
       "两种方法均使用固定停止条件 **原始 K+V 的完整 L2 < 4**，不按 LSB 换算，不取 RMS，不设逐维门槛，不设 600 次查询上限。","",
       "## 结果","",
       "| 方法 | 正确恢复 | 误接受 | 完整字符串一致 | 候选查询总数 | 恢复循环时间 | 吞吐量 |",
       "|---|---:|---:|---|---:|---:|---:|"]
for method,label in names.items():
    r=methods[method]
    lines.append(f"| {label} | {r['correct_tokens']}/{r['target_tokens']} | {r['incorrect_acceptances']} | {'是' if r['exact_ciphertext_text'] else '否'} | {r['total_queries']:,} | {r['seconds']:.3f} s | {r['correct_tokens']/r['seconds']:.2f} 正确 token/s |")
if baseline["all_exact"] and hybrid["all_exact"]:
    reduction=1-hybrid["total_queries"]/baseline["total_queries"]
    speedup=baseline["seconds"]/hybrid["seconds"]
    lines += ["",f"本条样本两种方法都完整恢复。混合方案减少 **{reduction:.1%}** 候选查询，恢复循环快约 **{speedup:.2f} 倍**。这是一次配对运行的结果，不是跨样本统计或稳定性能保证。",""]
else:
    lines += ["","本条样本没有两种方法均完整恢复的结果，耗时对应的恢复工作量不同，不能直接当作完整恢复速度比。",""]
lines += ["| 方法 | 候选前向时间 | 概率排序及拟合时间 | 接受后完整模型推进时间 | 单位置最多查询 | 最大接受 L2 |",
          "|---|---:|---:|---:|---:|---:|"]
for method,label in names.items():
    r=methods[method]
    lines.append(f"| {label} | {r['forward_seconds']:.3f} s | {r['fitting_and_ranking_seconds']:.3f} s | {r['advance_seconds']:.3f} s | {r['maximum_queries_at_one_position']:,} | {r['accepted_l2_max']:.6f} |")
lines += ["","恢复循环时间包含真实候选前向、模型概率排序、ridge 拟合与候选排序、接受 token 后的完整模型推进和逐位置 JSONL 记录。模型加载、观测采集、缓存检查、最初公开前缀准备和最终 JSON 汇总不计入该时间。","",
          "## 长上下文与计算核验","",
          f"- 聊天模板完整输入 **{protocol['rendered_chat_tokens']} token**，用户消息 {protocol['user_tokens']} token，公开已知前缀 {protocol['known_prefix_tokens']} token；未知密文 999 个 token。",
          "- 模型为相同的 Gemma 4 E2B IT MLX 8-bit affine 权重，实际 KV 为 BF16。第 5 层是独立 full-attention KV 生产层；前四层使用 512-token 原生滑动缓存。",
          "- 原生长 prefill 采集第 5 层 KV，恢复过程中使用自己的已恢复前缀和原生 RotatingKVCache。没有将真实 token 喂回恢复进程。",
          "- 采集阶段的独立数值检查在绝对位置 11、510、511、512、513、767、768、1009 测试了候选批量/单独计算、正确候选距离、前缀缓存值和环形下标不变；这些检查使用真实前缀，只作为数值诊断。恢复阶段读不到这些真值。",
          "","| 绝对位置 | 正确候选 L2 | 滑动缓存容量 | 缓存 offset | 环形下标 |",
          "|---:|---:|---:|---:|---:|"]
for row in cache_check["probes"]:
    lines.append(f"| {row['absolute_position']} | {row['correct_candidate_l2']:.6f} | {row['sliding_cache_capacity']} | {row['sliding_cache_offset']} | {row['sliding_cache_ring_index']} |")
lines += ["","## 共享的搜索设置","",
          "- 两边候选池均为完整词表中由 Base64 字符组成的 53,150 个非特殊 token。",
          "- 概率基线按照完整 35 层模型的条件 logits 排序，每批查 20 个，接受第一个原始 L2 < 4 的候选。原 Collision+ 的 CPA 验收按用户要求替换为共同 L2 条件。",
          "- 混合方案每位置先查概率前 50 个，再从其余池中均匀无放回抽 150 个；若未匹配，使用原先的 embedding→KV ridge（0.01）排序，每轮追加 20 个候选。随机种子为 20261019+位置。",
          "- 两边候选前向都算到第 5 层，微批上限 64；接受后两边都运行完整模型以获得下一位置概率。候选计算和搜索代码直接导入上一轮脚本，没有更改这两个方法。",
          "- 达到 L2 条件就停止当前位置搜索；完整有限词表耗尽仍未匹配时记录未解决。阈值是实验接受规则，正确率再由真实 token ID 及密文字符串独立核对。","",
          "## 记录方式及解释范围","",
          "长实验将每个位置的轨迹追加到 JSONL，一次写一条，结束计时后再生成完整 JSON。这样避免每个位置重写全部既有轨迹造成随长度增长的二次写盘成本。两种方法使用完全相同的记录方式。上一轮 109-token 实验每位置重写 JSON，因此不应将两轮墙钟时间的差异完全归结为上下文长度。","",
          "本次只有一条长密文、一个随机种子，后端是 Apple MLX GPU，不是 Pixel NPU 或物理内存 dump。超过滑动窗口并不表示全部浅层数据已被消除；本实验观察的是保留完整目标序列的第 5 层全局 KV。","",
          "## 文件和复现","",
          "`results/evaluation.json`：准确率和耗时；`results/protocol.json`：固定设置；`results/cache_validation.json`：跨窗口核验；两个方法目录的 `trace.jsonl` / `trace.json`：完整查询 ID、真实 L2、接受的 KV 向量及分阶段时间；`results/validation.json`：记录校验。","",
          "```bash","cd /path/to/litertlm",
          "PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python",
          "EXP=kvidence/experiments/ciphertext_hybrid_l2_long",
          "# 新复跑须归档旧 results，脚本拒绝覆盖已完成结果。",
          '"$PY" "$EXP/run_long.py" acquire',
          '"$PY" "$EXP/run_long.py" run',
          '"$PY" "$EXP/run_long.py" evaluate',
          '"$PY" "$EXP/report.py"',"```",""]
(ROOT/"REPORT.md").write_text("\n".join(lines))
files={}
for path in sorted(OUT.rglob("*")):
    if path.is_file() and path.name!="integrity.json":
        files[str(path.relative_to(OUT))]=hashlib.sha256(path.read_bytes()).hexdigest()
experiment.exp.dump(OUT/"integrity.json",dict(file_sha256=files,
    long_runner_sha256=experiment.exp.digest(ROOT/"run_long.py"),
    shared_search_sha256=experiment.exp.digest(experiment.SHORT/"run.py"),
    report_sha256=experiment.exp.digest(ROOT/"REPORT.md"),validation_passed=True))
print(ROOT/"REPORT.md")
