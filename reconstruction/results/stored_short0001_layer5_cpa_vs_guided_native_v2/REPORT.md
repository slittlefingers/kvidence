# 第 5 层 KV 恢复：优化后的 CPA 与 guided 对比

日期：2026-10-19（Australia/Melbourne）。使用 Pixel 10 / Gemma 4 E2B / 已有数据库样本 0001，直接恢复已保存的第 5 层 INT8 KV 对应的 109 个密文 token。两种方法各运行两次，均使用当前 `socket + optimized_v2` 执行器；每个进程独立重建前缀，全部完成后才使用私有真值评价。

## 主要结果

CPA 平均在线 **205.948 秒**，guided **95.372 秒**，本样本中平均在线时间之比为 **2.16×**。候选数之比为 5.95，guided 查询减少 83.18%；候选数比例不能称作 NPU 加速比。

| 重复 | 方法 | 正确 token | 有效候选数 | 候选物理调用 | 在线秒数 | 主模型调用秒数 | 整个进程秒数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | collision_plus_cpa | 108/109 | 98,304 | 768 | 206.100 | 55.047 | 215.737 |
| 1 | guided | 109/109 | 16,530 | 388 | 95.359 | 30.989 | 109.936 |
| 2 | guided | 109/109 | 16,530 | 388 | 95.385 | 31.109 | 109.979 |
| 2 | collision_plus_cpa | 108/109 | 98,304 | 768 | 205.797 | 54.814 | 217.979 |

执行顺序：第 1 轮 CPA → guided；第 2 轮 guided → CPA。在线耗时范围：CPA 205.797–206.100 秒，guided 95.359–95.385 秒。两次重复只能观察此样本的运行波动，不能估计数据集准确率或可靠的置信区间；交替执行也没有控制设备温度、频率和其他调度状态。

## 数据与公平性边界

- 观测来自 `kvidence/datasets/shadow_chat/encrypted_chat_100/capture/0001/all_layers.gkv.gz`，在模型生成回复后采集。本次没有重新采集目标 KV。
- 恢复 producer 索引 4（第 5 层）的 KV 位置 11–119；11 个模板 token 是公开前缀。观测包含的完整输入/回复长度不等于本次恢复长度：**这次恢复 109 个密文 token，不是 1000 token，也不是 100 条数据。**
- 两边使用完整原始 262144 词表、逻辑 batch 上限 256、实际编译接口 `prefill_128`；不筛选语言/Base64 字符，不设 600 次查询上限。
- guided 初始最多 50 个概率候选和 150 个随机候选，每轮 20 个 ridge 提议，seed=20261019。停止要求原始数值空间中拼接 K+V 的 **L2 < 4**，不是 4 LSB。完整接受阶段中取距离最小的通过候选。
- CPA 是按官方规则移植的 NPU 版本：概率顺序中首个通过 K/V 双阈值的候选，rank=8、官方 1% 容差。**仍采用独立已知文本 8 个位置×1024 候选的 pilot 校准，并未完成全文全词表校准。**有效阈值为 K<0.3821817103、V<10.3494870758。
- 两种方法的验收区域不同，这不是只改变搜索策略的同阈值消融。CPA 的 NPU 移植保留所有层候选 KV，用于提交；guided 查询只读目标层，提交时重新运行 decode。这些执行差异均包含在耗时中。主模型 compiled prefill 不导出 logits，CPA 另用 decode 取得下一位置排名所需 logits，保留查询 KV 作为前缀。
- 成对汇总已检查四次运行的模型、观测、原词表、batch、源码哈希、runtime、APK、transport 和 native mode 一致。模型预测来自同一 NPU 模型。主机 embedding 表只用于提议，候选验收均使用真实 NPU KV。

## 准确性及重复一致性

- `001_collision_plus_cpa`：密文第 98 个 token（KV 位置 108），`se`（561）→`Se`（1869）；拼接 L2=9.117676，K=0.371137，V=9.110119。
- `002_collision_plus_cpa`：密文第 98 个 token（KV 位置 108），`se`（561）→`Se`（1869）；拼接 L2=9.117676，K=0.371137，V=9.110119。

各次正确率、密文字节一致性、接受距离非零数量及最大值均见 [analysis.json](analysis.json)。输出是密文的文本表示，没有解密 LMSYS 明文；独立真值解码长度为 156 bytes。

两次 CPA 均复现第 98 个 token 的大小写误接受：前 97 个位置正确且 L2=0，后续 11 个 token 正确但 KV 距离非零。两次 guided 均为 109/109 正确、所有接受距离为 0，恢复的 156-byte 密文与真值完全相同。该结果受 CPA pilot 阈值范围限制，不能解释为正式全量校准的原论文方法也必然误接受。

本次两次重复的全部非计时 trace 字段逐项一致；与历史文件通信版相比，候选顺序、候选距离、接受 token/原始 KV、阶段记录和 commit/query 一致性也逐项核对，结果见 `trace_equivalence`。四个 worker 均正常结束，无 commit mismatch。重复采用相同 seed，因此查询轨迹相同不是跨随机种子的稳定性证据。

## 时间分解（两次运行的平均值）

| 互斥时间分量 | CPA（秒） | guided（秒） |
| --- | ---: | ---: |
| 主 decoder/prefill 调用 | 54.931 | 31.049 |
| 手机 worker 内部其余处理 | 133.042 | 35.762 |
| 主机候选评分/选择 | 0.250 | 16.208 |
| RPC 主机计时减 worker 计时 | 12.567 | 8.196 |
| 其余主机处理 | 5.159 | 4.157 |
| 在线总时间 | 205.948 | 95.372 |

Online 包含当前恢复过程的候选准备、模型调用、KV 处理、通信、评分和提交，不含初始化、资产加载、公开前缀预填，也不含离线 CPA 校准和主机 embedding 资产准备。进程墙钟时间另列在上表；它包含启动、检查、预填、保存和退出。

主模型时间为主 decoder/prefill 的 `LiteRtRunCompiledModel` 累计墙钟，包含运行时/驱动开销，不是 NPU 内核时间；CPU embedder、PLE 和辅助图计时在手机其他处理中。RPC 差额含通信、主机协议处理和调度，不能全称为传输时间。

以下是上述分量内部的细项，**不可再次与上表相加**：

| 嵌套细项 | CPA（秒） | guided（秒） |
| --- | ---: | ---: |
| 候选 embedding 准备与拼装 | 31.219 | 7.494 |
| 候选 KV 读回 | 34.687 | 2.098 |
| 候选 KV 整理与保留 | 34.757 | 0.914 |
| 候选输入写入 | 1.557 | 0.828 |
| 手机完整概率排名排序 | 10.151 | 10.362 |

从本次计时看，CPA 的手机端非主模型处理仍占较大部分，包括 embedding 准备，以及为提交保留所有层候选 KV 所需的读回和整理。guided 已减少这些成本，但仍有主机 ridge 评分、手机排名与候选准备。当前尚不能说剩余时间都是 NPU 计算，也不能说已达到硬件性能上限。

完整排序仍返回全词表，每位置 1 MiB ID 数据；当前优化没有缩小词表或仅返回 top-50。两个方法查询个数与物理调用比不同，因为编译接口每次执行 128 槽，guided 的部分阶段没有填满有效候选。

## 与历史实现对照

| 方法 | 历史文件通信版在线秒数（一次） | 当前平均在线秒数（两次） | 历史 / 当前 |
| --- | ---: | ---: | ---: |
| collision_plus_cpa | 1718.499 | 205.948 | 8.34× |
| guided | 610.764 | 95.372 | 6.40× |

历史数据保留在 [旧报告](../stored_short0001_layer5_cpa_vs_guided/REPORT.md)。这是不同时间采集的实现版本对照，不是交错控制的优化消融；不能把全部差额归因于某一项改动。当前两方法速度比使用本次四次实测计算，不沿用旧版比例。

## 复查与复现

- [成对运行汇总](SUMMARY.md)、[机器可读分析](analysis.json)、[逐位置对比](position_comparison.json)。
- 每个运行目录保存 `result.json`、`protocol.json`、`evaluation.json`、`trace.jsonl`、`recovered.txt` 和 `device/rpc.jsonl`；执行命令与顺序见 [comparison.json](comparison.json)。
- 当前独立校准资产：[calibration.json](../../assets/cpa_gemma4_pilot_native_v2/calibration.json)。
- 设备主运行库 SHA-256：`ce88d0add27860ebdd50295a5a2cfcce7c35ffb2c76dcf43827f35fd82f932a9`。
- instrumentation APK SHA-256：`c3ecffdb0e92172e900a6d9b7df3f562284014b6822d0baa4d5863876e36ee86`。
- 模型 SHA-256：`af1082986639ecde7db95d91be6fe54f8b6b458104734c5bafc204e69d6852dc`。

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
"$PY" "$APP/run.py" compare --model-name gemma4-e2b \
  --baseline collision_plus_cpa \
  --observation "$APP/experiments/stored_short0001_layer5/input/observation" \
  --calibration "$APP/assets/cpa_gemma4_pilot_native_v2/calibration.json" \
  --allow-pilot-calibration --batch-size 256 --transport socket \
  --native-mode optimized_v2 --repeats 2 --output "$APP/results/NEW_OUTPUT"
"$PY" "$APP/run.py" evaluate-comparison --comparison "$APP/results/NEW_OUTPUT" \
  --truth "$APP/experiments/stored_short0001_layer5/input/private_eval/truth.json"
```

这是已有真实设备观测上的单样本对比。正式结论仍需要完整 CPA 校准、多条输入、更多重复和明确的同阈值补充实验；本报告没有把这些尚未完成的实验填成结果。

## 配对明文实验

同一样本未加密原文（21 token）的实际恢复对比已完成，见 [明文报告](../plaintext_short0001_layer5_cpa_vs_guided_native_v2/REPORT.md)。该报告包含按 token 归一化的成本、概率/ridge 阶段统计及 CPA 的具体误接受证据；本页保留原密文结果。
