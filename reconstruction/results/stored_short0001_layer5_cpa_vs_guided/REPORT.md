# 样本 0001：已保存第 5 层 KV 的单条对比

> 历史文件通信版结果保留如下。当前优化版本已完成两次交替重复，见 [最新成对对比报告](../stored_short0001_layer5_cpa_vs_guided_native_v2/REPORT.md)。

日期：2026-10-19（Australia/Melbourne）。本次直接使用已有数据库中的 APK 快照，未重新推理或采集目标 KV。两个恢复进程均已完成，真值在运行结束后单独评价。

## 结果

| 指标 | Collision+ CPA，NPU 移植 | KVidence guided |
| --- | ---: | ---: |
| 正确 token | 108/109 | 109/109 |
| 整条完全一致 | 否 | 是 |
| 实际候选数 | 98,304 | 16,530 |
| 候选物理调用数 | 768 | 388 |
| 在线恢复时间 | 1718.50 s（28.64 min） | 610.76 s（10.18 min） |
| 模型调用累计时间 | 62.04 s | 33.16 s |
| 整条进程时间 | 1734.41 s | 627.84 s |
| 每位置候选数中位数 / 最大值 | 256 / 8960 | 50 / 440 |

在这一次运行中，guided 的在线时间为基线的 1/2.81，有效候选数减少 83.18%。物理调用数之比为 1.98，模型调用时间之比为 1.87。三种比例含义不同，不能把有效候选数比例称为 NPU 加速比。

## 观测与协议

- 输入来自 `kvidence/datasets/shadow_chat/encrypted_chat_100/capture/0001/all_layers.gkv.gz`，采集时点为生成回复后。
- 模型为 Pixel 10 上 Gemma 4 E2B Tensor G5；INT8 KV，第 5 层 producer（索引 4）。恢复密文触及的 109 个 token，KV 位置 11–119；11 个模板/说明 token 作为公开前缀。没有把目标真实前缀缓存装入恢复执行器。
- 两边完整原始词表 262144；不筛选 Base64 字符或语言，不设 600 次查询上限。成对运行逻辑上限 256，物理模型接口 128；guided 保持自身 50 概率 + 150 随机初始阶段和每轮 20 个引导提议。
- 恢复前仅检查公开前缀的 10 个非 BOS 位置，模型输出与已保存第 5 层 KV 距离均为 0，见 `experiments/stored_short0001_layer5/prefix_validation/validation.json`。
- 基线使用官方规则的 K/V 双阈值、1% 容差、rank=8；阈值来自独立的 **8 位置 × 1024 候选 pilot 校准**，不是完整校准。有效阈值 K<0.3821817103，V<10.3494870758。
- guided 使用原始 K+V 拼接 L2<4。两边验收区域不同；本次不是固定验收条件下仅替换搜索顺序的消融。
- 固定顺序：CPA 先、guided 后，各运行一次，未重复或交替。模型、观测、词表、软件源码及设备运行库身份由 paired protocol 检查。

## 唯一误接受及后续位置

CPA 在第 **98** 个密文 token（KV 位置 108）把 `se`（ID 561）选成了 `Se`（ID 1869）。该候选 K 距离约 0.371137，V 距离约 9.110119，均通过其校准阈值；拼接 L2 为 9.117676。它不是距离 0 的候选。

前 97 个位置，CPA 的 token 均正确且距离均为 0。误接受后，剩余 **11 个 token 仍全部正确**，但它们的 KV 距离为约 2.16–4.51。本次观察到错误前缀伴随后续 KV 偏差，但没有更多 token 错误；不能推广为错误不会累积的证明。guided 全部 109 个 token 正确，接受距离全部为 0。两边提交的目标层 KV 均与各自选中查询一致，没有 commit mismatch。

这里的 CPA 错误必须结合小规模校准理解：本次 pilot 没有覆盖所有相近错误候选，不能据此认定完成正式校准的原论文方法也会发生同样错误，也不能据此比较两者的普遍准确率。

## 时间口径与适用范围

Online 包含排序、候选准备、ADB、KV 读回、评分及提交，排除启动与公开前缀预填。Decoder 是 `LiteRtRunCompiledModel` 的累计调用耗时，包含接口执行开销，不是硬件内核周期计数。NPU 移植的 CPA 保留各层候选 KV 并额外取得 logits，存在相应传输/调度开销；guided 还包含约 26.79 s 的候选排序/引导评分。速度差不能全部归因于抽象算法的候选数。

单样本、固定执行顺序、小规模校准及不同验收标准限定了本结果。它验证了已有真实 APK 第 5 层 KV 可以接入当前框架，并提供了可复查的初步对比；不替代正式的校准和批量评估。

## 原始结果

- [成对汇总](SUMMARY.md)
- [逐位置对比](position_comparison.json)
- [分析数据](analysis.json)
- [CPA 独立评价](001_collision_plus_cpa/evaluation.json) / [恢复密文](001_collision_plus_cpa/recovered.txt)
- [guided 独立评价](001_guided/evaluation.json) / [恢复密文](001_guided/recovered.txt)
- 各目录 `trace.jsonl`、`protocol.json`、`device/rpc.jsonl` 保留逐步计算及身份记录。
