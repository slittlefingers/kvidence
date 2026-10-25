# 100 条明文 KV 数据集及样本 0076 恢复验证

已采集并整理 **100/100 条** `pilot_100_short` 原始明文，Pixel 10 / Gemma 4 E2B NPU，每条独立 Session，生成回复后导出全部可访问 KV。按用户指定，从新数据集中抽一条对比两种恢复方法；本报告的恢复准确率只覆盖 **1 条样本**。

## 数据集

数据目录：[plaintext_chat_100](../../../datasets/shadow_chat/plaintext_chat_100/README.md)。目标文本为原始 user prompt，长度 11–37 token，平均 17.49，总计 1749 token；保持原文，不拼接、截断、补长或添加存储问题。

每条保存 `all_layers.gkv.gz`、布局、回复和采集校验；另提取第 5 层（producer 索引 4）的 prompt KV，保存原始 INT8 码、位置、4-token 公开模板前缀及每张量量化参数。全量快照含运行时仍保留的 prompt/回复状态，不能假设滑动窗口层保留全部历史。单独的原文/token 标签只用于事后评价。

采集总墙钟 2842.77 秒（47.38 分钟）；推理耗时之和 2614.69 秒；压缩快照总计 340,723,185 bytes。全部下载哈希、gzip CRC 和布局校验通过。100 条明文与既有密文输入的配对已通过 AES-GCM 认证核验，旧密文 KV 哈希也通过检查；密钥没有复制给恢复进程。

既有 **100 条密文输入的 KV** 已统一归档到 [encrypted_chat_100](../../../datasets/shadow_chat/encrypted_chat_100/capture)，原始快照内容保持不变。两套数据按 ID 配对。这里的“密文 KV”指加密文本输入产生的 KV；快照本身是 gzip，并未加密存储。

## 采集前抽定的验证样本

使用 `random.Random(20261019).choice(sorted(ids))`，采集及恢复之前确定样本 `0076`，共 **15 token**：

> is NAME_1 a oxidisable, enzymatic, surfactant or particulate stain remover

两方法各运行两次，顺序 CPA → guided、guided → CPA。只复制此样本的最小 KV 观测到框架实验目录，哈希与数据集原件相同；恢复 audit guard 禁止读取源数据和私有标签。公开模板 3 个非 BOS 位置的对齐检查通过后才启动恢复，各方法使用自身恢复前缀。

## 实测结果

| 重复 | 方法 | 正确 token | 有效候选数 | 物理候选调用 | 在线秒数 | 主模型调用秒数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | collision_plus_cpa | 15/15 | 235,264 | 1838 | 419.401 | 115.275 |
| 1 | guided | 15/15 | 2,860 | 71 | 24.570 | 5.537 |
| 2 | guided | 15/15 | 2,860 | 71 | 16.353 | 5.470 |
| 2 | collision_plus_cpa | 15/15 | 235,264 | 1838 | 426.566 | 116.900 |

平均在线时间：CPA **422.984 秒**，guided **20.461 秒**，时间之比 **20.67×**。这是本样本和当前执行方式的端到端时间比。

未发现 token ID 误接受。

- `001_collision_plus_cpa`：阶段 {'probability': 15}；非零接受距离 0/15，最大 L2=0.000000；恢复文本字节完全一致：是。
- `001_guided`：阶段 {'ridge': 9, 'probability': 6}；非零接受距离 0/15，最大 L2=0.000000；恢复文本字节完全一致：是。
- `002_guided`：阶段 {'ridge': 9, 'probability': 6}；非零接受距离 0/15，最大 L2=0.000000；恢复文本字节完全一致：是。
- `002_collision_plus_cpa`：阶段 {'probability': 15}；非零接受距离 0/15，最大 L2=0.000000；恢复文本字节完全一致：是。

两次重复非计时 trace 字段一致：{'collision_plus_cpa': True, 'guided': True}。全部 commit/query KV 一致，worker 正常退出。

## 查询集中在哪里

本样本的两次重复均恢复正确，全部接受距离为 0，因而两方法的已恢复前缀相同。主要差距集中在以下两个 token：

| 原文位置 | token 片段 | CPA 概率排名（1 起） | CPA 实际计算候选 | guided 实际计算候选 |
| --- | --- | ---: | ---: | ---: |
| 2 | ` NAME` | 66,144 | 66,304 | 320 |
| 6 | ` oxid` | 161,434 | 161,536 | 320 |

` oxid` 是 `oxidisable` 的第一个 token，后面另有 `isable`。两个位置的完整批候选合计 227,840，占 CPA 全部候选约 96.84%。本样本含 `NAME_1` 占位符和专业词汇，不能把其概率排名或加速比推广为普通聊天的平均情况。

CPA 各位置首个通过者的概率排名合计 **232,683**，实际批量计算 **235,264**；逻辑批尾部多算 **2,581** 个。这里的顺序检查数量是 trace 推导值，不是实测串行耗时。当前巨大候选量主要来自低概率位置，不能只用 batch=256 解释。

guided 有 6 个位置在概率阶段通过，9 个位置经过 ridge。候选数量两次均为 2,860，非计时轨迹完全一致；在线耗时却为 24.570 / 16.353 秒。差异主要在手机端非主模型处理（13.507 / 6.406 秒），其次是主机评分（3.605 / 2.626 秒）。本次没有控制温度、频率、页缓存和调度状态，不把该波动归因于算法改变，也不只采用较快一次。

## 时间口径与基线限制

| 互斥时间分量 | CPA（秒） | guided（秒） |
| --- | ---: | ---: |
| 主 decoder/prefill 调用 | 116.088 | 5.504 |
| 手机端其余处理 | 281.559 | 9.956 |
| 主机评分/选择 | 0.662 | 3.115 |
| RPC 主机计时减 worker | 19.258 | 1.305 |
| 其余主机处理 | 5.417 | 0.581 |
| 在线总时间 | 422.984 | 20.461 |

Online 不含原始推理采集、初始化、公开前缀预填及离线校准。主模型时间是 LiteRT 调用墙钟，含运行时/驱动开销，不是 NPU 内核计数。RPC 差额还含协议处理和调度。

两边使用当前 `socket + optimized_v2`、完整 262144 词表、逻辑 batch 上限 256、物理 `prefill_128`，无 600 次查询上限。guided 仍为 50 概率 + 150 随机初始候选，每轮 20 个 ridge 提议，原始 K+V L2<4；这里的 4 不是 LSB。

CPA 保持官方概率顺序、独立 K/V 阈值和 1% 容差，仍采用 **8 位置×1024 候选 pilot 校准，未完成全文全词表校准**。它算完整逻辑批后取首个通过者，所以“计算候选数”包含最后一批多算的候选，不等于命中 token 的概率排名。逐位置表同时记录 CPA 排名与计算数量。

两方法验收区域、最小验收批次及提交方式不同：CPA 保留所有层候选 KV；guided 查询目标层后重新 decode 提交。因此速度比不能全部归因于 ridge 搜索。出现误接受时后续前缀/概率排名也会分叉。当前只验证一条，不报告 100 条恢复准确率或泛化保证。

## 原始记录

- [成对汇总](SUMMARY.md)、[分析 JSON](analysis.json)、[逐位置比较](position_comparison.json)。
- 各运行目录保存 `evaluation.json`、`trace.jsonl`、`recovered.txt`、`protocol.json` 和 `device/rpc.jsonl`。
- [数据集 manifest](../../../datasets/shadow_chat/plaintext_chat_100/manifest.json)、[统计](../../../datasets/shadow_chat/plaintext_chat_100/dataset_summary.json)、[文件校验值](../../../datasets/shadow_chat/plaintext_chat_100/checksums.json)。
- [选择与输入来源](../../experiments/plaintext_short100_validation/input/provenance.json)、[前缀验证](../../experiments/plaintext_short100_validation/prefix_validation/validation.json)。
