# 单条短密文：模型概率排序与 KV 引导搜索对比

本次固定选取短版 LMSYS 加密数据集的 `0001`，不按攻击结果挑样本。目标是恢复模型实际读入的 Base64 密文数据包；不尝试解密 AES 内部明文。

## 顺序恢复结果

恢复进程只读取公开提示词前缀、目标第 5 层 KV 和独立标定阈值。每种方法沿自己恢复出的前缀推进；没有将真实前序 token 喂回模型。无候选通过阈值时停止。

| 方法 | 正确恢复 token / 全部 | 连续正确前缀 | 完整恢复 | 候选查询总数 | 搜索及推进时间 |
|---|---:|---:|---|---:|---:|
| Collision+ 概率排序适配版 | 0/109 | 0 | 否 | 600 | 0.167 s |
| KV 引导（初始 200） | 49/109 | 49 | 否 | 13,540 | 9.798 s |

两种方法恢复的长度不同，因此上述总耗时不能用来计算恢复相同文本的加速比。候选查询包括失败位置的全部查询；标定、模型加载和最初的已知前缀准备另计。

## 放宽概率基线预算的补充实验

此轮在观察到 600 次预算不足后追加，使用同一个输入、模型、候选池、独立 CPA 阈值和读取 guard。允许概率基线继续遍历候选池，直到匹配或耗尽。它是更高预算的补充结果，不属于前面的等预算比较。

顺序恢复正确 **109/109 token**，连续正确前缀 109，完整恢复：是；实际候选查询 **57,920** 次，搜索及推进时间 **16.902 s**。

这一对照用于判断概率方法是否只是被查询上限截断，不能把较小预算下的停止解释为算法在任何预算下均不可恢复。

## 已知正确前缀的逐位置对照

以下是搜索诊断：109 个位置各自使用真实正确前缀，强制完成相同预算，并按已查询候选的实际 KV NMSE 最小值计分。它用于区分候选排序和前序错误传播，不是从未知前缀完整恢复密文的成功率。

| 每位置候选预算 | Collision+ 概率顺序：最近 KV 正确数 | KV 引导：最近 KV 正确数 |
|---:|---:|---:|
| 200 | 69/109 | 1/109 |
| 220 | 70/109 | 44/109 |
| 240 | 70/109 | 70/109 |
| 300 | 74/109 | 91/109 |
| 400 | 75/109 | 102/109 |
| 600 | 79/109 | 108/109 |

| 方法 | 固定预算查询总数 | 候选前向时间 | 拟合与排序时间 | 搜索总时间 |
|---|---:|---:|---:|---:|
| Collision+ 概率排序适配版 | 65,400 | 13.157 s | 0.005 s | 14.398 s |
| KV 引导（初始 200） | 65,400 | 12.215 s | 106.053 s | 119.513 s |

正确 token 在模型概率排序中的名次：最小 1，中位数 36.0，平均 523.2，最大 6029。第一个 token 的名次为 6029。

固定预算下，KV 引导在第 50 个位置漏查了正确 token `6`，该 token 在语言概率排序中为第 10 名。该位置在已知正确前缀的 600 次控制实验中也未被 KV 引导查到；这与先前错误 token 导致的前缀污染不同，说明两种排序的优势可能互补。

## 固定设置与实现

- 模型：`mlx-community/gemma-4-e2b-it-8bit`，revision `03dcf209f3f549b4075e7191e77cf69b3d48e1b2`。权重为 8-bit affine，group size 64；本次真实 KV 为 `mlx.core.bfloat16`，不是 INT8 KV。
- 后端：Apple MLX GPU；观察第 5 层独立 full-attention KV，K+V 共 1024 维。聊天模板完整输入 132 token，其中已知前缀 11，未知密文 109。
- 候选池：完整词表中独立解码结果匹配 `[A-Za-z0-9+/=]+` 的非特殊 ID，共 53,150 个。两种方法共享该池，未根据目标 ID 缩小到一个方便的子集。
- 模型概率排序：用原模型全部 35 层的条件 logits 排候选；不采用概率尾部额外截断。每批 20 个，最多 600 个不同候选。
- KV 引导：原 `kv_guided_search_gemma4_int8/search.py` 的 `proxy_ranking` 函数原样导入。每位置先随机查询 200 个候选，用主 embedding 到真实 K+V 的标准化 dual ridge（ridge=0.01）拟合，再每批提出 20 个候选，最多 600 次。使用 seed 20261019；未添加单字符兜底、未调参。
- 两边的候选 KV 查询都只执行到第 5 层，微批大小 64。概率方法接受候选后执行完整模型以获得下一位置 logits；KV 引导只推进所需层。

## Collision+ 适配与阈值边界

参照官方实现的概率排序、独立 K/V L2 距离、CPA 阈值及 `distance < 1.01 * threshold` 规则：概率顺序接受首个通过两项阈值的候选；KV 引导在已查询并通过同样阈值的候选中选最小真实 KV NMSE。

标定使用 3 条独立公开 Base64 输入的 12 个位置，每位置抽 199 个错误候选并查询正确候选的批量/单独前向，共 2412 次候选查询，耗时 1.582 s。K/V 阈值分别为 0.356486 / 6.854085。目标密文不参与标定。

这是 Gemma/MLX 上的 Collision+ 方法适配，不是直接运行原论文 CUDA/PyTorch 程序。原始标定脚本遍历完整词表；本实验标定错误候选有限，阈值不构成全域唯一性证明。逻辑批大小、截断预算和后端也不同，不能把这次失败解释为原论文所有设置均失败。

本次没有施加 INT8 KV 量化，也没有以单维几 LSB 作验收；BF16 形状差异通过独立 CPA 标定处理。原始距离、查询 ID 和预算轨迹完整保存。

## 校验与复现

记录审计：`passed=True`；重算全部初始 200 候选、概率排序前 600 项、唯一查询数、预算、阈值与顺序恢复记录；私有输入读取 guard 的两次负向检查均拒绝读取。

```bash
cd /path/to/litertlm
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
EXP=kvidence/experiments/ciphertext_collision_comparison
# 使用新输出目录避免覆盖本次结果，例如 --output "$EXP/results-new"。
"$PY" "$EXP/experiment.py" acquire
"$PY" "$EXP/experiment.py" recover
"$PY" "$EXP/experiment.py" diagnostic
"$PY" "$EXP/unpruned_probability.py"  # 补充较高预算概率对照，默认结果目录
"$PY" "$EXP/experiment.py" evaluate
"$PY" "$EXP/validate.py"
"$PY" "$EXP/report.py"
```

`results/evaluation.json`：结果；`results/sequential/`：真实顺序恢复轨迹；`results/diagnostic/`：已知前缀预算对照；`results/calibration.json`：独立标定；`results/observed/`：恢复输入；`results/private_eval/`：只供采集和评估的真值。GPU 命令需要 Metal 权限。

本次为一条样本、一个搜索种子、约百 token 的短上下文实验。没有模拟浅层滑动窗口淘汰，不是手机 NPU 或物理内存取证结果，也不能据此推广为所有密文均可完整恢复。

参考：[Collision+ 官方实现](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/collision.py)、[官方阈值标定](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/get_collision_threshold.py)。固定版本与 SHA-256 存于 `reference/manifest.json`。
