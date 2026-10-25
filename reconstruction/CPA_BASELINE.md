# Collision+ CPA 基线：论文与源码对齐

**当前实验决策：主对比采用官方 Collision+ 规则。** 默认 `compare` 使用 `collision_plus_cpa` 与 guided，成对运行默认逻辑 batch=256。基线使用独立 CPA 校准及 K/V 双阈值；guided 对整数 KV 默认使用 LSB 空间的 K+V L2<epsilon。两边不强行统一阈值，统一 L2 适配版只作显式选择的补充对照。完整校准尚未运行，当前 pilot 资产不能作为正式校准结果。

本页描述 `collision_plus_cpa`，其校准与接受规则未改变。LSB 迁移之后，整数 KV 的 `guided` 和 `collision_plus` 默认在整数刻度空间拟合、评分及验收；旧报告仍是原始数值指标，不重写。`--distance-space original` 可明确复现旧指标。

## 对齐依据

论文：[Shadow in the Cache, NDSS 2026](https://www.ndss-symposium.org/wp-content/uploads/2026-f258-paper.pdf)，§III-B 的式 (6)、§IV-B4、Appendix B/C。主实验的 rank 假设是 8；图中用于说明距离分布的 64 不是主实验默认值。论文 Appendix C 采用逻辑 batch=256。

实现依据：[官方 collision.py](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/collision.py)。本地 checkout 没有改动；固定版本 `6b40f36edb2f337557543e7e60b10022308883d4`。这是算法在 Gemma/LiteRT NPU 上的移植，不能表述为原 CUDA 实验的逐位复现。

| 项目 | 本框架 CPA 实现 |
| --- | --- |
| 校准 | 独立已知文本；每个位置在已知真实校准前缀下查询候选，区分真 token 和其他 token 的距离 |
| 默认候选覆盖 | 完整原始词表，不做语言筛选；只有显式 `--pilot-candidates` 才抽样 |
| 距离 | K、V 各自 Frobenius/L2；INT8 先反量化到原始单位，然后用 float32 计算；不平均 K/V、不按 LSB 归一化 |
| 统计 | 分别保存 K/V 的均值、样本标准差（ddof=1）、真 token 最大值、其他 token 最小值及样本数 |
| rank | 默认 8，可在校准时 `--rank` 指定 |
| 无重叠分支 | 只有 K、V 两者均满足真值最大距离 < 其他最小距离，才分别取间隔中点 |
| 有重叠分支 | 两个分量均调用官方目标函数及 bounded optimizer；target std=0 时保留官方直接取均值的行为 |
| 判定 | `dK < tauK + 0.01*tauK` 且 `dV < tauV + 0.01*tauV`，严格小于 |
| 选择 | 完成整个逻辑批次后，取模型概率顺序中首个通过者；不选最小距离 |
| 批量 | CPA 默认逻辑 256，Gemma NPU 物理 128，因此实际执行两次；可显式覆盖 |
| 搜索范围 | `--stop-partition 1` 默认全词表，对齐官方主入口；支持其截断及末批越界的语义 |
| 前缀推进 | 保留该逻辑批次所有候选的各层 KV 切片，提交命中候选本身的切片；检查提交的目标层切片与查询字节一致 |
| 失败 | 无通过候选就停止；不回退最近邻、不偷偷使用 epsilon=4 |

式 (6) 对应的阈值目标为：

$$
\tau^* = \arg\max_\tau F_{\mathrm{target}}(\tau)
\bigl(1-F_{\mathrm{other}}(\tau)\bigr)^{r-1}.
$$

这里的 $r$ 是预期概率排名，不是上下文长度。代码逐分量用正态近似；该假设和经验校准均不构成唯一性证明。零阈值仍为零，严格 `<0` 不会接受距离为零的候选；不会擅自加一个小数修补官方行为。若进入官方未定义的 `other std=0` 优化分支，明确报错。

## 必须披露的移植差异

1. Gemma NPU 的 `prefill_128` **不输出 logits**。选中候选后，额外用 decode 计算下一步 logits；丢弃这次 decode 的 KV，将原候选 KV 写入后续前缀。它保住了被验证的 KV，但不能声称 logits 等于官方同一次候选前向的输出。额外执行成本计入时间。
2. 概率排序直接稳定排序 float32 logits；官方先 softmax 再排序。理论顺序相同，浮点下的并列/下溢排序可能不同。
3. 原论文使用浮点模型；本移植使用编译后的 Gemma INT8 KV，并在 CPU 上反量化和计算 float32 距离。不能拿 Llama 阈值直接用于 Gemma，不声称不同设备 norm 逐位相同。
4. 当前受控目标 fixture 由连续 decode 采集。论文的整段 prefill 目标、真实 APK 捕获目标与它可能有计算路径差异。正式实验应对实际采集路径重新校准；不能把 fixture smoke 当作真实内存取证结果。
5. 支持显式公开前缀；示例为 BOS。深层恢复只使用自己的已恢复前缀，不加载目标的真实前缀 KV。无 BOS/logits 的空前缀初始 ID 搜索未在当前 NPU 接口开放。
6. `calibrate-cpa` 默认全候选，但校准文本长度取决于输入观测。全词表不等于全语境。保存全文 113 个文本 token 的配置，同时保留明确标注的 8 位置试验配置。

## 官方校准脚本的处理

固定版本 `get_collision_threshold.py` 的静态检查发现：距离构造时未切到当前候选位置，后续分析混淆 position/layer 维度，而且 `(K+V)/2` 与攻击端独立双分量阈值不一致。它也没有随仓库提供可直接使用的 Gemma 统计资产。

因此没有执行或照搬该数据组装脚本。我们的校准器对当前候选位置提取 K/V，按 `[position, candidate, component]` 组织距离。阈值计算则用 AST 单独加载官方纯函数，与移植函数交叉测试，无需运行官方攻击脚本。导出 `upstream_statistics.json` 可供官方 `get_collision_threshold(path, layer, n=8)` 直接检查；未校准层是 null，禁止套用其他层。

## 从独立校准文本到恢复

以下从工作区根目录运行。首次先构建 instrumentation native/APK，见 README。`inputs/cpa` 保存论文所用 Bitter Lesson 文本、固定 tokenizer 生成的 113 个原 ID、BOS 和来源；不添加 chat template 或外围引号。

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction

# 1. 已知明文目标，独立于待恢复密文/测试集。
"$PY" "$APP/run.py" fixture \
  --prefix-json "$APP/inputs/cpa/prefix.json" \
  --token-ids-json "$APP/inputs/cpa/full_tokens.json" \
  --output "$APP/fixtures/cpa_gemma4_bitter_full"

# 2. 113 个位置 × 262144 候选。开销大；按已完成位置断点续跑。
# 重跑此命令复用已完成位置，但拒绝模型、runtime、文本或算法版本改变。
"$PY" "$APP/run.py" calibrate-cpa \
  --observation "$APP/fixtures/cpa_gemma4_bitter_full/observation" \
  --known-tokens-json "$APP/inputs/cpa/full_tokens.json" \
  --rank 8 --output "$APP/assets/cpa_gemma4_bitter_full"

# 3. 在另一个观测上恢复。CPA 不使用 --epsilon；默认逻辑 batch 256。
"$PY" "$APP/run.py" run --method collision_plus_cpa \
  --calibration "$APP/assets/cpa_gemma4_bitter_full/calibration.json" \
  --observation /path/to/independent/observation \
  --output /path/to/new_cpa_result

# 4. 与原 guided 成对运行，重复时交替顺序。
"$PY" "$APP/run.py" compare --baseline collision_plus_cpa \
  --calibration "$APP/assets/cpa_gemma4_bitter_full/calibration.json" \
  --observation /path/to/independent/observation \
  --output /path/to/new_comparison --repeats 2
"$PY" "$APP/run.py" evaluate-comparison \
  --comparison /path/to/new_comparison --truth /path/to/private_eval/truth.json
```

`compare` 默认使用 CPA，必须提供独立 `--calibration`；guided 按自身配置的 LSB L2 阈值判定（浮点 KV 使用原始数值），两者不能称为统一验收标准的纯搜索速度对比。补充的统一阈值对照显式选择 `--baseline collision_plus`。报告 CPA 实验时必须同时报告准确率、失败数、在线时间和离线校准成本；框架不自动给出速度优势结论。

校准输出包含逐位置的全候选 K/V 距离、统计量、阈值、rank、候选范围、模型/manifest/runtime 哈希、检查点摘要。恢复只读取校准资产及待恢复观测，禁止打开 `private_eval` 和 `datasets`。直接把同一观测同时用于校准与测试会被拒绝；独立实验设计仍由研究者负责。

`--pilot-candidates` 的资产默认拒绝用于恢复，必须显式 `--allow-pilot-calibration` 才能作接口 smoke。这个开关不是将小样本提升为正式校准。

## 当前验证记录

测试 APK 哈希及 native_mode 共同绑定校准身份。当前 `optimized_v2` 已重新运行独立 8 位置 × 1024 候选校准：[当前 pilot](assets/cpa_gemma4_pilot_native_v2/calibration.json)，阈值与旧资产相同。它仍要求 `--allow-pilot-calibration`，不替代完整校准。之前的 `cpa_gemma4_pilot_socket` 和 `cpa_gemma4_pilot_native_optimized` 属于历史 APK，保留为历史记录。详见 [第二轮手机端优化](NATIVE_OPTIMIZATION_V2.md)。

实现及验证结果见 [VALIDATION.md](VALIDATION.md)。全文、全词表校准命令已具备；是否实际完成以资产覆盖记录为准，不能从命令存在推断实验已运行。
