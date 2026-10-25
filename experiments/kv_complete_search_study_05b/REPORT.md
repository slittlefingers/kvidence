# KVidence 找全与可信停止：完整研究报告

## 核心结论

**局部界有明显排除潜力，但本轮没有获得一个既大幅减少查询、又能对真实深层 KV 保证找全的算法。** 这个结论经过旧案例诊断、冻结后新目标检验、独立审计以及停止条件证明。

1. 按预测距离分组并使用真实最坏单侧误差，明显优于全域统一界；这证明局部化值得关注，但构造这个 oracle 界已经需要完整答案。
2. 用少量额外校准样本替代真实最坏误差，能很快给出一个集合，但实际出现遗漏；不能据此宣布“已经找全”。
3. 独立随机审计加完整回退具有可证明的单次漏解概率上界，不过查询代价接近 10,000 候选穷举。
4. 已证明的黑盒查询障碍不适用于所有已知参数模型。利用实际网络结构构造足够紧、低成本的局部证书，仍是尚未解决的部分。

原 KVidence 方法、参数、旧查询轨迹和数据未修改。新增代码与所有结果均在本目录。

## 1. 研究问题与设计

目标是在固定正确前缀、模型、有限候选词表和距离阈值下，返回全部相容 token。这包含命中目标之外的其他匹配候选；多个相容候选并不等于能够唯一认定原输入。
模型为 Qwen2.5-0.5B，10,000 候选，FP32 KV，第 12/24 层，四个短前缀。比较原线性代理与已冻结的线性 + RBF 代理，两者各自原查询轨迹不变。主检查点为 600 个查询。
旧组：64 个目标/层案例，含此前 development 和 confirmation。新目标：另选 16 个 token、两层共 32 例，其实际词表 ID 与之前所有目标不重叠。新目标协议在生成目标和运行之前固定。新目标仍共享模型、候选池和前缀，不能称为跨模型或新语料验证。
所有距离阈值 ε=0.10/0.25 均为归一化最大坐标差。没有把它们解释为 INT8 的 ±4 LSB。全部新实验都是对现有真实模型 KV 表的计数 oracle 回放；没有把回放耗时当作实际 NPU 推理耗时。

## 2. 已知真实最坏误差时，局部化有没有用？

下表是第 12 层、ε=0.25、已查询 600 个候选后，平均还需要实际检查多少候选。统一单侧界也报告在内，以区分“只管过预测”与“分组”各自带来的变化。

| 目标组 | 方法 | 统一对称界 | 统一单侧界 | 距离 8 组 | 距离 32 组 | 距离 128 组 | embedding 32 组 |
|---|---|---:|---:|---:|---:|---:|---:|
| 旧 confirmation | 原线性代理 | 9111.4 | 9091.5 | 2218.2 | 537.7 | 119.8 | 7635.0 |
| 旧 confirmation | 线性 + RBF | 9372.8 | 8203.1 | 1309.7 | 333.4 | 85.9 | 4842.2 |
| 新目标 | 原线性代理 | 9211.7 | 9196.2 | 3611.1 | 1924.8 | 852.2 | 7503.1 |
| 新目标 | 线性 + RBF | 9365.5 | 8181.1 | 2709.9 | 1637.8 | 743.2 | 4713.0 |

这些数是 oracle 诊断，不是可部署查询预算。分组由可用特征决定，但组内最坏误差仍由完整答案决定。新目标更难，不能只引用旧组较好的数字：新目标第 12 层在 ε=0.25 下的相容集合大小可达 192。
第 24 层在本次局部 oracle 界下明显更容易；新目标 ε=0.25 时真实集合大小为 1–2。这不支持“所有深层都具有同样难度”的结论。

一个理论解释是：若组内预测距离为同一常数且高于阈值，精确单侧残差界是否能排除整组，等价于该组最小真实距离是否超过阈值。因此最紧的 oracle 界本身可能已经包含“组内有没有解”的答案。详细恒等式见 [THEORY.md 第 3.1 节](THEORY.md#31-为什么分组有潜力)。

## 3. 不看完整答案，改用校准样本，会怎样？

固定 600 点代理，再从未查询候选中随机抽取额外 600 个校准点。以组内最大观测过预测残差估计半径，查询全部未被排除的候选并返回已验证匹配。下表每行是 16 个目标 × 5 次校准随机重复，共 80 次运行。重复共享目标，不是 80 个独立语料样本。

| 目标组 | 方法 | 分组数 | 平均总查询 | 找全运行 | 出现遗漏的目标数 | 漏项总数（含重复） |
|---|---|---:|---:|---:|---:|---:|
| 旧 confirmation | 原线性代理 | 1 | 8877.2 | 79/80 | 1/16 | 1 |
| 旧 confirmation | 原线性代理 | 32 | 1233.8 | 55/80 | 5/16 | 125 |
| 旧 confirmation | 线性 + RBF | 1 | 6631.7 | 80/80 | 0/16 | 0 |
| 旧 confirmation | 线性 + RBF | 32 | 1225.0 | 60/80 | 4/16 | 95 |
| 新目标 | 原线性代理 | 1 | 8814.2 | 66/80 | 4/16 | 25 |
| 新目标 | 原线性代理 | 32 | 1475.2 | 41/80 | 8/16 | 1020 |
| 新目标 | 线性 + RBF | 1 | 6668.3 | 62/80 | 6/16 | 27 |
| 新目标 | 线性 + RBF | 32 | 1449.5 | 36/80 | 9/16 | 870 |

局部经验半径过窄会制造“已经全部处理”的假象。它能大幅减少查询，但遗漏真实匹配候选。样本中偶尔全部找全也不能推出全候选覆盖保证。

### 额外校准量的预设敏感性分析：新目标、多解主组

| 方法 | 校准样本 | 分组数 | 平均总查询 | 找全运行 | 漏项总数（含重复） |
|---|---:|---:|---:|---:|---:|
| 原线性代理 | 200 | 1 | 7708.9 | 62/80 | 62 |
| 原线性代理 | 200 | 32 | 936.3 | 41/80 | 1211 |
| 原线性代理 | 600 | 1 | 8814.2 | 66/80 | 25 |
| 原线性代理 | 600 | 32 | 1475.2 | 41/80 | 1020 |
| 原线性代理 | 1200 | 1 | 9274.2 | 68/80 | 18 |
| 原线性代理 | 1200 | 32 | 2252.0 | 41/80 | 790 |
| 线性 + RBF | 200 | 1 | 5033.1 | 56/80 | 82 |
| 线性 + RBF | 200 | 32 | 909.7 | 36/80 | 1055 |
| 线性 + RBF | 600 | 1 | 6668.3 | 62/80 | 27 |
| 线性 + RBF | 600 | 32 | 1449.5 | 36/80 | 870 |
| 线性 + RBF | 1200 | 1 | 7455.4 | 73/80 | 9 |
| 线性 + RBF | 1200 | 32 | 2171.5 | 37/80 | 676 |

## 4. 加上有证明的审计之后，代价是多少？

对经验方案拟排除的集合做一次独立无放回审计。若发现匹配就补查全部剩余项；否则停止。这是单次运行至少 1−δ 正确的程序，不是绝对零失败，也不是观察到零命中后的后验概率。

| 目标组 | 方法 | δ | 平均总查询 | 找全运行 | 触发完整回退 |
|---|---|---:|---:|---:|---:|
| 旧 confirmation | 原线性代理 | 0.05 | 9692.2 | 79/80 | 24/80 |
| 旧 confirmation | 原线性代理 | 0.01 | 9939.6 | 80/80 | 25/80 |
| 旧 confirmation | 线性 + RBF | 0.05 | 9670.1 | 80/80 | 20/80 |
| 旧 confirmation | 线性 + RBF | 0.01 | 9934.1 | 80/80 | 20/80 |
| 新目标 | 原线性代理 | 0.05 | 9775.0 | 80/80 | 39/80 |
| 新目标 | 原线性代理 | 0.01 | 9955.1 | 80/80 | 39/80 |
| 新目标 | 线性 + RBF | 0.05 | 9802.2 | 80/80 | 44/80 |
| 新目标 | 线性 + RBF | 0.01 | 9960.5 | 80/80 | 44/80 |

旧组存在审计后仍漏解的运行，已原样保留。它们并不违反概率保证；δ>0 本来就允许小概率失败。新目标表中没有出现失败也不意味着升级为确定性保证。
汇总所有预设配置，而不只看主表，旧组共有 12 次、新目标共有 9 次审计后遗漏；原始记录均保留。
基准使用固定伪随机种子复现样本；概率定理针对排除提案固定之后的独立均匀审计抽样，不把某个固定种子的确定结果解释成随机保证。实际使用须满足该随机源条件。
若要求确定性零遗漏，补查全部被排除候选，总查询量为 10,000。该精确回退在每个经验运行上都执行并核对。

## 5. 已完成的理论与没有完成的理论

| 命题 | 状态 | 适用范围 |
|---|---|---|
| 有效下界排除，补查其余候选，必然找全 | 已证明并测试 | 固定前缀、有限词表、确定性查询分类 |
| 局部单侧残差界可用于安全排除 | 已证明 | 必须已有有效的组内误差上界 |
| 用认证的模型/代理 Lipschitz 常数与已查锚点构造局部界 | 条件性公式已证明 | 真实模型所需紧常数未取得 |
| 独立审计 + 完整回退，漏解概率至多 δ | 已证明、组合枚举核对并运行 | 残余标签固定、独立单次审计 |
| 无结构黑盒、只返回验证匹配时的最坏期望查询下界 | 已证明 | 不是固定已知神经网络上的不可能性结论 |
| 少量校准样本自动提供全候选误差界 | 不成立 | 本轮存在明确的实际反例 |
| 实际深层 KV 可低成本认证找全并大幅加速 | 未证明 | 需要紧的结构证书或进一步明确假设 |

完整证明、式 (7a) 的 oracle 解释、有限总体概率和单 token/全文边界见 [THEORY.md](THEORY.md)。不能把 LinFACT 的置信半径直接套到任意非线性 KV 代理；其 misspecification 分析有明确偏差与间隔条件。[原论文第 4–5 节](https://arxiv.org/html/2510.00073v1)

## 6. 一个可以逐项核对的失败实例

已单独保存[新目标中的第一个经验漏解案例](EXAMPLE.md)。它列出真实匹配项的距离、预测距离、校准半径与错误下界。这能直接看到：实际目标早已命中，其他符合阈值的 token 仍被错误排除；问题在误差界的有效性，而不是是否能找到一个好候选。

## 7. 审计、数据范围与复现

- 独立重新拟合 576 个检查点，并与冻结预测/原排序核对。
- 独立检查 18432 组局部 oracle 决策、11520 次经验运行、23040 次随机审计。
- oracle 有效界没有错误排除；经验方案和概率审计中的实际遗漏全部保留，不筛掉失败运行。
- 小有限总体上枚举全部审计子集，与超几何组合概率逐项核对；单个隐藏匹配项的 5% 最坏情形也有精确测试。
- 原始源码、数据、旧查询轨迹 SHA256 不变；新目标按真实 token ID 与旧目标去重。
- 现有候选池只覆盖特定英语/数字子集；实验不构成全词表、未知前缀、长文本、Gemma INT8 或 Pixel NPU 的保证。
- 本轮没有评估神经网络形式化验证器的实际成本；条件性 Lipschitz 构造未被标为已经运行成功。

完整原始结果与所有预设配置：

- [研究协议](README.md)、[新目标协议](NEW_TARGET_PROTOCOL.md)
- [全部汇总，包括 development、两个层、两个阈值和所有预设配置](results/summary.json)
- [旧案例局部界](results/local.json)、[旧案例经验方案与审计](results/empirical.json)
- [新目标与完整搜索轨迹](fresh_results/cases.json)、[新目标局部界](fresh_results/local.json)、[新目标经验方案与审计](fresh_results/empirical.json)
- [旧案例独立审计](results/validation.json)、[新目标独立审计](fresh_results/validation.json)

追加新目标与报告的复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 kvidence/experiments/kv_inverse_tiny/.venv/bin/python kvidence/experiments/kv_complete_search_study_05b/fresh_targets.py
PYTHONDONTWRITEBYTECODE=1 kvidence/experiments/kv_inverse_tiny/.venv/bin/python kvidence/experiments/kv_complete_search_study_05b/validate.py --fresh
PYTHONDONTWRITEBYTECODE=1 kvidence/experiments/kv_inverse_tiny/.venv/bin/python kvidence/experiments/kv_complete_search_study_05b/report.py
```

## 8. 本轮研究最终判断

本轮完成的是一条可复现的研究链：统一界诊断 → 局部单侧界诊断 → 无答案校准对照 → 新目标检验 → 带证明的随机审计与精确回退 → 黑盒限制与结构条件。
**目前不能声称 KVidence 已经获得“少量查询即可保证找全”的方法。** 已经得到的正面结果是局部化显著提高 oracle 排除能力；已经确认的缺口是如何低成本获得有效的局部误差证书。少量校准最大值不能填补这个缺口；无结构的随机审计可以提供概率保证，但节省很有限。
这项研究支持将“候选发现效率”和“完整集合证书成本”分开评价，保留原检索方法作为基线。若论文需要宣称实际模型上的可靠加速，仍须补充由网络结构支持的证书；不能把本轮条件定理、oracle 数字或未出现失败的测试代替该证明。

## English abstract

We evaluate complete compatible-token recovery from deep KV observations separately from target-token retrieval. Using frozen Qwen2.5-0.5B search trajectories and additional disjoint token targets, we find that prediction-distance partitions with exact one-sided residual envelopes can substantially reduce the unresolved candidate set. These privileged envelopes use the full answer table. Replacing them with finite-sample calibration maxima produces actual omissions, including after the original target has been recovered. We derive and implement a one-shot finite-population audit with exhaustive fallback, providing a per-run omission probability of at most delta without assuming proxy correctness. The singleton-omission case forces near-exhaustive audit costs. We provide conditional structural certificate formulas and a restricted black-box query lower bound, without claiming an impossibility result for fixed known neural networks. Efficient, valid local certificates for the actual deep-KV mapping remain unresolved.
