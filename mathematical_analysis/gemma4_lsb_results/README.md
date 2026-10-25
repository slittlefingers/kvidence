# Gemma 4 E2B KV8：最终整数码的 LSB 容错界

日期：2026-10-17（Australia/Melbourne）。这是固定短前缀上的有限候选验证，不是整段恢复实验或 Pixel NPU 性能测试。

## 1. 直接结果

固定正确前缀 `The password is`（含 BOS 共 4 token），对 ` apple`、` window`、` secret`、` blue` 四个目标，分别观察第 10、15 层实际 KV 生产层。竞争候选为完整 **86,512** 个符合现有 E5 严格 ASCII 字母数字规则的 token；不是只比较查询预算内的 600 个，也不是完整 262,144 词表。

下表给出可保证的整数 LSB 半径下界。每一维都可以独立变化，不要求噪声随机或彼此抵消；整数码被限制在 `[0,255]`，各份向量的 scale/bias 固定。

| 层 | 目标 | 仅观测扰动：保证至少 | 仅所有候选扰动：保证至少 | 观测和所有候选各自扰动：保证至少 |
|---|---|---:|---:|---:|
| 10 | ` apple` | ±7 LSB | ±5 LSB | 各 ±2 LSB |
| 10 | ` window` | ±6 LSB | ±4 LSB | 各 ±2 LSB |
| 10 | ` secret` | ±12 LSB | ±9 LSB | 各 ±4 LSB |
| 10 | ` blue` | ±11 LSB | ±8 LSB | 各 ±4 LSB |
| 15 | ` apple` | ±15 LSB | ±9 LSB | 各 ±4 LSB |
| 15 | ` window` | ±14 LSB | ±8 LSB | 各 ±4 LSB |
| 15 | ` secret` | ±17 LSB | ±9 LSB | 各 ±4 LSB |
| 15 | ` blue` | ±16 LSB | ±10 LSB | 各 ±5 LSB |

所以在本次四个目标上：

- 最终码扰动仅发生在观测或候选一侧时，第 10、15 层均能保证 ±2、±3、±4 LSB 后最近邻仍正确。
- 两侧都允许每维独立 ±2 LSB，第 10、15 层四个目标均通过。
- 两侧各自 ±3 或 ±4 LSB，第 15 层四个目标均通过；第 10 层 ` apple`、` window` 未通过使用的保守 L2 界，另外两个通过。

**表中数字不是实际失败上限。** 超过该数字或充分条件未通过，不能据此宣称一定误判。仅观测一侧的列使用未裁剪对称扰动盒的保守半径；对指定的 1、2、3、4 LSB，另外做了裁剪整数盒上的精确差距检查。候选一侧和双侧两列使用 L2 三角不等式充分条件，不是最优容错半径。

## 2. 使用的模型、执行路径和距离

模型是此前 E5 固定的 `mlx-community/gemma-4-e2b-it-8bit`，revision `03dcf209f3f549b4075e7191e77cf69b3d48e1b2`。W8 affine 权重保持打包，15 个 KV 生产层使用真实 `QuantizedKVCache(bits=8, group_size=64)`；在 Apple MLX GPU 上执行。前缀远短于 512，未涉及窗口淘汰。

目标向量从完整 `prefix + target` 的 35 层 prefill 读取；候选向量从同一正确前缀状态分叉，batch 64 做单 token decode，运行至第 15 层，同时读取第 10、15 层。扫描前后核对前缀缓存的整数码、元数据和 offsets 的 SHA-256 一致。

每个位置 K+V 有 1,024 维，每组 64 维。保留实际 packed 码并解包为 uint8，以及原始 BF16 scale/bias。本文距离采用缓存载荷定义的精确 affine 值

$$
z_j=s_jq_j+\beta_j.
$$

这与 E5 默认 BF16 反量化后再转 FP32 的分析向量不同。校验解包时，将元数据精确提升到 FP32 后，独立对照 MLX 的反量化输出。本文证书不自动适用于默认 BF16 距离打分、APK 或其他运行时。

完整候选扫描约 27.33 秒；数值证书分析约 7 秒。它们是这次有限实验的墙钟记录，不能当作 Pixel NPU 的攻击时间。

## 3. 最终码扰动模型

每份向量都有自己的固定 scale/bias。允许每维码变化为

$$
q'_j=q_j+\delta q_j,\qquad
|\delta q_j|\le r,\qquad 0\le q'_j\le255.
$$

于是实际值变化为 $s_j\delta q_j$。每个向量每 1 LSB 的 L2 半径上界是

$$
\alpha=\sqrt{\sum_j s_j^2}.
$$

本次目标的 $\alpha$ 约为 0.515–0.629（第 10 层）和 0.520–0.554（第 15 层）；因此每维允许 ±4 LSB 对应的 L2 扰动上界约为 2.06–2.52 和 2.08–2.22，已不是单个标量步长。

本模型不改变权重、此前缓存、scale/bias、offset 或输入 token。没有在内部每个算子处加噪声，也没有将扰动缓存重新送入模型传播到下一个 token。这里讨论的是：固定实际读出的候选与目标载荷，允许它们最后的整数码发生有界变化后，最近邻决策是否保持正确。

## 4. 仅观测扰动的精确有限域判定

令目标为 $t$，正确候选为 $c$，任一错误候选为 $v$。原平方距离差为

$$
m_v=\|v-t\|_2^2-\|c-t\|_2^2.
$$

仅目标变化 $t\mapsto t+\delta$ 时，新的差距精确等于

$$
m_v(\delta)=m_v-2\sum_j(v_j-c_j)\delta_j.
$$

每维码独立受区间约束，因此最大内积可以逐维取端点得到。对一个给定整数 r，把每个错误候选的最坏差距都计算出来；若完整候选集中所有差距严格大于 0，就保证正确候选对全部允许扰动仍然唯一最近。

若忽略 `[0,255]` 裁剪，使用更大的对称扰动盒，有充分条件

$$
m_v>2r\sum_j|(v_j-c_j)s_j^{(t)}|\quad\text{对全部错误候选成立}.
$$

第一列的半径来自这个较保守条件。对 r=1、2、3、4 的 `target_only_exact_squared_gap_units` 则使用实际裁剪区间计算最坏差距，本次全部为正。

此推导没有随机噪声、独立分布或高维抵消假设。

## 5. 候选一侧和双侧的充分条件

设名义距离为 $d_v=\|v-t\|_2$、$d_c=\|c-t\|_2$；各份向量的单 LSB L2 上界分别为 $\alpha_v,\alpha_c,\alpha_t$。

只有所有候选允许扰动、目标固定时，由三角不等式：

$$
d'_v-d'_c\ge
(d_v-d_c)-r(\alpha_v+\alpha_c).
$$

因此右侧对每个错误候选都严格大于 0，即保证正确。

目标与所有候选各自允许扰动时，同理：

$$
d'_v-d'_c\ge
(d_v-d_c)-r(\alpha_v+\alpha_c+2\alpha_t).
$$

目标扰动影响两项距离，所以出现 $2\alpha_t$。本次实际遍历全部 86,511 个错误候选，使用每个候选自己的 scale 范数，而不是只检查名义上最近的一个候选。

双侧界是充分条件；为负不构成实际失败反例。若需要更紧的判定，可进一步优化双侧逐维整数扰动问题，本文尚未实现。

## 6. 如何避免把浮点近似误当严格证书

BF16 元数据是有限二进制有理数。对一个观察层，选择共同单位 $2^{-k}$，将所有实际 scale/bias 和 affine 向量精确转换为整数。本次第 10 层为 $k=20$，第 15 层为 $k=21$。

平方距离、内积、裁剪区间和最坏目标扰动差距均使用整数运算。脚本先计算覆盖所有 affine 端点的绝对值界 M，并检查

$$
8\times1024\times M^2<2^{63}-1,
$$

以防这些 int64 中间运算溢出。元数据乘共同尺度后还逐项检查确为整数。

L2 三角界中的平方根不用普通浮点近似来决定是否通过：对整数平方和使用 `math.isqrt` 构造向下/向上的整数界，错误候选距离向下取界，正确候选距离及所有扰动半径向上取界，只有保守差距严格为正才接受。

显示的十进制 L2 距离仅供阅读，不参与认证决定。另用 Python `Fraction` 独立重算了八个案例的名义最近错误候选平方距离，共 8,192 个坐标，全部与整数结果一致；对小型二维模型穷举了 400 组裁剪扰动检查，目标盒最坏差距公式全部一致。

这里的“证书”是指定有限数据及最终码扰动模型下的精确算术证明，没有形式化验证整个 Python/NumPy/MLX 软件栈，也没有证明模型内部所有执行差异都属于该扰动模型。

## 7. 文件与复现

| 文件 | 内容 |
|---|---|
| [metadata.json](metadata.json) | 模型、前缀、目标、执行路径、缓存校验和范围 |
| [results.json](results.json) | 八个案例的逐半径结论、整数界和保证半径 |
| [candidates.json](candidates.json) | 完整 86,512 个候选原 ID 与文本 |
| [targets.npz](targets.npz) | 四个目标在两层的 q/scale/bias |
| `l10_*.npy`、`l15_*.npy` | 全部候选的 q/scale/bias，保留以便独立复核 |
| [measure_lsb_tolerance.py](../measure_lsb_tolerance.py) | 收集与精确整数分析脚本 |

在工作区根目录执行，重新收集时使用新的目录以保留本轮证据：

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/mathematical_analysis/measure_lsb_tolerance.py \
  --output kvidence/mathematical_analysis/gemma4_lsb_results_replication
```

只重新分析已经保存的本轮载荷：

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/mathematical_analysis/measure_lsb_tolerance.py --analyze-only
```

本轮未改变现有 E5 结果、模型、tokenizer 或 Ghost APK。后续需要扩展到更多目标和前缀，再单独验证前缀状态误差、元数据扰动、内部量化误差传播以及 NPU。
