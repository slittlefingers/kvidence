# 从 Gemma 参数直接推出所有前缀的 KV 约束

日期：2026-10-17。对象为本地 Gemma 4 E2B，第 10、15 层独立 full-attention KV 生产层。本文给出不需要枚举具体前缀的参数约束与证明；它尚不是 token 恢复的全域保证。

## 1. 分析的参考与量词

使用固定模型系数、精确实数算子构成的参考执行。所有允许前缀、位置和 token 都包含在量词中；假设相关输入有限、mask 的有效 key 集合非空、RoPE 为当前配置对应的正交变换。

这里约束的是每个 head 的每条 K/V，不是将全部缓存拼接后的总 L2。量化码、BF16 元数据和浮点实现偏差尚未包含在下面的精确实数界内；实际缓存需要另外加入执行误差界，不能把参考界直接当作 NPU 认证。

“仅由参数”具体指使用网络结构、有效权重、归一化 epsilon、tokenizer/输入域和声明的执行语义；不能省略计算规则，只给出一组权重就宣称已认证硬件执行。

实现依据：[本地 Gemma text 计算图](../experiments/kv_guided_search_gemma4_int8/.venv/lib/python3.13/site-packages/mlx_lm/models/gemma4_text.py)、[模型配置](../experiments/kv_guided_search_gemma4_int8/model/config.json)。实际 norm 参数和精确平方界记录于 [prefix_state_parameter_bounds.json](prefix_state_parameter_bounds.json)。本轮只读取参数，没有运行新增模型输入。

## 2. 第一个可完整证明的约束：任意前缀的 V 都有统一范数界

本地每个 V head 使用无可学习缩放的 RMSNorm。设线性投影产生任意向量 t，维数 d、epsilon 为正，则

$$
v=\frac{t}{\sqrt{\|t\|_2^2/d+\varepsilon}}.
$$

无论 t 由什么 token、什么前缀和什么投影权重产生，

$$
\|v\|_2^2
=\frac{\|t\|_2^2}{\|t\|_2^2/d+\varepsilon}
=\frac{d\|t\|_2^2}{\|t\|_2^2+d\varepsilon}
\le d.
$$

所以

$$
\boxed{\|v\|_2\le\sqrt d.}
$$

第 10、15 层的每个 K/V head 均为 d=512，因此全部参考 V 条目均满足 $\|v\|_2\le\sqrt{512}\approx22.6274$。这是计算规则对任意输入的直接限制，不是从少量前缀估计出的最大值。

## 3. 直接使用实际缩放参数约束 Q、K 和 attention score

对于带有效缩放向量 gamma 的 head RMSNorm，同理有

$$
\|N_\gamma(t)\|_2\le\sqrt d\|\gamma\|_\infty.
$$

正交 RoPE 不改变范数。读取实际 q_norm/k_norm 权重后：

| 生产层 | Q 每 head 范数上界 | K 每 head 范数上界 | V 每 head 范数上界 |
|---|---:|---:|---:|
| 10 | 23.511300 | 1.358971 | 22.627417 |
| 15 | 23.157747 | 1.381068 | 22.627417 |

实际 q/k 缩放在各自 head 维度上是常量：第 10 层分别为 1.0390625、0.06005859375，第 15 层分别为 1.0234375、0.06103515625。即使并非常量，使用最大绝对值的界仍有效。

代码的 score scale 为 1。由 Cauchy–Schwarz，对任意当前位置与任意历史 key：

$$
|q^Tk_j|\le\|q\|_2\|k_j\|_2
\le d\|\gamma_Q\|_\infty\|\gamma_K\|_\infty.
$$

所以第 10、15 层有效位置的参考 attention score 绝对值分别不超过 **31.951171875** 和 **31.982421875**。mask 使用的负无穷不是有效 score，不纳入此实数界。

参数是精确 BF16 二进制有理数，范数平方和 score 的上述代数上界使用其精确值；表中开方后的十进制只是显示近似。

## 4. 从条目范围到前缀影响：attention 不能任意放大

一个 head 的输出为 $o=\sum_j a_jv_j$，其中 $a_j\ge0$、$\sum_j a_j=1$。于是对任意数量的有效历史条目，

$$
\|o\|_2\le\sum_j a_j\|v_j\|_2\le\sqrt{512}.
$$

前缀可以改变权重和 V 的方向，但不能让这个参考加权平均的范数超出该界。多 head 拼接、输出投影和后续残差需继续按实际参数处理；不能把单 head 的界直接当作整个层的界。

进一步，比较两种候选在同一前缀下的当前 attention 计算。定义 query 差 $\delta_Q$，所有对应 key 的最大差 $\delta_K$，所有对应 value 的最大差 $\delta_V$。有效位置集合相同。利用 softmax 的 $\ell_\infty\to\ell_1$ 一致界，可以证明

$$
\|o_u-o_v\|_2
\le\delta_V+M_V(M_K\delta_Q+M_Q\delta_K).
$$

证明见 [length_uniform_proof.md 的 L1/L2](length_uniform_proof.md)。该式对参考运算成立，不含历史条目数的直接乘因子。这里给出实际参数代入值：

$$
\boxed{\|o_u-o_v\|_2\le\delta_V+30.75\delta_Q+532\delta_K\quad(m=10),}
$$

$$
\boxed{\|o_u-o_v\|_2\le\delta_V+31.25\delta_Q+524\delta_K\quad(m=15).}
$$

这些是上界，可能很保守。相同参考前缀的旧 K/V 对两种候选相同，但当前 token 自身的 K/V 通常不同，也参与 causal attention，不能遗漏其 $\delta_K,\delta_V$。若比较的是不同数值路径，则旧缓存的偏差也必须计入这些最大差。

## 5. 为什么这还没有证明 token 可恢复

上述结论证明所有前缀都受到同一组参数约束；证明并不需要计算每个具体前缀。然而，两个不同候选都落在同一个有界区域内，仍可能很接近或重合。

接下来需要结合允许 embedding/PLE 表行和模型参数，保留共同前缀关系，认证候选差异或实际匹配差距的正下界。范围约束提供输入域，但不自动提供这个下界。已有全局范数分析很松，不能把“可以得出有限上界”当成“上界足够小”。

Gemma 还包含 head/output 投影、GELU 门控、RMSNorm、PLE 与 layer scalar。每个参考算子的域和敏感度要正确传播；数值执行误差也需按实际路径建模。有限候选可用其差向量几何收紧分析，但不能只看主 embedding 的两两距离就忽略 PLE 或上下文。

因此，目前可明确报告：**已经从实际模型规则和参数，证明了参考 KV 条目与单 head attention 的全前缀范围约束；尚未证明实际数值执行下所有允许 token 都能稳健区分。**
