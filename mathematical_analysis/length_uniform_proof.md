# 正确前缀下 KV 数值误差的长度一致上界

日期：2026-10-17（Australia/Melbourne）。状态：包含完整条件性证明的数学草稿；没有认证 Gemma 4 的全输入分离间隔，也没有认证 Pixel NPU 的算子误差。本文件把“固定样本成功”提升为对所声明计算模型、全部允许序列长度成立的定理，而非由 128-token 数据外推。

## 1. 核心结论与边界

对固定深度、token 始终相同的 decoder Transformer，在下列条件成立时，可对所有允许上下文长度 N 给出同一个逐位置误差上界：

$$
\max_{i\le N}\|\widehat z_{m,i}-z_{m,i}\|_2
\le B_m,
$$

其中 m 是 KV 生产层；$B_m$ 由网络层数、权重、归一化、量化和局部执行误差界决定，**不需要额外乘 N，也不需要按 token 次数乘 $A^N$**。

该结论是逐位置最大误差的上界，不是“误差为零”“实际误差不随位置变化”或“误差界一定足够小”。若把全部历史条目拼成一个向量，总 L2 仍可随 $\sqrt N$ 增长，这与逐位置上界不矛盾。

如果再证明整个允许输入域上的候选分离间隔大于总扰动的两倍，并保证真候选被查询，则可用归纳法保证整段精确恢复。**长度一致误差界与统一候选间隔是两个需要分别证明的命题。**

## 2. 计算模型与假设

比较两次执行：可以是理想实数参考与量化执行，也可以是相同固定权重模型的两条有局部误差的执行路径。将共同算子视为实数函数，将其数值实现偏差单独计入。

1. **相同 token 与位置。** 每个位置输入 ID、模板、位置编码和 attention mask 一致。恢复算法在“此前全部恢复正确”的归纳路径上满足这一点。
2. **固定有限深度。** 当前 token 的初始表示来自 token embedding/PLE 等输入，上一位置的最后隐藏向量不直接成为本位置的初始隐藏向量；历史通过 KV 参与 attention。
3. **缓存单次写入。** 一个条目生成时量化或加入有界误差；之后直接复用，不在每个未来步骤反复给该旧条目增加误差。滑动淘汰可以发生，但双方保留相同有效位置和 mask。
4. **局部误差有统一界。** 每次新写入的 K/V 以及每个局部算子的误差，在声明的允许长度范围内有统一界。误差可以相关、同向或对抗性，不要求独立或均值为零。
5. **Q/K/V 范数有界。** 每个 head 的实际 Q、K、V 有统一范数上界，包括所考虑的量化偏差。
6. **点算子有界敏感度。** RMSNorm、线性层、dense FFN、PLE 门控等在所需有界区域内具有有限 Lipschitz 常数；固定权重、epsilon 为正，并正确纳入残差及缩放。

下面先证明精确 attention 与有界 KV/局部扰动的模型。真实 GPU/NPU 的浮点 attention 归约误差必须计入第 4 条；若这个算子误差界随长度变化，就不能省略该依赖。可以在指定最大上下文 $N_{\max}$ 内取统一界，但不能由此声称任意无限长度硬件执行的同一误差界。

**不能由 dtype 自动获得第 4 条。** “INT8”或“BF16”本身不提供每个真实算子在所有上下文中的误差认证。物理地址误认、metadata 损坏、offset 错误和不同 mask 也不在这些假设内。

## 3. 引理一：softmax 的一个不含长度因子的界

对任意维度 n 的实数向量 a、b：

$$
\boxed{
\|\operatorname{softmax}(a)-\operatorname{softmax}(b)\|_1
\le\|a-b\|_\infty.
}
\tag{L1}
$$

**证明。** 设 $p=\operatorname{softmax}(a)$，Jacobian 为 $J=\operatorname{diag}(p)-pp^T$。任意方向 u 满足

$$
\|Ju\|_1
=\sum_jp_j|u_j-\mu|,
\qquad \mu=\sum_jp_ju_j.
$$

由 Cauchy–Schwarz：

$$
\sum_jp_j|u_j-\mu|
\le\sqrt{\sum_jp_j(u_j-\mu)^2}
\le\sqrt{\sum_jp_ju_j^2}
\le\|u\|_\infty.
$$

沿 a 到 b 的线段积分 Jacobian，得到式 (L1)。证毕。

该常数与 n 无关。mask 中的 $-\infty$ 不作为实数参与积分：先限制到双方相同的有效 key 集合，再应用此引理。

## 4. 引理二：单 head attention 的长度一致误差界

对某个位置，定义

$$
p_j=\operatorname{softmax}_j(\tau q^Tk_j),\qquad
o=\sum_jp_jv_j,
$$

另一执行对应 $\widehat q,\widehat k_j,\widehat v_j,\widehat p_j,\widehat o$。$\tau$ 是固定 score 缩放；标准 attention 为 $1/\sqrt{d_k}$，Gemma 实现应使用其实际缩放。

假设

$$
\begin{aligned}
\|\widehat q-q\|_2&\le\delta_q,&\quad\|q\|_2&\le M_Q,\\
\max_j\|\widehat k_j-k_j\|_2&\le\delta_k,&\quad\max_j\|\widehat k_j\|_2&\le M_K,\\
\max_j\|\widehat v_j-v_j\|_2&\le\delta_v,&\quad\max_j\|v_j\|_2&\le M_V.
\end{aligned}
$$

则

$$
\boxed{
\|\widehat o-o\|_2
\le\delta_v+M_V|\tau|(M_K\delta_q+M_Q\delta_k).
}
\tag{L2}
$$

**证明。** 每个 logit 的差满足

$$
|\tau\widehat q^T\widehat k_j-\tau q^Tk_j|
\le|\tau|(M_K\delta_q+M_Q\delta_k).
$$

又有

$$
\widehat o-o
=\sum_j\widehat p_j(\widehat v_j-v_j)
+\sum_j(\widehat p_j-p_j)v_j.
$$

第一项范数至多 $\delta_v$，因为 $\widehat p_j\ge0$ 且和为 1。第二项至多 $M_V\|\widehat p-p\|_1$，再应用式 (L1)。证毕。

这一步说明历史条目数量不必成为误差的线性乘数，但不能忽略 Key 误差改变 attention 权重的影响。若有 H 个 heads，可对每个 head 应用该界，再乘拼接因子 $\sqrt H$ 和输出投影算子范数；H 是固定网络参数。

标准 dot-product attention 在无界输入域上并不具有全局 Lipschitz 性，见 [Kim 等，The Lipschitz Constant of Self-Attention](https://arxiv.org/abs/2006.04710)。本文明确限制 Q/K/V 范数，并采用逐行最大 L2 范数，因此不与该结果矛盾。

## 5. 为什么 Gemma 的结构能提供范数与局部敏感度界

RMSNorm 定义为

$$
R(x)=\gamma\odot\frac{x}{\sqrt{\|x\|_2^2/d+\epsilon}},\qquad\epsilon>0.
$$

直接得到

$$
\|R(x)\|_2\le\sqrt d\|\gamma\|_\infty,
\qquad
\operatorname{Lip}(R)\le\frac{\|\gamma\|_\infty}{\sqrt\epsilon}.
\tag{L3}
$$

**证明。** 第一个式子来自分母至少为 $\|x\|_2/\sqrt d$。无 gamma 的 Jacobian 是

$$
a^{-1}I-\frac{xx^T}{d\,a^3},
\qquad a=\sqrt{\|x\|_2^2/d+\epsilon}.
$$

其横向特征值为 $a^{-1}$，径向特征值为 $\epsilon/a^3$，均不超过 $1/\sqrt\epsilon$；再乘 gamma 的对角算子范数。证毕。

本地 E2B 实现对 Q/K 使用 head RMSNorm，V 使用无 gamma 的 RMSNorm，RoPE 为有界旋转。因此实数参考中 Q/K/V 的范数有与序列长度无关的界；量化存储值的偏差另加到实际界里。

Gemma dense block 的 attention、FFN、PLE 分支输出还有归一化和残差。对固定 token embedding 有界输入，隐藏状态的范数可按层数控制，例如对相应后归一化分支：

$$
H_\ell\le |s_\ell|
\left(H_{\ell-1}+M_{\mathrm{postA},\ell}
+M_{\mathrm{postF},\ell}+M_{\mathrm{postPLE},\ell}\right).
$$

有限域上的线性映射、GELU 和乘法门控均有有限敏感度，残差使用 $1+L$，层缩放使用 $|s_\ell|$。这些常数可由固定权重与声明域推导，不能用几次 Jacobian 采样代替其 supremum。

本地配置 `rms_norm_eps=1e-6`，式 (L3) 的全局界可能很大；**常数有限不意味着所得恢复证书已经足够紧。** 本文件没有将这些参数组合成通过了 Gemma 全输入认证的具体数值。

代码证据：[Gemma 4 text 实现](../experiments/kv_guided_search_gemma4_int8/.venv/lib/python3.13/site-packages/mlx_lm/models/gemma4_text.py)，尤其 K/V 在当前层 attention 之前生成，以及各分支的 RMSNorm/残差。

## 6. 主定理：按层递推，而非按 token 次数递推

令 $h_{\ell,i}$ 为第 i 个位置经过第 $\ell$ 层后的隐藏向量，定义逐行误差

$$
E_\ell(N)=\max_{i\le N}\|\widehat h_{\ell,i}-h_{\ell,i}\|_2,
\qquad
U_\ell(N)=\max_{0\le r\le\ell}E_r(N).
\tag{L4}
$$

### 6.1 当前 K/V 来自更低一层的当前隐藏表示

对独立 KV 生产层 $\ell$，当前 K/V 由 $h_{\ell-1,i}$ 投影和归一化产生，而不是由历史 $K_{\ell,<i}$ 或 $V_{\ell,<i}$ 递归产生。把局部存储误差上界写作 $\eta_{K,\ell},\eta_{V,\ell}$，则

$$
\begin{aligned}
\delta_{q,\ell}&\le a_{q,\ell}U_{\ell-1}+\eta_{q,\ell},\\
\delta_{k,\ell}&\le a_{k,\ell}U_{\ell-1}+\eta_{K,\ell},\\
\delta_{v,\ell}&\le a_{v,\ell}U_{\ell-1}+\eta_{V,\ell}.
\end{aligned}
\tag{L5}
$$

这些界对全部历史位置同时成立，因为使用的是位置最大值。旧条目只写一次，复用不会自行再增加一份存储误差。

若该层复用较早生产层的 KV，其源隐藏状态层数仍小于当前层，使用 $U_{\ell-1}$ 以及该生产层的投影和误差常数即可。

### 6.2 归纳得到长度一致上界

将式 (L5) 代入式 (L2)，再依次通过输出投影、后归一化、残差、FFN 和 PLE，可得到常数 $C_\ell,D_\ell\ge0$，使

$$
E_\ell(N)\le C_\ell U_{\ell-1}(N)+D_\ell.
$$

下面给出这些常数的一种具体组合方式，避免只用“存在常数”替代推导。对各 head 使用共同最大界，令

$$
\begin{aligned}
t_\ell&=\sqrt H\|W_{O,\ell}\|_{\mathrm{op}}
\bigl[a_{v,\ell}+M_V|\tau|(M_Ka_{q,\ell}+M_Qa_{k,\ell})\bigr],\\
n_\ell&=\sqrt H\|W_{O,\ell}\|_{\mathrm{op}}
\bigl[\eta_{V,\ell}+M_V|\tau|(M_K\eta_{q,\ell}+M_Q\eta_{K,\ell})\bigr]
+\epsilon_{\mathrm{attn},\ell}.
\end{aligned}
$$

其中 $\epsilon_{\mathrm{attn},\ell}$ 包含注意力归约和输出投影的局部执行误差，位于输出投影后的同一单位。令 $L_{A,\ell}$ 为 attention 后归一化的 Lipschitz 界，定义

$$
R_\ell=1+L_{A,\ell}t_\ell,\qquad
Z_\ell=L_{A,\ell}n_\ell+\epsilon_{A,\ell}.
$$

FFN 分支从输入到后归一化输出的 Lipschitz 界为 $f_\ell$，其局部误差界为 $\epsilon_{F,\ell}$；PLE 分支相对于当前隐藏向量的 Lipschitz 界为 $p_\ell$，误差为 $\epsilon_{P,\ell}$。PLE 的输入来自原 embedding：若两边有输入差异，把它的已知统一上界经辅助分支敏感度后计入 $\epsilon_{P,\ell}$；没有额外跨 token 状态。

则可选

$$
\begin{aligned}
C_\ell&=|s_\ell|(1+p_\ell)(1+f_\ell)R_\ell,\\
D_\ell&=|s_\ell|
\bigl[(1+p_\ell)((1+f_\ell)Z_\ell+\epsilon_{F,\ell})
+\epsilon_{P,\ell}\bigr]+\epsilon_{s,\ell}.
\end{aligned}
$$

这分别保留了 attention、FFN、PLE 三处残差的增益和层缩放；没有 PLE 时设 $p_\ell=\epsilon_{P,\ell}=0$。这些仍可能是很松的上界，但所有项都有明确来源。

这里 $D_\ell$ 汇总该层及其 KV 来源的局部误差，并包含实际 attention 算子的数值误差界；对声明的长度范围必须统一。令 $A_\ell=\max(1,C_\ell)$，则

$$
U_\ell(N)\le A_\ell U_{\ell-1}(N)+D_\ell.
\tag{L6}
$$

**定理（长度一致数值稳定性）。** 第 2 节假设成立，且上述常数对允许长度统一时，有

$$
\boxed{
U_L(N)\le
\left(\prod_{\ell=1}^{L}A_\ell\right)U_0
+\sum_{r=1}^{L}
\left(\prod_{\ell=r+1}^{L}A_\ell\right)D_r
=:\mathcal B_L,
}
\tag{L7}
$$

对允许域中的每个 N 均成立。

**证明。** 输入误差由统一 $U_0$ 控制。假设第 $\ell-1$ 层的界对全部位置成立，则同层历史 cache 的 K/V 来源均为该层或更早生产层的低层隐藏表示，式 (L5) 同时控制全部可见历史条目。式 (L2) 不含可见条目数因子，故本层各位置满足同一 $C_\ell U_{\ell-1}+D_\ell$。残差与点算子按局部界组合得到式 (L6)。对层数归纳、展开这 L 次递推，得到式 (L7)。证毕。

观察独立第 m 层 K+V 时，只需其输入隐藏层的界：

$$
B_m\le
\sqrt{
(a_{k,m}\mathcal B_{m-1}+\eta_{K,m})^2+
(a_{v,m}\mathcal B_{m-1}+\eta_{V,m})^2
},
\tag{L8}
$$

单 head 情况如上；多 KV heads 应按对应范数组合。Gemma E2B 的第 15 层 KV 生产不需要后面共享层的隐藏误差界，因为后面层不会反向改写已生成的独立 KV。

若所有局部误差界都写成某个统一误差尺度 $\varepsilon$ 的固定倍数，即 $U_0\le d_0\varepsilon$、$D_r\le d_r\varepsilon$，并在固定允许扰动范围 $\varepsilon\le\varepsilon_{\max}$ 上选取统一的范数/增益常数，那么式 (L7)、(L8) 可简写为

$$
\boxed{
\sup_{i\le N}\|\widehat{\mathrm{KV}}_{m,i}-\mathrm{KV}_{m,i}\|_2
\le\mathcal C_m\varepsilon,
}
$$

其中 $\mathcal C_m$ 与 N 无关。这里 $\varepsilon$ 应包含正常量化和执行误差，不能把 r=0 的“没有额外注入”误写成所有数值误差都为零。

**解释。** 这是真正可跨长度推广的推导：误差可能随位置变化并逐步接近上界，但不必按 10,000 个 token 重复乘 10,000 次增益。有限深度的下层到上层依赖与 attention 的归一化权重共同给出了更合适的控制量。

## 7. 一个无需估计增益的结构性推论

在相同 token、相同位置和相同确定性执行规则下，如果只修改独立生产层 m 的旧 K/V，低于 m 的缓存保持原样，且不存在额外跨层反向状态，则以后新生成的第 $1,\ldots,m$ 层 KV 不会因此改变。

**证明。** 每个新 token 的 embedding 相同；第 1 层新 KV 仅由该位置输入产生。依次向上归纳，生成第 m 层新 KV 前的全部运算只使用更低层隐藏及其缓存，而这些没有接收到第 m 层旧缓存的变化。第 m 层旧缓存会影响本层 attention 输出，从而影响更高层，却不进入本层已经先行生成的新 K/V。证毕。

这不是说第 m 层误差“完全没有影响”：原错误条目还在，attention 输出和 logits 仍可能改变。如果后续 token 由这些 logits 自主生成，token ID 变化会破坏相同输入条件。若较低层也有扰动，则会影响后续较高层的新 KV，正如顺序实验所测。

## 8. 把每次 ±r LSB 放入统一局部误差界

对理想 min–max affine 量化、每组 g 维、b 位、精确元数据及最近整数舍入，设每个码额外变化至多 r（允许裁剪），有

$$
\boxed{
\|Q_r(z)-z\|_2
\le\theta_r\|z\|_2,
\qquad
\theta_r=\frac{(r+1/2)\sqrt{2g}}{2^b-1}.
}
\tag{L9}
$$

**证明。** 组 G 的步长 $s_G=(\max z_G-\min z_G)/(2^b-1)$，每维误差不超过 $(r+1/2)s_G$。组内范围满足

$$
(\max z_G-\min z_G)^2\le2\|z_G\|_2^2.
$$

于是总平方误差不超过

$$
(r+1/2)^2\sum_G g s_G^2
\le\frac{2g(r+1/2)^2}{(2^b-1)^2}\|z\|_2^2.
$$

开方即得结论。常数组范围为零时，定义其精确重建为该常数，不影响证明。证毕。

g=64、b=8 时，$\theta_0\approx0.0222$、$\theta_2\approx0.1109$、$\theta_4\approx0.1997$。这是相对于量化前该向量范数的保守界，不是实际 Gemma 误差率，也不是 token 误判概率。

比较两条执行时，每条的量化误差预算分别加入。例如受害者仅正常舍入、攻击者额外有 ±r 码误差，且各自量化前 K 范数至多 M，则可用 $(\theta_0+\theta_r)M$ 控制两边相对于平滑 K 投影的局部误差项。该量与写入了多少历史条目无关。

实际 BF16 scale/bias、非理想量化和 attention 算术需另加元数据及执行误差项，不能直接使用理想式 (L9) 宣称实机认证。MLX 的组内量化定义和最后维分组规则见 [官方 quantize 文档](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)。

## 9. 整段 token 精确恢复的推论

选定参考模型及有效输入域，令 $\Delta_m^*$ 为该域所有允许正确前缀、当前位置和错误候选上的 K+V 分离间隔下确界（有限域时为最小值）。设目标读出误差至多 $\eta$、候选相对于同一参考的误差至多 $\epsilon_m$，后者由上述长度一致界获得。

如果

$$
\boxed{\Delta_m^*>2(\eta+\epsilon_m),}
\tag{L10}
$$

并且每一步查询包含真实候选，则整个允许域内的序列均可精确恢复。

**证明。** 初始前缀正确。假设之前所有 token 正确，攻击者输入与参考输入一致，长度一致界控制当前候选误差；半间隔最近邻证明使当前 token 正确。对 token 位置归纳即可。候选误差的界不因归纳到第 i 个 token 而额外累加 i 次。证毕。

这里的全域间隔条件不是从量化位宽或残差连接自动得到的。Gemma 的投影、非线性和量化可能产生很近的有限候选点；过滤非 ASCII 也不能保证全域正间隔足够大。

如何独立认证这个条件，见 [global_margin_proof.md](global_margin_proof.md)：给出结构下界、有限状态覆盖的完整推导、直接候选对验证，以及条件可能不成立时的歧义判据。本文的式 (L10) 仍是充分条件，尚不是 Gemma 已通过该条件的结论。

在有限最大上下文、有限词表下，输入域有限，间隔可以作为明确有限集合的最小值定义；但值可能为零，集合规模也可能使直接穷举不可行。**“可以定义最小值”不等于“已经证明它严格大于误差”。**

## 10. 目前哪些部分已经证明，哪些没有

| 项目 | 状态 |
|---|---|
| softmax 与单 head attention 的上述长度一致界 | 本文已给完整证明 |
| 有界局部误差下的按层统一传播式 (L7) | 本文已给完整证明 |
| 同层旧 KV 扰动不向下改变后续新 KV 的结构性推论 | 在声明的计算图与相同输入条件下已证明 |
| 理想 group-wise 量化的 ±r 码误差界 (L9) | 本文已给完整证明 |
| 全域间隔充分大时整段恢复正确 | 本文已给归纳证明 |
| Gemma W8/KV8 实际全域常数足够紧、足以认证 ±2/±4 | 未证明，不能由有限常数存在或少量成功样本代替 |
| Pixel NPU 局部算子误差在整个允许长度内的统一界 | 未认证 |
| 600 次代理查询始终包含真 token | 未证明，需要单独覆盖分析 |

这比 128-token 结果的外推强：已经建立一条可用于全部声明长度的理论路线。但它仍不等于“Gemma 任意输入在每步 ±4 LSB 下必定恢复正确”。如果全局常数过松，就应收紧分析域、使用更紧的局部/数据依赖认证，或者只对被证书接受的位置报告保证；不能隐去间隔与执行误差条件。

## 参考与实现依据

- [Attention Is All You Need](https://arxiv.org/html/1706.03762)：attention 概率加权及逐位置 FFN 的基本计算图。
- [The Lipschitz Constant of Self-Attention](https://arxiv.org/abs/2006.04710)：无界输入域上的 dot-product attention 限制；本文范数域限制必不可少。
- [MLX quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)：group-wise affine 量化及码打包定义。
- [本地 Gemma 4 text 实现](../experiments/kv_guided_search_gemma4_int8/.venv/lib/python3.13/site-packages/mlx_lm/models/gemma4_text.py)：新 KV 的生产顺序、归一化、残差、PLE 及共享 KV。
- [本地 QuantizedKVCache 实现](../experiments/kv_guided_search_gemma4_int8/.venv/lib/python3.13/site-packages/mlx_lm/models/cache.py)：新条目量化并写入 `prev:offset`，原条目复用。
- [顺序实验](sequence_error_results/README.md)：有限数据的传播现象，可与定理相容，但不代替假设认证。

这些基本工具并非本文声称新发现的数学工具；此稿提供针对 KV 恢复问题的组合推导。本文没有新增模型实验，没有对硬件实现进行形式化验证。
