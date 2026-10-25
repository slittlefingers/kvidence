# Gemma W8：继续证明时发现的 FFN 障碍

日期：2026-10-17。模型沿用本地 Gemma 4 E2B W8，revision `03dcf209f3f549b4075e7191e77cf69b3d48e1b2`；实际读取张量的哈希记录在核验 JSON 中。

## 1. 本轮结论及其边界

**仍未证明：** 对所有允许前缀，实际 INT8 执行产生的深层 KV 在自然数值误差下总能唯一恢复 token。

**本轮证明了一个实际参数决定的障碍：** 第一层 FFN 加残差、尚未进入 PLE 的实数参考映射，在整个连续隐藏空间上不是单射。此外，在上一轮使用的数字 `0` 的 post-attention 椭球内部，就存在 Jacobian 奇异点。因此，不能在这个整个椭球上证明统一的严格正距离缩放下界。

这次保留了真实 FFN 的所有矩阵、GELU、前后 RMSNorm 及输入依赖。结论不再只是“独立输出范围放松得太宽”。但这里的隐藏状态仍是连续输入，**没有证明它们可由合法 token 和同一个前缀产生**；非单射结论中的重合输入也未被构造为合法 token 对。

所以，本轮否定了“用所有连续隐藏状态上的逐层可逆性／统一距离保持来证明恢复”的一条路线，未否定有限码本上的深层 KV 恢复。后面的 PLE 还会注入 token 相关信息，也必须单独纳入。

没有进行新的 Transformer 前缀推理、攻击成功率测试或前缀枚举。

**后续作用域核验：** [有限前缀证书](../gemma_finite_prefix_certificate/README.md) 进一步证明，在前缀仅允许 BOS 和数字 0–9 的实数参考模型中，这里的特定负行列式见证点 $h_-$ 不可达。但用于排除该点的低维空间，在加入明确的 320 个英文单词 token 后已满维，因此不能把数字域的排除结论直接推广到英文。这不改变本文件的连续空间非单射定理，也没有构造合法 token 的碰撞。

## 2. 被证明的实际函数

令 $d=1536$，$m=6144$。从第一层 W8 载荷及其 BF16 scale/bias 读取的精确 affine 系数为 $W_g,W_u,W_d$。定义

$$
N_\gamma(x)=\frac{\gamma\odot x}{\sqrt{\|x\|_2^2/d+\varepsilon}},
\qquad \varepsilon=10^{-6}.
$$

本轮函数为

$$
\boxed{
T(h)=h+N_{\gamma_f}\left(
W_d\left[G(W_gN_{\gamma_p}(h))\odot W_uN_{\gamma_p}(h)\right]
\right).
}
$$

其中 $\gamma_p$ 是 `pre_feedforward_layernorm.weight`，$\gamma_f$ 是 `post_feedforward_layernorm.weight`，$G$ 逐元素作用。

参考激活明确取本地实现中两个 Python binary64 常数所定义的实数函数：

$$
G(a)=\tfrac12a\left[1+\tanh\left(\alpha(a+\beta a^3)\right)\right],
$$

$$
\alpha=\frac{7186705221432913}{9007199254740992},
\qquad
\beta=\frac{6444110634811895}{144115188075855872}.
$$

这里固定的是量化权重系数，**不是在证明实际 BF16/INT8 算子的逐位行为**；实际 kernel 的常数转换、舍入和 KV8 均不在这个参考定理内。整个函数因 $\varepsilon>0$ 而光滑。

## 3. 第一条尝试：用单调性保留输入差异

对凸域 $\Omega$，如果存在 $c>0$，使

$$
\frac{J_T(h)+J_T(h)^\mathsf T}{2}\succeq cI
\qquad(h\in\Omega),
$$

则对 $x,y\in\Omega$，令 $v=x-y$，沿线段积分可得

$$
v^\mathsf T(T(x)-T(y))
=\int_0^1v^\mathsf TJ_T(y+tv)v\,dt
\ge c\|v\|_2^2.
$$

由 Cauchy–Schwarz，

$$
\|T(x)-T(y)\|_2\ge c\|x-y\|_2.
$$

这个条件允许分支在某些方向上有很大增益，比要求整个残差分支 Lipschitz 常数小于 1 更宽松。然而实际参数核验给出一个固定二进制有理方向 $v$，满足

$$
v^\mathsf TJ_T(h_-)v < -20.2696 < 0.
$$

所以不能在含该点的开放区域使用这个欧氏强单调条件。这不是所有可能的加权条件或有限候选方向条件都被否定。

## 4. 两个固定输入及行列式符号证书

令 $e_0$ 是 token ID `236771`（数字 `0`）的精确 W8 embedding 行。提出两个可精确存储的输入：

$$
h_+\approx\sqrt d\,e_0,
\qquad h_-\approx\frac{11}{10}\sqrt d\,e_0.
$$

最终输入是 `proposal.npz` 和 `proposal_scaled.npz` 中存储的 **30 位小数二进制有理向量**。近似式仅解释如何找到它们；证书直接核验存储的确切向量，不假定近似等式成立。

两点均在以下开放椭球内：

$$
\Omega_0=\left\{h:
\left\|\frac{1}{\sqrt d}
\operatorname{diag}(\gamma_A)^{-1}(h-\sqrt d\,e_0)
\right\|_2<1\right\}.
$$

本地 $\gamma_A$ 每维非零。椭球坐标范数平方的认证上界分别小于 $3.61\times10^{-19}$ 和 $0.06780$，所以两点严格位于内部。这只是范围条件，不是 attention 可达性证明。

**认证结果：**

$$
\boxed{\det J_T(h_+)>0,\qquad\det J_T(h_-)<0.}
$$

没有直接把浮点 `det` 的正负当成证明。对每个点，核验程序重新包络真实 Jacobian $J$，并读取浮点算法提出的固定有理矩阵 $R,P,L,U$。令 $A=PLU$，其中 $P$ 为置换、$L$ 为单位下三角、$U$ 为非零对角的上三角，检查

$$
\|I-RJ\|_\infty<1,
\qquad \|I-RA\|_\infty<1.
$$

| 输入 | $\|I-RJ\|_\infty$ 认证上界（向上简写） | $\|I-RPLU\|_\infty$ 认证上界（向上简写） | 行列式符号 |
|---|---:|---:|---|
| $h_+$ | 0.217 | 0.0015 | 正 |
| $h_-$ | 0.498 | 0.0033 | 负 |

**符号推导：** 若 $\|I-B\|<1$，则 $I+t(B-I)$ 在 $0\le t\le1$ 上都可逆，其行列式从 1 连续变化而不经过 0，故 $\det B>0$。分别取 $B=RJ$ 和 $B=RA$，得到 $\det J$ 与 $\det A$ 同号。$\det A$ 的符号可以从置换奇偶性和 $U$ 的对角符号精确确定。

因此，浮点求逆、LU、特征向量仅用于**提出证书**；最终不等式决定是否接受，任一上界达到 1 都拒绝认证。

## 5. 定理一：椭球内不存在统一的正距离缩放下界

**命题。** 对上述实际参数参考函数，不存在 $c>0$ 使

$$
\|T(x)-T(y)\|_2\ge c\|x-y\|_2
\quad\text{对所有 }x,y\in\Omega_0\text{ 成立}.
$$

**证明。** 椭球为凸集，$h(t)=(1-t)h_++th_-$ 全部在其内部。$\det J_T(h(t))$ 连续且端点异号，中间值定理给出 $t_*\in(0,1)$ 使

$$
\det J_T(h(t_*))=0.
$$

记 $h_*=h(t_*)$，存在单位向量 $w$ 满足 $J_T(h_*)w=0$。因 $h_*$ 是内点，充分小的非零 $s$ 满足 $h_*+sw\in\Omega_0$。可微性给出

$$
T(h_*+sw)-T(h_*)=sJ_T(h_*)w+o(|s|)=o(|s|).
$$

故距离比趋于 0，与任何统一正下界 $c$ 矛盾。证毕。

这个结论对该连续椭球成立；不能直接推广为有限 token 对没有正间隔。单个 Jacobian 奇异点也不单独推出函数非单射，因此下一节另外证明全空间上的非单射。

## 6. 定理二：这个实际 FFN 残差函数在连续全空间上非单射

**命题。** 存在 $x\ne y\in\mathbb R^{1536}$ 使 $T(x)=T(y)$。

**证明。** 写 $T(h)=h+b(h)$。由于最后一个 RMSNorm，

$$
\|b(h)\|_2\le\sqrt d\|\gamma_f\|_\infty=:M
\quad\text{对全部 }h\text{ 成立}.
$$

考虑连续光滑变形 $T_t(h)=h+t b(h)$。若 $\|T_t(h)\|\le K$，则 $\|h\|\le K+M$，所以这个变形是 proper 的；其拓扑度等于恒等映射的度，即 1。这里使用 proper 同伦不改变拓扑度的标准定理：[MIT Analysis II，Lecture 29，Theorem 5.24](https://ocw.mit.edu/courses/18-101-analysis-ii-fall-2005/8f5cc88d4b6d8e97e2988fa9bbffb736_lecture29.pdf)。

认证得到 $\det J_T(h_-)<0$。逆函数定理给出一个含 $h_-$ 的小邻域，该邻域映到一个开集，而且其中的 Jacobian 行列式仍为负。由 Sard 定理，可以在这个像开集中选取整个映射 $T$ 的正则值 $z$；它至少有一个负符号的原像。

proper 性保证正则值只有有限个原像，且

$$
1=\deg T=\sum_{h\in T^{-1}(z)}\operatorname{sign}\det J_T(h).
$$

该计数公式及有限性见 [MIT Analysis II，Lecture 28，Lemma 5.19 与 Theorem 5.20](https://ocw.mit.edu/courses/18-101-analysis-ii-fall-2005/db5c8bc5b520598cd608e80fe3e41607_lecture28.pdf)。已有至少一个 $-1$ 项，而总和为 1，因此还必须有至少两个 $+1$ 项。于是某个输出有至少三个不同原像，$T$ 非单射。证毕。

这里是存在性证明，不宣称已经输出了具体重合的向量对。那些其他原像不保证位于 $\Omega_0$ 中，也不保证对应任意合法 token 或同一个前缀。

## 7. 数值证书具体保证了什么

`certify.py` 从原始 W8 载荷重新读取系数，并按链式法则构造 Jacobian：

$$
J_T=I+J_{N_f}(y)W_d
\left[\operatorname{diag}(u\odot G'(g))W_g+
\operatorname{diag}(G(g))W_u\right]J_{N_p}(h),
$$

其中 $g=W_gN_p(h)$、$u=W_uN_p(h)$、$y=W_d(G(g)\odot u)$，且

$$
J_{N_\gamma}(x)=\frac{\operatorname{diag}(\gamma)}{r}
-\frac{(\gamma\odot x)x^\mathsf T}{d r^3},
\qquad r=\sqrt{\|x\|_2^2/d+\varepsilon}.
$$

核验分为两个部分：

1. 非线性向量使用 96 位小数的整数区间。开方用整数平方根；`tanh` 通过指数函数的范围缩小、Taylor 有理余项界和整数区间反复平方包络。矩阵乘向量使用精确整数点积，因此不依赖浮点 `tanh` 的精度承诺。
2. 大矩阵运算使用带误差半径的 binary64 矩阵。在标准 IEEE 乘加模型下，点积长度为 n 的舍入误差由 $\gamma_{2n}=2nu/(1-2nu)$ 控制，$u=2^{-53}$；矩阵范数和后续误差传播均向外取整，并加入下溢余量。实际核验值有限，没有溢出。

这是**在所述算术模型下的计算辅助证明**，不是 Lean/Coq 形式化证明，也不是全部运算都使用任意精度有理数的证明。所用 binary64 计算运行于 CPU；这里的浮点误差界用于验证数学证书，不能代替目标 BF16/NPU 误差模型。

原始 JSON 保留比上表更完整的向外取整数值、参数哈希、提议文件哈希和作用域标记。

额外实现核对见 [arithmetic_checks.json](arithmetic_checks.json)：209 个 `tanh` 输入与独立 110 位十进制参考一致，3 组矩阵范数包络经 60 个精确有理数结果核对通过。这些用于检查实现错误，不以采样代替上述区间不等式证明；复现程序为 [check_arithmetic.py](check_arithmetic.py)。

## 8. 对原问题的实际影响

本轮确实排除了一个过强前提：实际 Gemma 的所有连续隐藏状态差异，不会都通过这个 FFN 被统一保留下来。不能继续把“每层残差天然可逆”作为完整证明的基础。

而用户要证明的是**固定有限码本、合法前缀、同一个前缀下的候选对**。这些集合可能完全避开上述重合状态和奇异方向。要证明原命题，后续必须保留这种可达性及成对约束，或者直接认证最终候选的匹配距离差；不能仅靠连续全空间的可逆性，也不能据本轮非单射结果宣布恢复不可能。

本轮未证明第 10／15 层的全域分离、自然 ±2–4 LSB 的容忍度、顺序恢复正确性或搜索覆盖率。没有将观测到的最大单维误差替换成每维独立的对抗扰动。

## 9. 复现

在工作区根目录运行。生成提议与核验分别执行；核验不使用提议阶段给出的行列式诊断结果。

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_ffn_proof_attempt/probe.py
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_ffn_proof_attempt/probe.py --scale 11/10 --output-stem proposal_scaled
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_ffn_proof_attempt/certify.py --proposal proposal
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_ffn_proof_attempt/certify.py --proposal proposal_scaled
```

- [正符号点核验结果](proposal_verification.json)
- [负符号点核验结果](proposal_scaled_verification.json)
- [证书核验器](certify.py)
- [浮点提议程序](probe.py)
