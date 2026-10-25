# 有限前缀词表的结构性证明，以及不能直接推广的原因

日期：2026-10-17。模型：本地 Gemma 4 E2B W8，revision
03dcf209f3f549b4075e7191e77cf69b3d48e1b2。

## 结论

**数字前缀上的结论成立，但其关键低维结构不能直接推广到普通英文词表。**

| 前缀允许的 token 集合 | 第一层 attention 增量所在线性空间的维数 | 完全消除该增量后剩余的维数 |
| --- | ---: | ---: |
| BOS 加单 token 数字 0–9 | 88 | 1448 |
| 本文明确列出的 320 个带前导空格的英文单词 token | 1536 | 0 |

两行均由实际参数的精确算术证书核验，不是根据维数猜测，也没有采样前缀。
这里限制的是**前缀允许使用的 token 集合**，不只是当前待猜的候选集合。
320 个词是明确的有限子集，不是对“正常英语”的完整形式化。

第二行证明：如果希望找一个固定非零线性投影，让所有允许的 attention 增量都落入其所用**线性张成空间**的核，那么这条路线已无剩余方向。它不证明真实增量遍历整个空间，更不证明存在合法 token 碰撞。

## 1. 为什么小词表产生结构

以下均为 W8 系数精确 affine 解码后、非线性和矩阵计算按实数定义的参考模型。
没有把它当作实际 BF16/NPU/KV8 执行的证明。

令 $d=1536$，第一层有 8 个 query head、1 个 KV head，head 维数为 256。
允许的前缀 token 集合为 $A$；$e_t$ 是 token $t$ 的 W8 embedding。
第一层的 V 来自该位置的 embedding，尚未混入历史。
输入 RMSNorm 和无可学习增益的 V norm 只给下面的向量乘正标量：

$$
u_t=W_VD_{\gamma_{\mathrm{in}}}e_t.
$$

因此，允许的 V 都落在

$$
U_A=\operatorname{span}\{u_t:t\in A\}.
$$

RoPE 作用于 Q/K，不改变上述 V 空间。无论前缀顺序、长度及合法 mask 如何，各 query head 的 attention 输出都是这些 V 的加权和。
令 $W_{O,h}$ 为输出投影矩阵对应第 $h$ 个 head 的列块，$D_{\gamma_A}$ 为 post-attention RMSNorm 的对角增益，则 attention 残差增量属于

$$
S_A=\sum_{h=1}^{8}D_{\gamma_A}W_{O,h}U_A.
$$

理由是：输出投影是线性的，post-attention RMSNorm 又只在乘 $D_{\gamma_A}$ 之外施加一个标量。
所以当前 token 为 $v$、前缀为 $p$ 时，FFN 之前的隐藏状态满足

$$
H_p(v)\in c_v+S_A,\qquad c_v=\sqrt d\,e_v.
$$

这里要求当前 token 也属于 $A$。子空间包含关系是必要条件，不把所有子空间中的点都当成可达点。
第一层 PLE 的注入在这一观测位置之后。

令 $P_A$ 是到 $S_A^\perp$ 的正交投影，则有

$$
P_AH_p(v)=P_Ac_v.
$$

这就是小词表证明的关键：投影后历史项消失，保留一个只由 token 决定的量。
甚至对不同的允许前缀 $p,q$，也有

$$
\|H_p(u)-H_q(v)\|_2
\ge \sqrt d\,\|P_A(e_u-e_v)\|_2.
$$

这个条件比原问题中“两个候选共享同一前缀”更强；其优势是易于认证，代价是可能丢失有用的共同前缀约束。

## 2. 对数字前缀实际证明了什么

取 $A=\{\mathrm{BOS},0,\ldots,9\}$。
把 $D_{\gamma_A}W_{O,h}u_t$ 排成矩阵 $B$，形状为 $1536\times88$。
精确整数和有理数核验得到：

- $B$ 满列秩，故 $\dim S_A=88$。
- 数字之间全部 45 对的上述隐藏状态 L2 间隔均 **大于 34.9618**。
- 上一轮 FFN 的特定负行列式见证点，与全部 10 个仿射空间 $c_v+S_A$ 的距离均 **大于 4.1838**。

因此，在这个数字前缀域和实数参考模型中，那个特定见证点不可能是当前数字的真实 FFN 输入。
这不是排除所有 FFN 奇异点，也没有证明 FFN 在这些可达集合上单射。
这些数值是隐藏空间的 L2 距离，**不是 LSB 容忍度**。

与上一轮任意前缀的 1.05756 下界相比，本轮 34.9618 使用了更窄的前缀域；不能把它说成相同作用域上的无条件提高。

### 精确核验为何不依赖浮点 SVD 的正确性

SVD 只提出近似左逆 $R$ 和方向 $w$。随后将其固定为二进制有理数，用精确整数与有理数验证

$$
\|I-RB\|_2\le\eta<1,\qquad
\sigma_{\min}(B)^{-1}\le
\frac{\|R\|_F}{1-\eta}\le\kappa.
$$

这里用保守的 $\eta=88\|I-RB\|_\infty$，
认证值约为 $2.046831\times10^{-6}$，$\kappa$ 约为 $0.2305465$。
对拟议方向 $w$，

$$
\|w-P_Aw\|_2\le\kappa\|B^\mathsf Tw\|_2=:\delta.
$$

所以 $\Delta e=e_u-e_v$ 满足

$$
\sqrt d\,\|P_A\Delta e\|_2
\ge
\frac{\sqrt d\,[w^\mathsf T\Delta e-\delta\|\Delta e\|_2]}
{\|w\|_2+\delta},
$$

当右侧分子为正时得到分离下界。开方均使用整数平方根给出向外有理包络。
见 [生成与参数来源](build.py)、[证书](certificate.npz)、
[结果](results.json) 和 [独立核验](verification.json)。
独立核验器读取已提取的系数；重新运行生成程序才会重新检查模型提取过程。

## 3. 为什么扩大到英文后，这个结构消失

一般只能保证

$$
\dim S_A\le\min\{1536,\;8\min(|A|,256)\}.
$$

上界达到 1536 **并不自动说明真实秩等于 1536**。因此本轮进一步检查了实际权重。

从本地 canonical single-token 词表中选取 320 个明确的英文单词，每个均带一个前导空格，例如 the、of、and 对应的 token。
完整列表和 token ID 保存在 [generalization_results.json](generalization_results.json)。
这不是随机句子测试，也未运行 Transformer 前缀推理。

实际参数满足：

$$
\operatorname{rank}[u_t]_{t\in A}=256,\qquad
\operatorname{rank}W_O=1536.
$$

此外 $D_{\gamma_A}$ 的全部对角元非零。因此

$$
U_A=\mathbb R^{256},
$$

$$
S_A
=D_{\gamma_A}\operatorname{range}
[W_{O,1}\ \cdots\ W_{O,8}]
=\mathbb R^{1536}.
$$

从而

$$
S_A^\perp=\{0\},\qquad P_A=0.
$$

上一节的投影分离下界对所有 token 对都变成零。
任何包含这个子集的更大前缀词表，在线性张成空间这一步也会得到同样结论。
同时，这时 $c_v+S_A=\mathbb R^{1536}$，所以数字域上排除旧 FFN 见证点的论证也不能直接迁移。

### 满秩结论如何严格核验

W8 affine 解码后的权重、embedding 和 RMSNorm 增益均为二进制有理数。
取素数 $p=65521$，所有分母都是 2 的幂，在模 $p$ 下可逆。

我们用精确整数运算构造两个矩阵的模 $p$ 值，并找到：

1. $[u_t]$ 的一个 $256\times256$ 子式，其行列式模 $p$ 非零；
2. $W_O$ 的一个 $1536\times1536$ 子式，其行列式模 $p$ 非零。

有理数子式若为零，模 $p$ 后也必须为零。因此上述非零结果严格推出实数矩阵的满行秩。
没有用浮点奇异值阈值判定秩。

[提取与秩检查](check_generalization.py) 使用行消元；
[独立子式核验](verify_generalization.py) 使用列消元，并输出
[核验记录](generalization_verification.json)。
证书来源通过参数哈希和文件哈希记录；这不是形式化证明助手的内核认证。

## 4. 推广边界到底意味着什么

**已经证明的限制：** 在上述英文前缀词表上，依赖这个低维线性空间的“完全消除历史项，再比较 embedding 投影”方法失去区分能力。

**没有证明的结论：**

- 真实 attention 增量能取整个 $\mathbb R^{1536}$ 的任意值。它仍受 softmax、共享前缀及 RMSNorm 范围等约束。
- 两个不同 token 在同一合法前缀下发生真实碰撞。
- 深层 KV 恢复不可能，或自然量化的最大单维 2–4 LSB 误差必然导致误判。
- 数字前缀的 88 维结构能直接用于第二层及更深层。后续 V 已依赖上下文，不再是每个 token 对应一个固定向量。

原问题仍可以在数学上表述和研究，但需要另一种保留更多约束的证明。
一个方向是直接约束**同一个前缀对两个候选产生的相对影响**，而非要求对全部历史影响都存在完全正交的方向。
满秩不排除这类成对界：历史影响可能在每个方向都非零，但两个候选之间的差仍受到共同前缀的限制。
本轮没有完成这种成对界，更没有得到实际 Gemma INT8 深层 KV 的全域稳健恢复证书。

## 5. 复现

在工作区根目录依次运行：

    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_finite_prefix_certificate/build.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_finite_prefix_certificate/verify.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_finite_prefix_certificate/check_generalization.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_finite_prefix_certificate/verify_generalization.py
