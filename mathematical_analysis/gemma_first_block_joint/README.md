# 继续经过 FFN 和 PLE：可消去的子空间与不能直接延用的方向下界

日期：2026-10-17。模型：本地 Gemma 4 E2B affine W8，revision
`03dcf209f3f549b4075e7191e77cf69b3d48e1b2`。层号从 1 开始。

## 1. 本轮究竟证明了什么

在[上一轮共同位置／差值证明](../gemma_joint_prefix_geometry/README.md)的基础上，继续检查真实 FFN 和 PLE。

**正面结论：** PLE 的直接增量总在一个由参数固定的 256 维子空间内。
实际参数允许在隐藏状态留下 1280 维正交补；在第 10、15 层的 K、V 各留下 256 维投影，消去紧邻前一层 PLE 的直接增量。
进一步归一化该投影后，方向还与这一处 PLE 及两次 RMSNorm 的标量分母无关。
这是对全部 token 特征、全部实数隐藏状态成立的结构定理，不要求前缀只含数字或少量英文词。

**同时发现的限制：** 在第一 attention 范围允许的一对共同偏移输入上，实际 FFN 能将固定加权单位方向距离从约 0.809 收缩到约 0.0113；加上真实 token 对应的 PLE 后约为 0.0106。
该结果经过整数区间算术认证，说明上一轮的 0.0436009 方向下界不能仅凭现有范围条件原样传过 FFN。
尚未证明这对输入由某个合法共同前缀产生，也没有证明输出碰撞。

**最终深层恢复仍未证明。** 这次消去的是一个直接分支；早先层的 PLE 已经进入 FFN 输入，不能随之删去。
原来的全域 $\rho_{uv}$、投影后实际候选分离下界以及自然量化／执行误差预算，还没有全部认证。

以下使用固定 W8 反量化系数和实数参考非线性。实际 BF16、KV8、NPU 执行不是这一定理中的精确实数计算，须另计偏差。

## 2. PLE 的直接影响可以精确投影消去

### 2.1 隐藏状态中的定理

设第 $\ell$ 层完成 FFN 残差之后、进入 PLE 之前的状态为 $x\in\mathbb R^{1536}$。
本地模型的 PLE 和层末 scalar 为

$$
a_v(x)=G(W_{g,\ell}x)\odot w_{\ell,v}\in\mathbb R^{256},
$$

$$
h_\ell=s_\ell\left[
x+\frac{D_{\gamma_{P,\ell}}W_{P,\ell}a_v(x)}
{\sqrt{\|W_{P,\ell}a_v(x)\|_2^2/1536+\varepsilon_P}}
\right].
$$

这里 $G$ 是模型实际使用的近似 GELU，$w_{\ell,v}$ 包含主 embedding 投影和逐层 embedding 表两项。
矩阵为无加性 bias 的线性层；W8 文件中的 `biases` 是权重量化的反量化元数据，已经并入固定矩阵系数，不能当成输出 bias。

定义固定矩阵

$$
C_\ell=D_{\gamma_{P,\ell}}W_{P,\ell}\in\mathbb R^{1536\times256}.
$$

将门控、token 特征及正的 RMSNorm 分母全部收进某个 $z_{\ell,v}(x)\in\mathbb R^{256}$，得到恒等式

$$
h_\ell=s_\ell[x+C_\ell z_{\ell,v}(x)].
\tag{B1}
$$

若 $C_\ell$ 满列秩，其列空间正交补的投影为

$$
P_\ell=I-C_\ell(C_\ell^TC_\ell)^{-1}C_\ell^T.
$$

于是

$$
\boxed{P_\ell h_\ell=s_\ell P_\ell x,\qquad
\operatorname{rank}P_\ell=1536-256=1280.}
\tag{B2}
$$

**证明。** $P_\ell C_\ell=0$，将它乘到 (B1) 即可。非线性门控、两个 token 的不同 PLE 特征均不影响这个等式。证毕。

“有 1280 维不受直接增量影响”并不意味着“任意两个 token 都在这些维度上不同”。后者仍需分离证书。

### 2.2 将投影直接放到目标 KV 上

目标层 $m\in\{10,15\}$ 读取前一层的输出

$$
h=s_{m-1}[x+C_{m-1}z],
$$

其中 $x$ 是第 $m-1$ 层完成 FFN 残差后的状态。
沿用[合并归一化恒等式](../gemma_fused_rho/README.md)，对 K 或 V 的一个 head，RoPE 之前的输出为

$$
U_j(h)=\frac{T_jh}{t_j(h)},\qquad
t_j(h)=\sqrt{h^TM_jh+c_j}>0.
$$

其中

$$
T_K=D_{\gamma_K}W_KD_{\gamma_{\rm in}},\qquad
T_V=W_VD_{\gamma_{\rm in}},\qquad
T_j\in\mathbb R^{512\times1536}.
$$

设 $R_j=T_jC_{m-1}$。实际参数认证了 $\operatorname{rank}R_j=256$，因此可以定义

$$
Q_j=I-R_j(R_j^TR_j)^{-1}R_j^T,\qquad L_j=Q_jT_j.
$$

得到

$$
\boxed{Q_jU_j(h)=\frac{s_{m-1}L_jx}{t_j(h)}.}
\tag{B3}
$$

**证明。** $Q_jT_jC_{m-1}=Q_jR_j=0$，直接代入即可。
此外，$T_j$ 满行秩，所以它映满 $\mathbb R^{512}$，从而
$\operatorname{range}(Q_jT_j)=\operatorname{range}Q_j$，二者的秩都是 256。证毕。

本地相应 $s_{m-1}>0$。若 $L_jx\ne0$，进一步得到

$$
\boxed{
\frac{Q_jU_j(h)}{\|Q_jU_j(h)\|_2}
=\frac{L_jx}{\|L_jx\|_2}.
}
\tag{B4}
$$

这消去了**紧邻前一层 PLE 的直接增量以及归一化标量对这个方向的影响**。
它没有消去 PLE 对幅度 $t_j(h)$ 的影响，也没有消去更早的 PLE、attention、FFN 对 $x$ 的影响。
若 $L_jx=0$，投影为零，不能使用单位方向式。

K 的缓存已经过位置旋转：对已知共同位置 $i$，应先撤销参考 RoPE，或在旋转坐标使用 $Q_{j,i}=O_iQ_jO_i^T$。
本地参考 RoPE 为正交的部分旋转，因此不改变 L2 范数；实际旋转计算误差仍归入执行预算。V 无此旋转。

### 2.3 实际参数证书

所有秩都用精确 dyadic 系数模素数 65521 的非零子式证明。
分母是 2 的幂，模该奇素数可逆；模素数非零的子式在有理数／实数中也非零。
矩阵形状给出相反方向的秩上界，因此得到精确秩，而非浮点奇异值阈值判断。

| 参数对象 | 形状 | 精确秩 | 作用 |
|---|---:|---:|---|
| $C_1,C_9,C_{14}$，分别认证 | 1536 × 256 | 256 | 隐藏正交补维数 1280 |
| 第 10、15 层各自 $T_K,T_V$ | 512 × 1536 | 512 | KV 线性映射满行秩 |
| 第 10、15 层各自 $T_KC_{m-1},T_VC_{m-1}$ | 512 × 256 | 256 | 每个 KV head 的正交补维数 256 |

共 11 个非零子式。生成器和核验器分别采用行消元／列消元，核验器重新解码源权重，并检查全部源张量哈希和正 layer scalar。
证据见 [ple_results.json](ple_results.json)、[ple_verification.json](ple_verification.json)。
该证书证明投影在数学上存在、维数准确；没有认证浮点构造投影矩阵的条件数或误差。若将投影实现为数值算法，需另行控制该误差，不能直接把逆 Gram 矩阵的浮点结果当成精确投影。

## 3. 它怎样帮助最终恢复证明

这不是要求攻击改用余弦距离，而是为 L2 分离寻找可证明的分量。
固定一个共同前缀 S，令某个投影 head 的干净向量为

$$
y_v=Q_jU_j(h_v)=r_v n_v,\qquad
r_v=\|y_v\|_2,\quad \|n_v\|_2=1.
$$

如果能在**全部允许共同前缀、全部不同候选**上证明

$$
r_v\ge r>0,\qquad \|n_u-n_v\|_2\ge\delta>0,
$$

则

$$
\begin{aligned}
\|y_u-y_v\|_2^2
&=(r_u-r_v)^2+r_ur_v\|n_u-n_v\|_2^2\\
&\ge r^2\delta^2.
\end{aligned}
$$

因此原 KV 的 L2 距离也至少为 $r\delta$，因为正交投影不会增加范数。
若 K、V 两个 head 均有界，可把两个平方下界相加。

借助 (B3)–(B4)，这些条件可以转为 PLE **之前**的状态条件：认证

$$
\|L_jx_v\|_2\ge\mu>0,\qquad
t_j(h_v)\le\bar t,
$$

以及所有成对 $L_jx_u,L_jx_v$ 的单位方向间隔 $\delta$，即可取

$$
r=\frac{s_{m-1}\mu}{\bar t}.
\tag{B5}
$$

**误差条件必须写清两侧。** 若目标观测误差至多 $b_T$，每个重算候选误差至多 $b_C$，令 $\eta=b_T+b_C$。
正确候选到观测的距离至多 $\eta$，错误候选至少为 $r\delta-\eta$；所以

$$
\boxed{r\delta>2(b_T+b_C)}
\tag{B6}
$$

足以保证该位置最近邻恢复。
若两侧各有上界 b，右侧是 **4b**；只有一侧误差为 b、另一侧精确时才是 2b。
正交投影本身不放大 L2 误差；若直接除以很小的投影范数再匹配，误差可能被放大，不能略去 $r>0$。

目前尚未为全部深层可达状态认证 B5 和方向间隔，也未认证实际 $b_T,b_C$。
±4 LSB 是用户观察到的最大坐标差，不是每维独立取端点的对抗扰动；本节没有将其换成那种更强的误差模型。
这条投影分离路线也没有自动给出原固定锚点 $A$ 对应的 $\rho_{uv}$，是进一步减少最终分离条件中一个直接分支的依赖。

## 4. FFN 为什么不能照搬上一轮的方向下界

### 4.1 保留两路共同偏移和真实 token 特征

固定两个真实 token：数字 `1`（236770）与 `2`（236778）。
在第一 attention 残差的已知范围内选

$$
h_v=\sqrt{1536}\,[e_v+D_{\gamma_A}\xi],\qquad v\in\{1,2\},\quad\|\xi\|_2<1.
\tag{B7}
$$

两路使用**完全相同**的 dyadic 向量 $\xi$，所以输入差严格等于 $\sqrt{1536}(e_1-e_2)$。
该点同时满足先前的单路椭球范围和第一 attention 方向分离条件。
这里尚未施加真实 attention 所有结构约束；没有证明存在共同合法前缀恰好实现 B7。

将这两个固定点依次送入真实 W8 FFN 残差、真实当前 token 的 PLE，再乘实际 layer scalar。
PLE 特征完整保留为

$$
w_{1,v}=\frac{
N_{\gamma_{\rm modelPLE}}(W_{\rm modelPLE}^{(1)}e_v)
+16e_{{\rm PLE},v}^{(1)}
}{\sqrt2}.
$$

主 embedding 的 $\sqrt{1536}$ 与 model projection 的 $1/\sqrt{1536}$ 在该实数参考中抵消。
没有把两个 token 的 PLE 替换成一样的向量，也没有把 FFN 替换成独立任意增量。

### 4.2 精确区间核验结果

为与上一轮 0.0436009 的证书一致，全程使用同一个固定坐标变换
$D=\operatorname{diag}(|\gamma_A|)$，比较

$$
\left\|\frac{D^{-1}a}{\|D^{-1}a\|_2}
-\frac{D^{-1}b}{\|D^{-1}b\|_2}\right\|_2.
$$

程序除以带符号的 $\gamma_A$ 与此距离相同：两者仅差共同的对角正交反射。
B7 中带符号的偏移也可通过同一反射改写为上一轮的绝对值椭球，不改变 $\|\xi\|$。

| 位置 | 固定加权单位方向距离（近似显示） | 普通 L2 距离（近似显示） |
|---|---:|---:|
| FFN 输入 | 0.8086605194 | 35.67291627 |
| FFN 残差后、PLE 前 | 0.0113092028 | 156.93726532 |
| 完整 PLE 与 layer scalar 后 | 0.0105798875 | 2.82518271 |

严格证书直接检验了：第一项大于 0.8，第二项小于 0.012，第三项小于 0.011，且第一项超过第三项的 70 倍。
完整有理数区间端点见 [witness_verification.json](witness_verification.json)。
偏移严格位于开放球内，因为

$$
\|\xi\|_2^2=
\frac{1090986098118912740263979}{1208925819614629174706176}<1.
$$

**普通 L2 距离在 FFN 后反而增加了。** 本结果是固定加权方向的收缩，不能描述成“FFN 把实际隐藏距离缩小 70 倍”。
层末普通距离还受到 layer scalar 的影响，不能把它当成最终 KV 间隔。

浮点优化仅提出一个待核验点。随后独立于优化结果的核验器重新读取 W8 码与元数据，用 96 位定点向外区间、精确整数仿射乘积、有界 tanh 计算实际非线性。
近似 GELU 的 binary64 常数被作为固定精确系数；RMSNorm epsilon 为实数 $10^{-6}$。
这些是明确指定的平滑参考算子，实际常数舍入和执行偏差仍需纳入误差预算。

### 4.3 严格能推出和不能推出的命题

这个固定见证否定了：**仅凭 B7 这种共同偏移范围及先前方向约束，所有允许输入经过第一 FFN／完整块后仍保持原 0.0436009 下界。**
因此不能把 FFN 当作方向间隔的自动保留器。

它没有否定某个更小的正统一下界，也没有证明两个输出相同；没有证明真实共同前缀能达到见证；没有证明深层 KV 碰撞或实际量化下恢复失败。
单个见证不是“全域都收缩”的证据。直接投影消去 PLE 的定理也不会自动修复 FFN 中的方向变化。

## 5. 接下来缺少的证明是什么

问题现在可以更具体地写成：对实际可达的成对状态 $x_u(S),x_v(S)$，认证目标层的固定映射 $L_j$ 下，既有非零可观测幅度，又有足够大的候选差异。
此前层的候选／共同前缀依赖仍须保留，不能回到两个独立的大球；同样不能只依据某个固定点的 Jacobian 可逆推断整个域。

新结果减少了需要直接处理的一个 PLE 分支，但没有完成 FFN、前缀可达性和执行误差这三个剩余环节。
原恢复定理目前仍处于条件证明阶段，不能据此报告可容忍 2 或 4 LSB。

## 6. 复核文件与命令

- [certify_ple.py](certify_ple.py)：生成固定子空间的参数秩证书。
- [verify_ple.py](verify_ple.py)：独立重提取源子式、检查非零行列式与正 scalar。
- [propose.py](propose.py)：浮点提议器；其 JSON 始终标记为未认证提议。
- [common_block_weighted_direction.npz](common_block_weighted_direction.npz)：被认证的固定 dyadic 共同偏移。
- [certify_witness.py](certify_witness.py)：重新解码实际参数并做整数区间核验；不使用优化器报告的距离作为证明。
- `common_block_direction.*` 是早先使用普通方向目标的诊断提议，未被本证书采用，不能混用两种度量的数值。

在仓库工作目录执行：

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_first_block_joint/certify_ple.py
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_first_block_joint/verify_ple.py
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_first_block_joint/certify_witness.py
```

本轮没有新增合法前缀推理／攻击成功率实验，没有修改 Android 应用或 LiteRT-LM。
