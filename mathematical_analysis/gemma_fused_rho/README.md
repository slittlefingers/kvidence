# 继续认证 rho：合并 Gemma 的归一化，并保留两路候选的抵消项

日期：2026-10-17。模型为本地 Gemma 4 E2B W8，revision
`03dcf209f3f549b4075e7191e77cf69b3d48e1b2`。

## 1. 先用人话说明本轮推进

我们仍要证明：**前缀改变时，两个候选之间的差异，最多能偏离固定基准多少。**
这个最大偏离才是原问题中的 $\rho_{uv}$。定义沿用
[参数分离证明](../parameter_separation_proof.md)，没有改成首层 KV 恢复实验。

以前把模型拆成很多步，每一步都取“可能放大多少”的最大值，再相乘。
这个方法虽然可以给出合法上界，却把前后步骤之间的抵消丢掉了。
本轮把 FFN 和最后生成 KV 的几步先合成一个函数，得到三个结论：

1. FFN 的两个线性分支带来的显式 $1/r^2$，会被后面的 RMSNorm 约掉。
   但 $r$ 仍在 GELU 门控和正则项中，不能说 FFN 与前缀无关。
2. 生成 K/V 时，两次 RMSNorm 也可以准确合并。
   最终分母只需要一个二次型；独立乘两个归一化放大系数会丢失这种结构。
3. 两路候选的平方距离中有负的交叉项。
   这些项可保留在 $\rho^2$ 的精确表达式里，不能先全部改成两个独立最大值。

**实际参数完成项：** 对前 15 个独立 KV 生产层的 30 个 FFN gate/up 矩阵，
以及第 10、15 层的 4 个 K/V 矩阵，认证了合并后的算子范数上界。
相对于“Frobenius 范数乘最大 gamma”的分开估计，上界缩小约 **3.216–43.141 倍**。
独立程序已重算全部源矩阵的列统计量，核对 119 个参数张量哈希及全部有理数上界。

**未完成项：** 还没有把全部合法同前缀端点的联合范围认证出来，
所以没有得到足够小的最终 $\rho_{uv}$，没有证明深层 KV 在自然 ±2–4 LSB 误差下全部可恢复。
本轮没有运行新的前缀样本、完整 Transformer 推理或 E6 搜索实验。

## 2. 参考模型与记号

所有恒等式针对固定 affine W8 系数、实数非线性运算的参考模型。
真实 BF16、KV8、NPU kernel 和重建历史缓存的执行偏差另行进入误差预算。
改变运算顺序在实数参考下等价，并不意味着两个浮点实现逐位相同。

采用本地计算图中的有效 norm 缩放向量 $\gamma$，不额外加 1。
模型无 FFN bias、无 MoE，隐藏维度 $d=1536$；本次目标第 10、15 层的 KV head 维度 $k=512$。
所有相应 RMSNorm 的参考 $\varepsilon=10^{-6}$。

写

$$
N_{\gamma,\varepsilon}(x)
=D_\gamma x/\sqrt{\|x\|_2^2/\dim(x)+\varepsilon},
\qquad D_\gamma=\operatorname{diag}(\gamma).
$$

GELU approximation 写成 $G(t)=t\Phi(t)$，其中

$$
\Phi(t)=\frac12\left[1+\tanh\bigl(\alpha(t+\beta t^3)\bigr)\right].
$$

$\alpha,\beta$ 沿用此前参考模型的固定 binary64 常数。
下面的代数恒等式对任意固定 $\Phi$ 都成立，不依赖某个 GELU 导数估计。

## 3. 定理一：整个 FFN 分支可以先合并，再取界

令 $h$ 为 attention 残差之后、FFN 之前的隐藏状态，定义

$$
r(h)^2=\|h\|_2^2/d+\varepsilon_p,\qquad
A_g=W_gD_{\gamma_p},\quad A_u=W_uD_{\gamma_p},
$$

$$
a(h)=A_gh,\qquad b(h)=A_uh,
\qquad Q(h)=W_d\left[a(h)\odot b(h)\odot\Phi(a(h)/r(h))\right].
$$

原来的 FFN 分支为

$$
f(h)=N_{\gamma_f,\varepsilon_f}
\left(W_d\left[G(W_gN_{\gamma_p,\varepsilon_p}(h))
\odot W_uN_{\gamma_p,\varepsilon_p}(h)\right]\right).
$$

则有恒等式

$$
\boxed{
f(h)=D_{\gamma_f}\frac{Q(h)}
{\sqrt{\|Q(h)\|_2^2/d+\varepsilon_f r(h)^4}}.
}
$$

**证明。** 两个线性分支分别为 $a/r$ 和 $b/r$，而
$G(a/r)=(a/r)\odot\Phi(a/r)$。
所以 post norm 之前的向量恰为 $Q/r^2$。
代入 post norm，并将分子、分母同时乘严格正数 $r^2$，即得结论。证毕。

这不是把 FFN 近似为线性函数；$\Phi(a/r)$ 的真实非线性完整保留。
完整 FFN 残差仍是 $T(h)=h+f(h)$，不能漏掉 $h$。
此前得到的连续空间非单射结论也没有因此失效。

### 3.1 两路差分中的共同项怎样保留

对同一前缀、两个当前候选的端点，用 $\Delta x=x_u-x_v$、
$\bar x=(x_u+x_v)/2$ 表示差与均值。记
$\phi_i=\Phi(a_i/r_i)$。逐坐标有

$$
\Delta(a\odot b\odot\phi)
=\bar\phi\odot(\bar a\odot\Delta b+\bar b\odot\Delta a)
+\overline{a\odot b}\odot\Delta\phi.
$$

这里 $\overline{a\odot b}=(a_u\odot b_u+a_v\odot b_v)/2$，
**不是** $\bar a\odot\bar b$。并且
$\Delta a=A_g\Delta h$、$\Delta b=A_u\Delta h$。
乘 $W_d$ 得到 $\Delta Q$ 的精确式。

因此认证对象可以保留 $\bar a,\bar b,\Delta h,\Delta\phi$ 的共同来源。
若再次把它们改成互相独立的最大值，就可能重新产生很大的粗界。

## 4. 定理二：最终 K/V 是一个带二次型分母的映射

令 $h$ 为目标 KV 生产层的输入隐藏状态，
$B=W D_{\gamma_{\rm in}}$、$z=Bh$，并令

$$
r_{\rm in}(h)^2=\|h\|_2^2/d+\varepsilon_{\rm in}.
$$

将 input norm、K 或 V 投影、head norm 合并，得到 RoPE 之前的函数

$$
\boxed{
U(h)=D_\gamma\frac{Bh}
{\sqrt{\|Bh\|_2^2/k+\varepsilon_h r_{\rm in}(h)^2}}.
}
$$

K 使用其 head norm 的 $\gamma_K$；V 的 $\gamma=\boldsymbol 1$。
证明与上一节相同：先写成 $z/r_{\rm in}$，再将外层 norm 的分子分母同时乘 $r_{\rm in}>0$。

进一步定义参数决定的矩阵与常数

$$
M=\frac{B^TB}{k}+\frac{\varepsilon_h}{d}I,
\qquad c=\varepsilon_h\varepsilon_{\rm in},
$$

则

$$
\boxed{U(h)=D_\gamma Bh/\sqrt{h^TMh+c}.}
$$

$M$ 严格正定，且本地参数给出 $c=10^{-12}$。
**这个 $10^{-12}$ 是实数参考公式中的常数，不是量化误差大小，也不是恢复误差上界。**

对于同位置、无幅度缩放的正交 RoPE，K 的两端距离不变。
但计算与固定锚点的偏差时，必须一起变换锚点：
$\|R_i\Delta U-a_K\|=\|\Delta U-R_i^Ta_K\|$。
不能在 $\rho$ 中忽略 RoPE 而保留原方向的锚点。
本地目标层在本层 attention 之前生成 KV；第 10 层 KV 仅依赖前 9 层的残差处理。

### 4.1 一个可以严格说明的效果：共同缩放与改变方向不同

固定非零 $h$，令 $\lambda>0$。从上式直接得到

$$
U(\lambda h)=\frac{D_\gamma Bh}{\sqrt{h^TMh+c/\lambda^2}},
$$

$$
\frac{dU(\lambda h)}{d\log\lambda}
=\frac{c/\lambda^2}{h^TMh+c/\lambda^2}U(\lambda h).
$$

也就是说，**单纯沿同一方向缩放隐藏向量的影响，受到这个明确系数控制**。
利用 $M\succeq(\varepsilon_h/d)I$，该系数至多

$$
\frac{\varepsilon_{\rm in}}
{\|\lambda h\|_2^2/d+\varepsilon_{\rm in}}.
$$

这说明把所有变化都按任意方向的最大放大率计算会丢掉结构。
但前缀可以改变隐藏方向，不能把任意前缀变化当成这种单纯缩放。
此前第一层 pre-FFN 的隐藏范数下界也不能直接移作第 10／15 层输入的下界。

## 5. 定理三：两个归一化端点的共同变化可以精确保留

FFN 和 K/V 都具有统一形式

$$
U_i=D_\gamma z_i/t_i,\qquad
t_i^2=\|z_i\|_2^2/k+\varepsilon s_i^2,\qquad t_i>0.
$$

FFN 取 $k=d,z_i=Q(h_i),s_i=r(h_i)^2$；
K/V 取 $z_i=Bh_i,s_i=r_{\rm in}(h_i)$。
引入辅助向量

$$
Z_i=(z_i,\sqrt{k\varepsilon}\,s_i),\qquad \|Z_i\|_2=\sqrt{k}\,t_i.
$$

任意两个非零向量 X,Y 都满足

$$
\left\|\frac X{\|X\|}-\frac Y{\|Y\|}\right\|^2
=\frac{\|X-Y\|^2-(\|X\|-\|Y\|)^2}{\|X\|\|Y\|}.
$$

**证明。** 展开两边，均为 $2-2X^TY/(\|X\|\|Y\|)$。证毕。
将其用于 $Z_u,Z_v$，再扣掉最后一维的平方差，得到精确式

$$
\boxed{
\left\|\frac{z_u}{t_u}-\frac{z_v}{t_v}\right\|^2
=\frac{\|\Delta z\|^2+k\varepsilon(\Delta s)^2-k(t_u-t_v)^2}{t_ut_v}
-k\varepsilon\left(\frac{s_u}{t_u}-\frac{s_v}{t_v}\right)^2.
}
$$

右边整体非负，但存在两个不能忽视的减项。
如果只需要较粗的合法上界，可删去这两个非正项，得

$$
\|U_u-U_v\|_2
\le\|\gamma\|_\infty
\frac{\sqrt{\|\Delta z\|^2+k\varepsilon(\Delta s)^2}}{\sqrt{t_ut_v}}.
$$

这个简化上界在径向变化时仍可能很松。
它的分母、分子和减项必须在同一对可达端点上联合约束；恒等式本身不自动给出小的全域界。
区间计算中，不能给两个减项擅自代入未经认证的正下界。

## 6. 怎样接回原本的 rho，而不是停在局部算子

设 $\Gamma=D_\gamma^2$。加权后，直接展开有

$$
\|U_u-U_v\|^2
=\frac{z_u^T\Gamma z_u}{t_u^2}
+\frac{z_v^T\Gamma z_v}{t_v^2}
-\frac{2z_u^T\Gamma z_v}{t_ut_v}.
$$

给定固定锚点 $a$，进一步有

$$
\boxed{
\begin{aligned}
\|U_u-U_v-a\|^2
={}&\frac{z_u^T\Gamma z_u}{t_u^2}
+\frac{z_v^T\Gamma z_v}{t_v^2}
-\frac{2z_u^T\Gamma z_v}{t_ut_v}+\|a\|^2\\
&-\frac{2a^TD_\gamma z_u}{t_u}
+\frac{2a^TD_\gamma z_v}{t_v}.
\end{aligned}
}
$$

对目标 KV 的 K、V 分块分别取 $a$ 为 $A(z_u^{\rm input}-z_v^{\rm input})$
的相应部分，K 的锚点按第 4 节处理 RoPE，然后将上述平方表达式相加，
就是原来的 $\|r_{uv}(S)\|^2$。输入特征 $z^{\rm input}$ 与上式的投影向量 $z=Bh$ 是不同对象。

这给出了**准确、直接对应原目标的联合优化表达式**。
交叉内积在这里是平方 L2 距离的代数项，并不是提出另一种 cosine 搜索算法。

如果构造一个由参数、embedding 和计算图限定的联合集合 $\mathcal C_{uv}$，
覆盖所有合法同前缀 $S$ 所产生的两路端点，
并认证上述总表达式在此集合上不超过 $R_{uv}^2$，则立即得到

$$
\rho_{uv}\le R_{uv},\qquad
\|A(z_u^{\rm input}-z_v^{\rm input})\|>R_{uv}+2b
\ \Longrightarrow\ \text{该候选对稳健分离}.
$$

**尚未完成的是这个联合集合上的最大值认证。**
本轮的矩阵证书与局部恒等式，没有代替这一步。
在继续传播时，还必须保留 FFN 残差、真实 PLE 输入及 layer scalar：

$$
\Delta h_{\rm afterFFN}=\Delta h_{\rm beforeFFN}+\Delta f,
\qquad
\Delta h_{\ell}=s_\ell(\Delta h_{\rm afterFFN}+\Delta p).
$$

若 PLE 门为 $G(W_Ph)$，token 的实际 PLE 输入为 $w$，则乘积差为
$\bar w\odot\Delta G+\overline G\odot\Delta w$。
$\Delta w$ 不为零，不能按共同前缀项抵消。
旧 KV 不一致时也必须另加历史数值误差预算。

## 7. 真实参数证书：先合并 W 与 gamma

对任何矩阵 W 与对角缩放，有

$$
\|WD_\gamma\|_2^2
\le\min\left\{
\sum_j\gamma_j^2\|W_{:j}\|_2^2,
\|WD_\gamma\|_1\|WD_\gamma\|_\infty
\right\}.
$$

第一项是 Frobenius 上界，第二项是标准行／列绝对和上界。
其中第一项总不超过原来分开估计的
$\|\gamma\|_\infty^2\|W\|_F^2$。
注意：下面比较的是这两种**认证上界**，不是实际奇异值，也不是完整 FFN 的 Lipschitz 常数。

| 位置 | 分开估计上界（约） | 合并后认证上界（约） | 上界缩小倍数（约） |
|---|---:|---:|---:|
| 第 1 层 FFN gate | 12,398.833 | 2,098.977 | 5.907 |
| 第 1 层 FFN up | 12,755.069 | 2,348.348 | 5.432 |
| 第 10 层 FFN gate | 39,262.414 | 1,247.388 | 31.476 |
| 第 10 层 FFN up | 39,143.117 | 1,443.713 | 27.113 |
| 第 15 层 FFN gate | 7,400.710 | 235.540 | 31.420 |
| 第 15 层 FFN up | 7,748.030 | 254.985 | 30.386 |
| 第 10 层 K projection | 20,564.081 | 2,565.361 | 8.016 |
| 第 10 层 V projection | 20,529.861 | 475.880 | 43.141 |
| 第 15 层 K projection | 2,975.298 | 419.949 | 7.085 |
| 第 15 层 V projection | 3,241.613 | 464.057 | 6.985 |

表中十进制仅作展示；严格上界使用 [results.json](results.json) 的有理数端点。
全部 34 项均通过 [verification.json](verification.json) 的独立核验。

认证从本地 packed W8 权重及 BF16 scale/bias 解码精确整数系数，
所有整数乘加前检查溢出界，平方根上端点由整数计算并由有理数平方不等式复核。
核验器不调用生成器，不使用 SVD；全部源列平方和、列绝对和及最大加权行和均重新计算。

另外做了 42 个实际权重的有限块恒等式检查，最大相对差约 $7.06\times10^{-16}$，
以及 35 个成对 Gram／辅助向量上界检查。
这些固定隐藏输入**没有被声称为合法前缀可达输入**，检查只用于发现实现错误。

直接 binary64 计算 Gram 式时，接近相消的项会造成数值精度损失。
程序因此对给定 binary64 的 $\gamma,z,t$ 使用精确有理数核验
“Gram 展开 = 逐维差平方和”，并逐维核验两次浮点运算的舍入界。
这些检查不将浮点相消误差误认成真实网络差异，也不构成全域恢复证书。

## 8. 复现与剩余缺口

在工作区根目录运行：

    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_fused_rho/analyze.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_fused_rho/verify.py

本轮证据不需要下载新模型，不需要 GPU/NPU，也不枚举前缀。
新的代数形式适用于本地这些层的全部实数输入，不限于十个数字；
但“公式适用全域”与“全域最大值足够小”是两个不同结论。

还需完成：同前缀联合可达范围的认证、最终 $\rho_{uv}$ 与有限码本间隔的严格比较，
以及真实执行路径的自然误差预算。
“最大单维差 4 LSB”不能直接替代该预算，也不能据本轮结果换算成已能承受多少 LSB。
当前是从参数改进证明表达式和界，尚未完成用户要求的深层稳健恢复定理。

后续推进见 [共同位置与候选差值](../gemma_joint_prefix_geometry/README.md)。
该文给出可代入原 $\rho$ 的联合能量条件，并证明数字候选在第一 attention 残差处的全前缀方向分离。
同时实际参数见证说明：保留完全相同的加性偏移和不变的隐藏差值仍不足以排除深层 KV 的径向退化；
必须进一步约束真实 FFN/PLE/attention 所产生的联合方向关系。见证未被证明为合法前缀可达。
