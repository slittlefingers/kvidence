# 继续证明：共同位置、候选差值与归一化后的可分离性

日期：2026-10-17。模型：本地 Gemma 4 E2B W8，revision
`03dcf209f3f549b4075e7191e77cf69b3d48e1b2`。

## 1. 本轮推进与严格边界

这一轮不只重写局部算子，而是进一步检查：**同一前缀带来的共同变化，究竟怎样影响候选之间的差异。**

完成了三项工作：

1. 推导共同位置误差 C 与候选差值误差 E 的成对传播定理；给出最终 KV 处直接约束原 $\rho$ 的充分条件。
2. 用实际 embedding 和第一层 post-attention 参数证明：当前数字 `0`–`9` 的全部 45 对候选，对任意有限参考文本前缀，在第一 attention 残差位置不仅有距离，而且不会退化为同方向、仅长度不同。一个固定加权坐标系中的单位方向距离统一大于 **0.0436009**。
3. 用第 10、15 层实际 K/V 参数严格证明：如果仍只使用现有的宽隐藏范围，即使进一步要求两路加完全相同的偏移、原差值完全不变，也允许归一化后的差距极小。这是对一类联合放松的反证，不是合法前缀碰撞。

**尚未完成：** 第 2 项的方向约束尚未通过真实 FFN、PLE 和后续 attention 传播到深层 KV。第 1 项的全域联合常数尚未全部认证。因此仍未证明真实自然 ±2–4 LSB 误差下的全域恢复。

后续已推进到[真实 FFN 与 PLE](../gemma_first_block_joint/README.md)：PLE 的直接增量可投影消去，且在目标层 KV 保留非零维数；但经严格区间核验，一个尚未证明前缀可达的共同偏移输入可以在 FFN 后低于这里的 0.0436009 下界。因此本页的第一 attention 方向证书仍有效，但不能原样套用到后续层。

本轮没有前缀枚举、Transformer 前向或新攻击实验。精确算术使用固定 affine W8 系数与实数参考算子；实际 BF16/KV8/NPU 执行另外计入误差预算。

## 2. 为什么必须同时跟踪共同位置与差值

对同一前缀 S 的两路隐藏输入定义

$$
m=\frac{h_u+h_v}{2},\qquad d=h_u-h_v,
\qquad h_u=m+d/2,\quad h_v=m-d/2.
$$

给两路都加同一个向量，只改变 m，不改变 d。但非线性变换后，输出差仍可能改变。

一个仅用于解释的例子：两点 $(1,1)$ 与 $(0,1)$ 的差是 $(1,0)$。
都加 $(1,-1)$ 后变成 $(2,0)$ 与 $(1,0)$，差仍是 $(1,0)$；
但若随后做没有 epsilon 的单位长度归一化，输出完全相同。
Gemma 的 epsilon 严格为正，下面的实际参数证明保留它，并不把这种理想化的完全相等直接当作 Gemma 碰撞。

因此，“共同前缀相同”不是“其影响会在全部非线性之后自动抵消”。
而且实际模型给两路带来的增量通常也不完全相同：attention 权重、FFN gate 和 PLE 都依赖当前候选。

## 3. 可沿算子传播的成对界

### 3.1 定理：共同变化通过曲率影响候选差异

设 g 是二次连续可微函数，定义输出差和输出均值

$$
\mathcal D_g(m,d)=g(m+d/2)-g(m-d/2),
$$

$$
\mathcal M_g(m,d)=\tfrac12[g(m+d/2)+g(m-d/2)].
$$

固定参考对 $(m_0,d_0)$，并已认证

$$
\|m-m_0\|\le C,\qquad \|d-d_0\|\le E.
$$

若在以下积分路径及其连接线段覆盖的区域内有

$$
\|Jg\|_2\le L,\qquad
\|D^2g[a,b]\|_2\le H\|a\|_2\|b\|_2,
$$

则

$$
\boxed{
\|\mathcal D_g(m,d)-\mathcal D_g(m_0,d_0)\|
\le LE+H\|d_0\|C,
}
$$

$$
\boxed{
\|\mathcal M_g(m,d)-\mathcal M_g(m_0,d_0)\|
\le LC+\frac H4(\|d_0\|+E)E.
}
$$

**证明。** 对 m 和 d 分别求导：

$$
\partial_m\mathcal D_g=Jg(m+d/2)-Jg(m-d/2),\qquad
\partial_d\mathcal D_g=\tfrac12[Jg(m+d/2)+Jg(m-d/2)],
$$

$$
\partial_m\mathcal M_g=\tfrac12[Jg(m+d/2)+Jg(m-d/2)],\qquad
\partial_d\mathcal M_g=\tfrac14[Jg(m+d/2)-Jg(m-d/2)].
$$

通过 Hessian 的线段积分，第一种 Jacobian 差的范数至多 $H\|d\|$。
对 $\mathcal D$，先在固定 m 处把 d 改为 $d_0$，再以固定 $d_0$ 改变 m，得到第一式。
对 $\mathcal M$，先以固定 d 改变 m，再改变 d；后一条路径上差向量的范数至多
$\max(\|d\|,\|d_0\|)\le\|d_0\|+E$，得到第二式。证毕。

这些路径包含 $m\pm d_0/2$、$m_0\pm d/2$ 等混合端点。
仅在两个真实端点认证了分母下界，并不自动保证连接路径也满足它。
使用局部 L/H 时必须覆盖这些路径；使用全空间界则无需这个额外几何假设，但界可能很松。

若 g 是线性函数，$H=0$；共同位置 C 完全不进入差值误差。
非线性情况下，新增项明确是 $H\|d_0\|C$，而不是给两路各加一份独立的 $LC$。

### 3.2 残差、PLE 和 attention 如何纳入

对 $g(h)=h+f(h)$，可以取 $L\le1+L_f$、$H\le H_f$；层末 scalar 使 C/E 的界都乘 $|s_\ell|$。
PLE 必须把 $(h,w_v)$ 作为输入，其中 $w_v$ 是实际 token 的 PLE 特征，不能省略它的差值。

对 attention，若比较不同前缀，g 的输入必须包含共享历史状态 S。
此时两路的历史部分相同，其差向量的历史部分为零，但均值部分包含 S。
**不能只对固定 $g_S(h)$ 求 h 的导数，就声称已经控制了 S 的变化。**
长度、mask 与位置需固定或逐域处理，且相应导数界需覆盖完整声明域。

这里给出了有效的联合递推形式，但未报告 Gemma 所有层的 L/H 已足够紧。

### 3.3 最终 KV 的导数界确实可以由参数写出

沿用 [合并归一化证明](../gemma_fused_rho/README.md)，每个 head 在 RoPE 之前为

$$
U(h)=Th/\sqrt{h^TMh+c},\quad
T=D_\gamma B,\quad M=B^TB/k+(\varepsilon_h/d_h)I,\quad c=\varepsilon_h\varepsilon_{\rm in}>0.
$$

这里 $d_h$ 是隐藏维度，避免与候选差向量 d 混淆。
设 $t(h)=\sqrt{h^TMh+c}$，则

$$
\|DU(h)[a]\|\le\frac{\sqrt{k}\|\gamma\|_\infty}{t(h)}\|M^{1/2}a\|,
$$

$$
\|D^2U(h)[a,b]\|
\le\frac{2\sqrt{k}\|\gamma\|_\infty}{\sqrt3\,t(h)^2}
\|M^{1/2}a\|\|M^{1/2}b\|.
$$

**证明要点。** $T^TT\preceq k\|\gamma\|_\infty^2M$，因此
$\|TM^{-1/2}\|\le\sqrt{k}\|\gamma\|_\infty$。
把归一化写成增广向量 $(M^{1/2}h,\sqrt c)$ 的单位长度归一化再投影。
对于 $R(X)=X/\|X\|$，一阶导数范数不超过 $1/\|X\|$。
对单位方向 a，写 $s=a^TX/\|X\|$，直接求导得

$$
\|D^2R(X)[a,a]\|^2
=\frac{1+2s^2-3s^4}{\|X\|^4}
\le\frac4{3\|X\|^4}.
$$

对任意输出测试方向投影后，Hessian 是实对称矩阵，其双线性算子范数由对角二次型控制，
所以同一 $2/\sqrt3$ 常数适用于任意 a,b。再复合上述线性映射即得结论。

在指定路径上认证 $t\ge t_{\min}$，并利用
$\lambda_{\max}(M)\le\|B\|_2^2/k+\varepsilon_h/d_h$，即可得到普通欧氏 L/H。
全空间总可取 $t_{\min}=\sqrt c$，但这不意味着得到的全域界能通过恢复条件。

## 4. 不经粗导数连乘，直接给出最终 rho 的联合条件

对一个 KV head，令 $p=Tm$、$q=Td$，两端分母为 $t_u,t_v$。
则精确两端差为

$$
\boxed{\Delta U=\alpha q+\beta p,}
$$

$$
\alpha=\tfrac12(1/t_u+1/t_v),\qquad
\beta=\frac{-2m^TMd}{t_ut_v(t_u+t_v)}.
$$

**证明。** 把 $T(m\pm d/2)/t_{u/v}$ 相减。
再利用 $t_u^2-t_v^2=2m^TMd$，将 $1/t_u-1/t_v$ 有理化即可。

因此，若能在全部合法同前缀端点上认证以下联合量：

$$
\underline t\le t_u,t_v\le\overline t,\quad
|m^TMd|\le J,\quad \|Tm\|\le P,\quad
\|T(d-d_0)\|\le E_T,
$$

定义固定 $q_0=Td_0$，以及

$$
\alpha_0=\tfrac12(1/\underline t+1/\overline t),\qquad a_*=\alpha_0q_0,
$$

则有具体可用的充分界

$$
\boxed{
\|\Delta U-a_*\|
\le
\tfrac12(1/\underline t-1/\overline t)\|q_0\|
+E_T/\underline t+JP/\underline t^3.
}
$$

证明是将误差写成 $(\alpha-\alpha_0)q_0+\alpha T(d-d_0)+\beta p$，
分别使用 $\alpha\le1/\underline t$、$|\beta|\le J/\underline t^3$。

各 K/V 分块的右端记为 $R_j$，则原先指定锚点 A 的目标满足

$$
\boxed{
\rho_{uv}^{(A)}
\le\|a_*-A(z_u-z_v)\|+\sqrt{\sum_jR_j^2}.
}
$$

这里 $a_*$ 是各块锚点的拼接，$z$ 仍是主 embedding 与 PLE 的完整输入特征。
若选择的 A 正好给出 $a_*$，第一项为零；否则不能删掉它。
共同位置的正交 RoPE 需同步作用于锚点。

**这次明确了需要控制的耦合项 $m^TMd$。**
它是两路投影前归一化能量差的一半，而不是一个候选的独立最大范数。
但上述 $J,P,E_T,\underline t,\overline t$ 尚未在深层的全部可达域上被认证到足够紧。

## 5. 另一条充分条件：直接保留归一化后的横向差异

由上一节 $\Delta U=\alpha q+\beta p$，投影到 p 的正交补后，$\beta p$ 完全消失。
定义

$$
\eta^2=
\begin{cases}
\|q\|^2-(p^Tq)^2/\|p\|^2,&p\ne0,\\
\|q\|^2,&p=0.
\end{cases}
$$

若两端分母都不超过 $\overline t$，则

$$
\boxed{\|\Delta U\|\ge\eta/\overline t.}
$$

因为 $\alpha\ge1/\overline t$。对 K/V 拼接，其平方距离至少为各块下界平方之和。

这里的条件可以直接写成固定参数矩阵的多项式不等式，而不需要先假设某个 KV 间隔存在。
设 $G=T^TT$。要保证 $\eta\ge\mu>0$，以下条件充分，且覆盖 p=0 的情况：

$$
d^TGd\ge\mu^2,
$$

$$
\boxed{
(m^TGd)^2\le(m^TGm)(d^TGd-\mu^2).
}
$$

G 完全由实际投影和 norm 参数决定；m,d 的约束必须由 embedding、PLE 和前面的计算图认证。
若所有允许候选对、所有可达端点均满足该条件，且
$\sqrt{\sum_j(\mu_j/\overline t_j)^2}>2b$，便得到稳健分离充分条件。

这不是改成 cosine 搜索：它仍然证明 L2 距离，只是保留其 Gram 行列式中的方向约束。
它也不要求差向量对所有前缀都接近某一个固定方向 $A(z_u-z_v)$。
原来的小 $\rho$ 条件是充分条件，不是恢复的必要条件；例如输出差随前缀整体旋转时，
距离可以不变，而相对固定方向的 $\rho$ 仍然很大。

## 6. 一项通过实际参数的全前缀正证书

### 6.1 从 embedding 与 post-attention gamma 推出方向条件

对第一 attention 残差，设 $D=\operatorname{diag}(|\gamma_A|)$，各对角元非零，
则每个当前 token 的输出可写成

$$
H_S(v)=\sqrt{d_h}(e_v+D x_v),\qquad\|x_v\|\le1.
$$

该范围对任何有限参考前缀成立。定义固定加权坐标

$$
y_v=D^{-1}H_S(v)/\sqrt{d_h}=a_v+x_v,\qquad a_v=D^{-1}e_v.
$$

对两个候选令

$$
A=\|a_u\|^2,\quad B=\|a_v\|^2,\quad C=a_u^Ta_v.
$$

若 $A+B+2C>0$ 且

$$
\boxed{q_*:=\frac{AB-C^2}{A+B+2C}>1,}
$$

则对于所有 $\lambda\ge0$ 和所有允许前缀都有

$$
\boxed{\|y_u-\lambda y_v\|\ge(1+\lambda)(\sqrt{q_*}-1)>0.}
$$

**证明。** 令 $t=\lambda/(1+\lambda)$，则

$$
\left\|\frac{a_u-\lambda a_v}{1+\lambda}\right\|^2
=\|(1-t)a_u-ta_v\|^2
\ge\frac{AB-C^2}{A+B+2C}.
$$

右边是相应二次多项式在整个实数 t 轴上的最小值，因此也不超过 $t\in[0,1]$ 上的最小值。
扰动项满足 $\|x_u-\lambda x_v\|\le1+\lambda$；用反三角不等式即得结论。
同时 $A,B>1$，故两路 y 均非零。

取 $\lambda=\|y_u\|/\|y_v\|$，并利用 $\|y_u\|\le\sqrt A+1$、
$\|y_v\|\le\sqrt B+1$，进一步得到

$$
\boxed{
\left\|\frac{y_u}{\|y_u\|}-\frac{y_v}{\|y_v\|}\right\|
\ge(\sqrt{q_*}-1)
\left(\frac1{\sqrt A+1}+\frac1{\sqrt B+1}\right)>0.
}
$$

这个条件完全由两行 embedding 和 gamma 决定，不含任何被枚举的前缀。

### 6.2 实际结果及作用域

本地十个单 token 数字 `0`–`9` 的全部 45 对均满足 $q_*>1$。
全部平方、内积和条件判断使用精确有理数；平方根使用有理端点并由独立平方不等式核验。
统一加权单位方向距离下界大于 **0.0436009**。

这比仅证明隐藏向量不相等更强：即使忽略它们的长度，也能保证上述固定加权坐标中的方向不同。
因为 D 可逆，也排除了原隐藏向量正比例共线。

**当前候选仅认证了这十个数字；旧前缀不受数字词表限制。**
该值不是普通 KV 距离、不是 LSB 容忍度，也没有证明经过 FFN/PLE 后仍保持这个界。
这项结果定位的是深层证明链的起点，不是在进行首层 KV 攻击实验。

## 7. 为什么仅改进范数界仍然不够：实际深层参数的严格检验

### 7.1 所检验的联合放松

沿用 [P5 参数报告](../gemma_parameter_condition_results/README.md) 的主残差路径锚点

$$
\bar h_v=\sqrt{d_h}\,s e_v,
\qquad \|h_v-\bar h_v\|\le R.
$$

s 是目标层之前的 layer scalar 乘积，R 由所有 attention/FFN/PLE post norm 幅度递推得到。
两个目标层的 R 分别约为 202.400149、129.736302。

现在进一步考察一个比两路独立球更受限的集合：

$$
\mathcal J_{uv}=
\{(\bar h_u+\xi,\bar h_v+\xi):\|\xi\|\le R\}.
$$

在这里，两路的共同偏移完全相同，而且隐藏差值**一丝不变**。
这不表示真实前缀必然满足这种加性形式，也不表示集合中每一点都可达。
它只是旧乘积球内一个严格保留共同变化的子集，用来检验“只保留这种相关性是否足够”。

### 7.2 构造与距离上界的证明

令 $d_0=\bar h_u-\bar h_v$、$m_0=(\bar h_u+\bar h_v)/2$。
选一个整数 $\lambda>1/2$，取

$$
\xi=\lambda d_0-m_0.
$$

于是两路变成

$$
h_u=(\lambda+1/2)d_0,\qquad h_v=(\lambda-1/2)d_0.
$$

共同偏移没有改变差值，但把差异转为同方向的长度差。
对某一 KV head，记

$$
A_0=d_0^TMd_0>0,\qquad N_0=\|Td_0\|^2,\qquad c=10^{-12}.
$$

由合并后的 KV 公式，对 $a>0$ 有

$$
U(a d_0)=Td_0/\sqrt{A_0+c/a^2}.
$$

将两路标量分母之差有理化，利用每个根号至少为 $\sqrt{A_0}$，得到

$$
\boxed{
\|U(h_u)-U(h_v)\|^2
\le\frac{N_0c^2\lambda^2}{A_0^3(\lambda^2-1/4)^4}.
}
$$

这是严格正 epsilon 下的实数界，不是将两次浮点结果相减后看到零。
K、V 的平方界相加即得该层 KV 的界；相同位置的正交 RoPE 不改变距离。

### 7.3 真实参数的认证结果

每层对 45 对数字分别构造一个上述共同偏移，并用精确有理数验证 $\|\xi\|^2\le R^2$。
结果如下。所有显示值都对应文件中的有理数证书。

| 项目 | 第 10 层 | 第 15 层 |
|---|---:|---:|
| 验证的候选对 | 45 | 45 |
| 无偏移参数锚点的统一 KV 距离下界 | 大于 8.42011 | 大于 14.62465 |
| 构造共同偏移后，各对 KV 距离的最大上界 | 小于 $4.176\times10^{-20}$ | 小于 $4.434\times10^{-22}$ |
| 每个构造点的两端隐藏范数 | 都大于 150 | 都大于 85 |
| 两端 K 分母 | 都大于 350 | 都大于 34 |
| 两端 V 分母 | 都大于 64 | 都大于 39 |

因此，这里的退化不是由“隐藏状态接近零”或“归一化分母接近零”造成的。
单独加强这些分母下界，也不能从这个联合集合中排除上述点。

对原定义中的任意固定基准向量 a，反三角不等式给出

$$
\sup_{\mathcal J_{uv}}\|\Delta KV-a\|
\ge\|a\|-\epsilon_{uv},
$$

其中 $\epsilon_{uv}$ 是对应构造点的 KV 距离上界。
所以仅在这个集合上认证小 $\rho$，不能得到大于 $\epsilon_{uv}$ 的分离保证。
这不依赖必须选哪一个 A；文件还记录了使用无偏移参数锚点时的具体 $\rho$ 下界。

**严格限制：** 上述点没有通过真实 attention/FFN/PLE 的可达性证明。
每层的构造分别进行，未证明它们能由同一真实前缀同时实现。
这不证明真实 Gemma、联合多层观测、自然量化执行或恢复算法失败。
极小的参考向量距离也不能在未分析量化边界时直接等同于 KV8 码完全相同。

## 8. 得到的研究结论与下一处缺口

现在可以区分两件事：

- **实际通过的正条件：** 第一 attention 残差处，数字候选的方向差异由 embedding 与 norm 参数统一保证，前缀不受小词表限制。
- **明确不够的放松：** 后续只记录两路范数、差值大小，甚至假定完全共同的加性偏移，仍允许归一化消去几乎全部可观察差异。

因此，后续应从第 6 节的参数证书出发，使用真实 FFN/PLE 关系与第 3 节的联合传播，
认证第 4 节的能量交叉项，或第 5 节的 Gram 不等式。
不能把本轮初始方向下界直接照搬到深层，也不能把此前连续全空间 FFN 非单射当成合法 token 对的反例。

这明确了一个可计算的证明目标：G 由模型参数给定，允许的 $(m,d)$ 由代码本和前面各层共同限定，
在该集合上证明 $(m^TGd)^2\le(m^TGm)(d^TGd-\mu^2)$ 等联合条件。
当前尚未对深层完成这个全域认证；没有把剩余难题写成已经满足的假设。

## 9. 文件与复现

- [certify.py](certify.py)、[results.json](results.json)：90 个联合放松见证、真实参数和整数矩阵乘积。
- [certify_cones.py](certify_cones.py)、[first_attention_cones.json](first_attention_cones.json)：45 对数字的全前缀方向证书。
- [verify.py](verify.py)、[verification.json](verification.json)：独立源数据与有理数核验。

在工作区根目录按顺序运行：

    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_joint_prefix_geometry/certify.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_joint_prefix_geometry/certify_cones.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_joint_prefix_geometry/verify.py

核验器重新读取所有相关源张量，用有界中国剩余定理核验全部四个投影乘积的所有元素，
而不是重复生成器的大整数乘法或抽查条目。模数经素性检查，其乘积大于误差可能范围，故同余推出精确相等。
全部 20,480 个参数锚点坐标通过有理数平方不等式核验；全部候选对、位移范围、分母下界与方向条件均复核。
浮点数仅用于显示结果；决定正性、包含关系和距离界不依赖浮点相减。
