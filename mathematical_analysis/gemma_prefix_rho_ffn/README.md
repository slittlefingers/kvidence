# 回到 p／rho：直接认证前缀偏移经过 FFN 时的参数上界

日期：2026-10-17。对象：本地 Gemma 4 E2B W8，revision `03dcf209f3f549b4075e7191e77cf69b3d48e1b2`。

## 1. 当前目标与实际完成程度

本轮研究的是前缀变化如何影响两候选的相对差异，即此前记作 p／$\rho$ 的量。
这里口头的 p 正式记作 $\rho$，应与条件定理 C5 中另行定义的加性余项 $p_{uv}$ 区分。
没有读取 token embedding 行，没有选择数字对，没有执行前缀推理，也没有把前缀限制在未经证明的小输入球里。

**已完成：** 为第 1–14 层实际 FFN 生成并核验一阶／二阶导数上界的参数系数，给出 FFN 对 p 的完整递推上界；另用 post RMSNorm 的有界输出给这个新增项设置全域上限。
第 10、15 层 KV 分别在第 9、14 层残差处理之后生成，因此覆盖这些生产层之前的全部 FFN。

**尚未完成：** 这些全空间导数上界很松；还没有得到足够小的完整深层 p，也没有证明其能通过恢复阈值。不能将局部系数核验表述为最终 p 已验证成功。

此前数字 1、2 附近输入球的尝试已单列为旁支，不用于本轮的 p 证明。

2026-10-18 后续：[完整 p 参数计算器](../parameter_p/README.md)已经给出全链条保守上界。它使用本节幅度上限所属的全域路线，接上 attention、PLE、scalar 和最终 KV；本节导数系数仍作为后续收紧工具。完整上界已可计算，不再要求先证明 p 足够小。

## 2. 要约束的是两路相对差异的变化

在 FFN 输入处，对任意允许前缀 S 写

$$
h_u=m+d/2,\qquad h_v=m-d/2.
$$

固定参考对

$$
\bar h_u=m_0+d_0/2,\qquad\bar h_v=m_0-d_0/2.
$$

假设此前的参数包络已经对整个声明前缀域保证

$$
\|m-m_0\|_2\le C,\qquad\|d-d_0\|_2\le E.
$$

E 是这一步输入差相对参考差的偏移；C 是两路共同位置的变化。
共同前缀的影响通常会同时改变 C 和 E，不能直接假定 E=0。

令真实 FFN 分支为 f，残差输出为 $T(h)=h+f(h)$。要约束的输出偏移是

$$
E_{\rm out}\ge
\left\|[T(h_u)-T(h_v)]-[T(\bar h_u)-T(\bar h_v)]\right\|_2.
$$

这正是 [共同位置／差值递推](../gemma_joint_prefix_geometry/README.md)中的成对误差，而不是另一个候选分离实验。

## 3. 定理：FFN 对 p 的新增量可以同时用三种界限制

设在所需端点与积分路径上

$$
\|Jf\|_2\le L_f,\qquad
\|D^2f[a,b]\|_2\le H_f\|a\|_2\|b\|_2,
$$

并且对所有端点有 $\|f(h)\|_2\le B_f$。则

$$
\boxed{
E_{\rm out}\le E+
\min\left\{
4B_f,\quad
L_f(E+2C),\quad
L_fE+H_f\|d_0\|_2C
\right\}.
}
\tag{F1}
$$

**证明。** 输出差偏移等于输入差偏移加

$$
\Delta_f=f(h_u)-f(h_v)-f(\bar h_u)+f(\bar h_v).
$$

第一种界：四项各至多 $B_f$，故 $\|\Delta_f\|\le4B_f$。

第二种界：每个对应端点位移至多 $C+E/2$，在相应连接线段上使用 $L_f$，得 $\|\Delta_f\|\le L_f(2C+E)$。

第三种界：先在固定 m 处把 d 改为 $d_0$，费用至多 $L_fE$；再以固定 $d_0$ 把 m 改为 $m_0$。
后一步的导数是 $Jf(m+d_0/2)-Jf(m-d_0/2)$，由 Hessian 界其范数至多 $H_f\|d_0\|$，费用至多 $H_f\|d_0\|C$。
最后加输入差偏移 E，取三种合法上界的较小值即得 F1。证毕。

第三种界包含混合端点 $m\pm d_0/2$，以及这些线段间的连接路径。使用局部分母下界时必须覆盖这些路径；下面的全空间常数没有这个覆盖缺口。

特别地，若另有严格依据保证两路只发生共同平移、E=0，则

$$
E_{\rm out}\le\min\{4B_f,\ 2L_fC,\ H_f\|d_0\|C\}.
$$

这明确指出：即使原候选差不变，非线性曲率仍可能把共同前缀变化转为候选差偏移。

### 3.1 与原固定锚点 A 的 p 如何衔接

F1 的输出参考差是 $T(\bar h_u)-T(\bar h_v)$。
若原目标指定的是 $A(z_u-z_v)$，还需要加入锚点差异：

$$
\rho_{\rm out}^{(A)}
\le E_{\rm out}
+\left\|T(\bar h_u)-T(\bar h_v)-A(z_u-z_v)\right\|_2.
$$

不能把这个第二项悄悄省去。后续 attention、PLE、scalar 和最终 KV 映射也仍要继续传播。
F1 验证的是完整 p 递推中的 FFN 组成部分，并没有替代整个 $F_\theta$ 的认证。

## 4. L、H、B 如何由实际 FFN 参数直接给出

### 4.1 固定参考算子

本地隐藏维度 $n=1536$，RMSNorm 的参考 epsilon 为 $10^{-6}$。
将 pre norm 的可学习缩放并入权重：

$$
A_g=W_gD_{\gamma_p},\qquad A_u=W_uD_{\gamma_p},
\qquad q(h)=\frac{h}{r_p(h)},
\quad r_p(h)=\sqrt{\|h\|^2/n+\varepsilon_p}.
$$

定义

$$
a=A_gq,\quad b=A_uq,\quad
y=W_d[G(a)\odot b],\qquad
f(h)=\frac{D_{\gamma_f}y}{r_f(h)},
\quad r_f(h)=\sqrt{\|y\|^2/n+\varepsilon_f}.
$$

若域上有经认证的 $r_p\ge\underline r_p>0,r_f\ge\underline r_f>0$，以下界有效。
完全不约束前缀或隐藏状态时，总可取 $\underline r_p=\underline r_f=10^{-3}$。

### 4.2 GELU 的全实轴常数已认证

对本地近似 GELU

$$
G(x)=\tfrac{x}{2}[1+\tanh(\alpha(x+\beta x^3))],
$$

$\alpha,\beta$ 使用前述实数参考中固定的 binary64 常数。认证得到

$$
|G(x)|\le|x|,\qquad |G'(x)|\le6/5,\qquad |G''(x)|\le1
\quad\text{对所有实数 }x.
$$

证明实现将 $[0,8]$ 完整划分为 1024 个闭区间，用 96 位向外整数区间包含导数的全部取值，非点采样。
在尾部利用 $\operatorname{sech}^2(t)\le4e^{-2t}$，以及对 $x\ge8,k\le5$，$x^ke^{-2\alpha x}$ 单调递减的解析界。
负半轴使用 $G'(-x)=1-G'(x)$、$G''(-x)=G''(x)$。
核验器另用 2048 段复核标量不等式；这共享区间算术原语，不能称为第二套独立区间库。

### 4.3 矩阵参数系数

设已认证

$$
\|A_g\|_2\le g,\quad\|A_u\|_2\le u,\quad\|W_d\|_2\le w,
\quad\|a\|_\infty\le M_g,\quad\|b\|_\infty\le M_u,
\quad\|\gamma_f\|_\infty=\gamma.
$$

由于 $\|q\|\le\sqrt n$，可取 $M_g=\sqrt n\max_i\|(A_g)_{i,:}\|_2$ 的认证上界，$M_u$ 同理。
本轮复用已认证的合并矩阵范数与最大行绝对值和，两者较小者乘 $\sqrt n$ 提供合法坐标上界。

令

$$
K_1=w\left(\tfrac65 M_ug+M_gu\right),\qquad
K_2=w\left(M_ug^2+\tfrac{12}{5}gu\right),
\quad c_n=\frac2{\sqrt{3n}}.
$$

对 $y(q)=W_d[G(A_gq)\odot A_uq]$，链式法则与逐坐标乘积范数界给出

$$
\|D_qy\|\le K_1,\qquad \|D_q^2y\|\le K_2.
$$

具体地，二阶导数是一个 $G''(a)b(A_gv_1)(A_gv_2)$ 项和两个 $G'(a)(A_gv_1)(A_uv_2)$ 交叉项；分别用上述矩阵范数界得到 K2。

对于 RMSNorm，利用增广向量单位归一化的导数公式，

$$
\|Dq\|\le1/\underline r_p,\qquad
\|D^2q\|\le c_n/\underline r_p^2.
$$

同样，post norm 的两个界分别为 $\gamma/\underline r_f$、$c_n\gamma/\underline r_f^2$。
先后应用复合函数的一阶、二阶链式法则，得到

$$
\boxed{L_f\le\frac{A_1}{\underline r_p\underline r_f},}
$$

$$
\boxed{H_f\le
\frac{A_2}{\underline r_p^2\underline r_f}
+\frac{A_3}{\underline r_p^2\underline r_f^2},}
\tag{F2}
$$

其中完全由模型参数决定的系数为

$$
A_1=\gamma K_1,\qquad
A_2=\gamma(K_2+c_nK_1),\qquad
A_3=\gamma c_nK_1^2.
$$

实际计算使用 $\sqrt n,c_n$ 的向上有理数包络；所有其他运算使用精确有理数。
这使 F1 中的 L/H 从未指定常数变成可由参数检查的具体表达式。

### 4.4 输出幅度限制避免导数界无限放大 p

post RMSNorm 直接给出

$$
B_f=\sqrt n\|\gamma_f\|_\infty.
$$

更细地，四端点分支偏移落在

$$
\Delta_f\in4\sqrt n\,D_{\gamma_f}\mathbb B_2,
$$

其中 $\mathbb B_2$ 是单位 L2 球，因此任意固定方向 w 上还有

$$
|w^T\Delta_f|\le4\sqrt n\|D_{\gamma_f}w\|_2.
$$

这些界对所有前缀成立，但未利用真实 FFN 与前缀的全部依赖；不能因为得到了有限上限，就声称 p 足够小。

## 5. 实际参数核验结果与限制

第 1–14 层的所有 $A_1,A_2,A_3,B_f$ 及精确源参数证据保存于 [results.json](results.json)。
下面是取全空间分母下界 $10^{-3}$ 时的显示近似，L 列针对残差 T，即 $1+L_f$。

| FFN 层号 | 残差 L 的上界 | H 的上界 | F1 中四端点新增量上限 $4B_f$ |
|---|---:|---:|---:|
| 1 | $1.016\times10^{18}$ | $6.367\times10^{32}$ | 约 7485.64 |
| 9 | $2.023\times10^{16}$ | $3.897\times10^{30}$ | 约 485.00 |
| 14 | $3.382\times10^{17}$ | $2.508\times10^{32}$ | 约 2106.56 |

决定不等式的值是 JSON 中的有理数，表格只显示量级。
这些巨大数值来自保守的上界构造；它们不是实际最大增益或曲率的测量，不表示真实 FFN 或真实 p 有这么大。
导数项很大时，F1 会选择输出幅度上限，不能把它们直接相乘后当成有意义的恢复预算。

这一次确实把 FFN 对 p 的贡献变成了可核验参数界；但这组界还不足以认证小 p。
需要收紧的明确位置是：与真实前缀可达域相容的联合范围、覆盖所需路径的分母下界，以及 gate/up/down 之间的相关性。
先前第一层 ASCII **端点**的分母下界，不能未经证明就用于所有候选连线、均值移动路径或深层状态。

第 10、15 层最后 K/V 的观测度量，以及到原固定 A 的锚点差异仍须正确接入。
这里的隐藏向量单位、导数上界和 p 新增量都不是 LSB 数，也不是实际 NPU 误差预算。

## 6. 复核内容与复现

- [bound_ffn.py](bound_ffn.py)：全实轴激活界、14 层参数系数与输出上限。
- [verify.py](verify.py)：用另一解码路径重读 14 个 down projection，重算全部列平方和、绝对值和及范数；有理数复算系数，核对 70 个新源张量哈希。
- gate/up 使用此前已独立核验的 [合并矩阵证书](../gemma_fused_rho/verification.json)，本轮检查结果 SHA256 的绑定。
- [verification.json](verification.json)：当前核验结果。

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_prefix_rho_ffn/bound_ffn.py
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_prefix_rho_ffn/verify.py
```

模型是固定 W8 系数的实数参考；实际 BF16、KV8 和 NPU 执行偏差仍未认证。主结论保持：**p 的 FFN 递推组件已参数化并核验，完整深层小 p 尚未证明。**
