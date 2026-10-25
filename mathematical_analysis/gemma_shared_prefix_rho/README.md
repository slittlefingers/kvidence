# 回到 rho：共享前缀的成对影响界

日期：2026-10-17。模型：本地 Gemma 4 E2B W8，revision
03dcf209f3f549b4075e7191e77cf69b3d48e1b2。

## 1. 本轮证明了什么，尚未证明什么

本轮回到原目标：

$$
r_{uv}(S)=F_\theta(S,z_u)-F_\theta(S,z_v)-A(z_u-z_v),
\qquad
\sup_{S\in\mathcal S}\|r_{uv}(S)\|_2\le\rho_{uv}.
$$

$S$ 是两个候选共享的正确参考前缀；$F_\theta$ 是指定深层的 KV 观测；
$z_v$ 包含主 embedding 和实际 PLE 输入。原定义见
[参数分离证明 P1–P3](../parameter_separation_proof.md)。

**新的完成项：**

1. 证明一个保留共同前缀、显式包含当前 token K/V 变化的 attention 差异界。
2. 该界不要求小词表或低秩，也不需要额外乘上下文长度。
3. 代入实际第 10、15 层参数，当前 Key 差异的线性系数分别从 532、524 降到 266、262。
4. 对第一层实际参数的 60 对候选、480 个 head，用向外整数区间核验了更紧的非线性界；前缀没有被限制为这些候选。
5. 推导 RMSNorm 和 FFN 的精确两端差分关系，为后续保留共同变化提供表达式。
6. 将第一层 pre-FFN RMSNorm 的分母证书从 16 个候选扩展到整个明确的 ASCII 候选集，共 **143,955 个 token**；全部通过。全集合统一的隐藏范数下界为 **3.6**，这一处成对放大上界从 **116,500** 降至约 **1,268.22**。这不是最终 $\rho$ 的同比例改善。

**尚未完成：** 最终深层 KV 的 $\rho_{uv}$ 仍未得到足够紧的认证。
本轮实际检查中，经过输出投影和 post-attention RMSNorm 的粗范数传播仍不能给出正分离下界。
以下参考模型使用固定 W8 affine 系数和实数算子；实际 BF16/KV8/NPU 算术及重建缓存偏差另属执行误差预算。
没有将观测到的最大单维 2–4 LSB 改成每维独立的对抗扰动。

## 2. 第一条引理：softmax 取决于分数差的跨度

令 $p=\operatorname{softmax}(a)$、$q=\operatorname{softmax}(b)$，
只保留两者共同有效的 mask 位置。定义

$$
R=\max_j(a_j-b_j)-\min_j(a_j-b_j),
\qquad
\operatorname{TV}(p,q)=\frac12\sum_j|p_j-q_j|.
$$

则

$$
\boxed{\operatorname{TV}(p,q)\le\tanh(R/4).}
\tag{R1}
$$

这是对任意有限长度成立的数学界。所有位置共同增加同一个分数时 $R=0$，softmax 不变。
因此使用跨度比直接把每个位置的绝对变化相加更贴合实际依赖。

**证明。** 写 $L_j=p_j/q_j$，则 $\mathbb E_qL=1$。
若 $m=\min L_j$、$M=\max L_j$，有 $M/m=e^R=:c$。
由绝对值函数在区间端点之间的弦线包络，

$$
\operatorname{TV}(p,q)
\le\frac{(M-1)(1-m)}{M-m}
=\frac{c+1-cm-1/m}{c-1}.
$$

由 $cm+1/m\ge2\sqrt c$，

$$
\operatorname{TV}(p,q)
\le\frac{\sqrt c-1}{\sqrt c+1}
=\tanh(R/4).
$$

$R=0$ 的情况直接成立。两位置 logits 分别取
$(R/4,-R/4)$ 和 $(-R/4,R/4)$ 可达等号，所以常数不能在一般 logits 类上任意减小。证毕。

## 3. 第二条引理：共同的旧 Value 不应被重复计入

两个候选的旧值 $v_j$ 完全相同，只有当前位置的 Value 分别为 $v_u,v_v$。
所有 Value 范数不超过 $M_V$，并设

$$
\|v_u-v_v\|_2\le d_V\le2M_V,\qquad
\operatorname{TV}(p,q)\le T\le1.
$$

则两个 attention 输出满足

$$
\boxed{
\|o_u-o_v\|_2\le d_V+(2M_V-d_V)T.
}
\tag{R2}
$$

**证明。** 对每个位置匹配质量 $\min(p_j,q_j)$，总匹配质量为 $1-t$，其中 $t=\operatorname{TV}(p,q)$。
匹配到同一个旧位置的质量贡献完全抵消。匹配到当前 token 的质量至多 $1-t$，贡献不超过 $(1-t)d_V$。
未匹配质量在两侧均为 $t$，每侧 Value 范数至多 $M_V$，所以贡献至多 $2M_Vt$。
总和不超过 $d_V+(2M_V-d_V)t$。该式随 $t$ 单调不减，再用 $t\le T$。证毕。

这比“权重改变造成的最大影响，再加上一整份当前 Value 改变”更紧，因为两份影响不能同时占用全部概率质量。

## 4. 用实际 Q/K 关系控制跨度

对同一 head，设当前候选 query 为 $q_u,q_v$，旧 keys 相同并满足 $\|k_j\|\le M_K$。
令 score scale 为 $\tau$；本地 Gemma 实现取 $\tau=1$。
定义

$$
B=|\tau|M_Kd_Q,\qquad \|q_u-q_v\|_2\le d_Q,
$$

$$
s=\tau(q_u^\mathsf Tk_u-q_v^\mathsf Tk_v).
$$

各旧位置的 logit 差位于 $[-B,B]$，当前位置的差为 $s$，所以

$$
R\le\max(B,s)-\min(-B,s)
=B+\max(B,|s|).
\tag{R3}
$$

当前位置也参与 causal attention，不能从这个公式里删掉。
同一位置的 RoPE 是共同正交变换，求当前 self score 或 query 差范数时可以消去它；旧位置的旋转则由范数界覆盖。

如果尚未得到精确的 $s$，但已有

$$
\|k_u-k_v\|_2\le d_K,\qquad \|q_v\|_2\le M_Q,
$$

则

$$
|s|
\le|\tau|(M_Kd_Q+M_Qd_K).
$$

因此

$$
\boxed{
R\le|\tau|(2M_Kd_Q+M_Qd_K).
}
\tag{R4}
$$

关键在于：当前 Key 差异只改变一个位置的分数。旧 Keys 共享，不能把这一项当成所有旧位置都能独立改变。

结合 (R1)、(R2)，得到可用于任意词表的成对上界：

$$
\boxed{
\|o_u-o_v\|_2
\le
d_V+(2M_V-d_V)
\tanh\left(
\frac{|\tau|M_Kd_Q}{2}
+\frac{|\tau|M_Qd_K}{4}
\right).
}
\tag{R5}
$$

当有精确 self score 区间时，用 (R3) 替代 (R4) 更紧。
进一步用 $\tanh x\le x$，可得到简单但较松的形式：

$$
\boxed{
\|o_u-o_v\|_2
\le d_V+|\tau|M_VM_Kd_Q
+\frac{|\tau|M_VM_Q}{2}d_K.
}
\tag{R6}
$$

旧通用界的最后一个系数为 $|\tau|M_VM_Q$。
这个减半来自共享旧缓存的结构，而不是经验拟合。

单个旧位置加当前位置、两次执行的当前 logits 分别为 $c/2,-c/2$、共同旧 logit 为 0、当前和旧 Value 分别取 $+M_Vw,-M_Vw$ 时，
输出差为 $2M_V\tanh(|c|/4)$。因此当前 score 的线性系数 $M_V/2$ 在这个放松的 attention 类中是锐的。
这不是实际 Gemma token 或合法前缀的碰撞实例。

## 5. 代入实际 Gemma 第 10、15 层参数

这两层 head 维数均为 512，$M_V=\sqrt{512}$。
由 RMSNorm 参数直接得到
$M_Q=\sqrt{512}\|\gamma_Q\|_\infty$、
$M_K=\sqrt{512}\|\gamma_K\|_\infty$。
精确参数与认证值见 [results.json](results.json) 的 deep_layer_parameters。

| 层 | 新界中 $d_Q$ 的系数 | 新界中 $d_K$ 的系数 | 旧界中 $d_K$ 的系数 |
| --- | ---: | ---: | ---: |
| 10 | 30.75 | 266 | 532 |
| 15 | 31.25 | 262 | 524 |

这些表内数字是精确二进制有理参数的代数结果。因此

$$
\|o_u-o_v\|_2\le d_V+30.75d_Q+266d_K
\quad(\text{第 10 层}),
$$

$$
\|o_u-o_v\|_2\le d_V+31.25d_Q+262d_K
\quad(\text{第 15 层}).
$$

更紧的非线性式 (R5) 还保证上界不超过 $2\sqrt{512}$。
这没有证明深层的 $d_Q,d_K,d_V$ 已经足够小；它们仍须从共同前缀下两路隐藏状态的联合包络得到。
这里的 $d_K$ 是**两个候选当前 Key 的差异**，不是 LSB 数，也不是把旧缓存量化偏差忽略掉后的完整实际误差。
当两条执行的旧缓存数值不同，必须另加那部分偏差；不能直接套用“完全共享旧缓存”的减半结论。
本层 KV 在 attention 之前生成；第 10 层 attention 的界控制其残差输出及后续层的传播，不能直接把它当成本层新 KV 的分离保证。

## 6. 对实际候选进行参数认证：不运行具体前缀

第一层当前 Q/K/V 可直接由候选 embedding 与固定权重计算，与旧前缀内容无关。
选取数字 0–9 的 45 对，以及带前导空格的 the、a、cat、dog、yes、no 的 15 对，合计 60 对。
数字与英文之间的交叉配对不在本次数值清单中。
**前缀允许任意合法文本 token，不限制为这 16 个候选。**

对每对、每个 head：

1. 精确读取 W8 affine 系数和输入/输出归一化参数；
2. 使用整数点积和 96 位小数向外区间计算 $d_Q,d_K,d_V,s$；
3. 用 (R3) 与 (R2) 认证差异上界。

没有枚举前缀，没有运行上下文推理，也没有把成功率当成证明。
这一步是在给最终 $\rho$ 构造 attention 组件，不是重复做首层 KV 恢复攻击。

**数值结果：**

- 旧未截断的通用界约为 193.71–1120.86；加上简单的输出直径约束后，上界为 32。
- 新界在所有 480 项上均严格小于旧的已截断界，但很多改善极小，普通十进制显示仍为 32。
- 最大相对改善出现在数字 6 与 7 的第 3 个 head：从 32 降至约 **25.35155**，约降低 **20.8%**。
- 这些是最坏情况**上界的收紧**，不是实际误差下降，也不是成功率提升。

数据见 [完整结果](results.json)、[系数证书](coefficient_certificate.npz) 和
[认证程序](certify_attention.py)。
独立 110 位十进制实现交叉核对 480 项区间/界，另检查 8 项 RMS 差分恒等式和 4 项 softmax 常数，全部通过；
见 [算术核对](arithmetic_checks.json)。同一核验器还对第 8.1 节的 16 个分母证书进行了独立的精确有理数核验。这些有限数值交叉核对用于发现实现错误，通用命题依靠上述解析证明和整数区间；分母证书的多项式比较是精确核验。

## 7. 为什么还没有得到有效的端到端 rho

令 $a_u,a_v$ 为 output projection 后、post RMSNorm 前的向量。
简单传播会使用

$$
\|a_u-a_v\|_2\le
\|W_O\|_2\sqrt{\sum_h D_h^2}.
$$

然后对 post RMSNorm 使用全局敏感度
$\|\gamma_A\|_\infty/\sqrt\varepsilon$，
或其输出直径 $2\sqrt{1536}\|\gamma_A\|_\infty$。

实际第一层 $\|\gamma_A\|_\infty=101$。
两种界取较小者后，本轮所有候选的 attention 残差修正范数界仍为约 **7916.75**。
而初始 embedding 基准距离只约 **35.67–68.55**。
所以这条完整范数传播没有得到任何正的第一 attention 残差分离下界，尚不必谈 FFN、PLE 和深层 KV。

这表示**所使用的传播证书过松**；没有证明真实 $\rho$ 接近 7916.75。
也不推翻之前的第一 attention 定向分离证书：那份证书使用各向异性的投影约束，而这里尝试界定完整修正向量的 L2 范数，两者不同。
输出投影的 Frobenius 上界、归一化的全局最小分母和最大 gamma 都可能造成松弛；本轮没有证明其中某一个是唯一原因。

## 8. 继续处理归一化：精确的两端关系

不必先把归一化替换成统一的 $1/\sqrt\varepsilon$。
定义

$$
R_\gamma(x)=D_\gamma x/r_x,\qquad
r_x=\sqrt{\|x\|_2^2/d+\varepsilon}.
$$

利用

$$
r_x-r_y=\frac{(x+y)^\mathsf T(x-y)}{d(r_x+r_y)},
$$

可以直接整理出恒等式

$$
R_\gamma(x)-R_\gamma(y)
=S_\gamma(x,y)(x-y),
$$

$$
\boxed{
S_\gamma(x,y)=D_\gamma\left[
\frac{r_x^{-1}+r_y^{-1}}{2}I
-\frac{(x+y)(x+y)^\mathsf T}
{2d\,r_xr_y(r_x+r_y)}
\right].
}
\tag{R7}
$$

它仅依赖两端，没有要求连接两端的全部连续隐藏状态都合法可达。
进一步，对无 gamma 的归一化，有精确恒等式

$$
\left\|\frac{x}{r_x}-\frac{y}{r_y}\right\|_2^2
=\frac{\|x-y\|_2^2}{r_xr_y}
-\frac{d(r_x-r_y)^2}{r_xr_y}
-\frac{d\varepsilon(r_x-r_y)^2}{r_x^2r_y^2}.
$$

因此

$$
\boxed{
\|R_\gamma(x)-R_\gamma(y)\|_2
\le
\frac{\|\gamma\|_\infty}{\sqrt{r_xr_y}}\|x-y\|_2.
}
\tag{R8}
$$

如果能从参数和联合状态约束认证两端分母的正下界，就可用它替换全局的 $\sqrt\varepsilon$。
只知道差向量小或值域满维都不足以认证这个下界；任意放松域中的两端仍可能靠近零。
post-attention RMSNorm 的原始输入分母仍未得到有用的统一下界。不过，对紧随其后的 pre-FFN RMSNorm，可以利用带 embedding 锚点的椭球约束得到实际正下界。

### 8.1 从实际 embedding 与 gamma 认证 pre-FFN 分母

第一 attention 残差后的隐藏状态满足

$$
h_v=\sqrt d\,e_v+D_{\gamma_A}t,\qquad \|t\|_2\le\sqrt d.
$$

这对任何旧前缀成立，不要求词表低秩。
令所有 $\gamma_{A,j}\ne0$，并定义可直接从一行 embedding 与参数计算的量

$$
\alpha_v^2=\sum_j\frac{e_{v,j}^2}{\gamma_{A,j}^2},
\qquad
\beta_v^2=\sum_j\frac{e_{v,j}^2}{\gamma_{A,j}^4}.
$$

如果 $\alpha_v>1$，则

$$
\boxed{
\|h_v\|_2\ge
L_v:=\frac{\sqrt d(\alpha_v^2-\alpha_v)}{\beta_v}>0.
}
\tag{R10}
$$

**证明。** 取 $w_j=e_{v,j}/\gamma_{A,j}^2$。
有 $\|w\|=\beta_v$、$\|D_{\gamma_A}w\|=\alpha_v$，于是

$$
w^\mathsf Th_v
\ge\sqrt d\,\alpha_v^2-\sqrt d\,\alpha_v
=\sqrt d(\alpha_v^2-\alpha_v).
$$

再由 $w^\mathsf Th_v\le\beta_v\|h_v\|$ 得证。
这里的方向用于认证隐藏范数下界，不要求它与 attention 影响空间正交，也没有回到低秩词表假设。

由此，该候选的 pre-FFN RMSNorm 分母满足

$$
r_v\ge\underline r_v:=
\sqrt{L_v^2/d+\varepsilon}.
$$

再代入两端关系 (R8)，对同一或不同允许前缀的候选对，均有

$$
\|N_{\rm pre}(h_u)-N_{\rm pre}(h_v)\|_2
\le
\frac{\|\gamma_{\rm pre}\|_\infty}
{\sqrt{\underline r_u\underline r_v}}
\|h_u-h_v\|_2.
\tag{R11}
$$

**实际参数证书：** 本次 16 个候选均通过 $\alpha_v^2>1$。
认证的最小隐藏范数下界约为 3.989443，最小分母下界约为 0.1017976，
而全局界只使用 $\sqrt\varepsilon=0.001$。
实际 $\|\gamma_{\rm pre}\|_\infty=116.5$，所以覆盖本次候选的成对放大上界从 116,500 降至约 **1,144.43**，收紧约 **101.8 倍**。

这覆盖这些候选的全部允许旧前缀，包含未枚举的英文、数字等前缀。
本小节的 16 行证书本身只覆盖这些当前候选；扩展时可逐行检查条件 $\alpha_v>1$，无需枚举这些候选之前的所有序列。下一小节给出对完整已声明 ASCII 集合的检查。
它也没有给出 FFN 整体的收缩性或最终 KV 分离保证。

证书保存在 results.json 的 pre_FFN_endpoint_certificates。
独立核验先用精确有理数重算 $\alpha_v^2,\beta_v^2$。
对宣称的下界 $l_v$，设 $a=\alpha_v^2$、$c=l_v^2\beta_v^2/d$；
检查

$$
a>1,\quad l_v>0,\quad
a^2-a-c\ge0,\quad
(a^2-a-c)^2\ge4ac
$$

即可严格推出 $\sqrt c+\sqrt a\le a$，即 $l_v\le L_v$。
分母下界则用平方不等式直接核验。这里不依赖浮点开方的舍入方向。

### 8.2 已扩展到全部 143,955 个声明的 ASCII 候选

候选表沿用 [gemma4_ascii_candidates.json](../gemma4_ascii_candidates.json)。
其明确规则是：非特殊 token、单 token 解码为可打印 ASCII 或 tab/CR/LF，且 canonical 单 token 编码/解码往返一致。
它包含英文、数字、标点等可用 token，但“正常英语”不是这个集合的数学定义。
其余语言、特殊模板 token，以及不满足这个往返条件的 ID 不在当前候选认证集合内。
**旧前缀的 token 仍不受这个 ASCII 条件限制。**

检查整个 embedding 表中这些行后得到：

| 项目 | 认证结果 |
| --- | ---: |
| 当前候选总数 | 143,955，全部通过 |
| $\alpha_v^2$ 的统一下界 | 大于 2.8296158 |
| FFN 输入隐藏范数统一下界 | $\|h_v\|_2\ge18/5=3.6$ |
| pre-FFN RMSNorm 分母统一下界 | 大于 0.0918613 |
| 成对放大上界 | 小于 1268.217 |
| 原全局放大上界 | 116,500 |
| 这一上界的收紧比例 | 约 91.86 倍 |

因此，在本参考模型中，对这个集合里的任意两个当前候选、任意允许文本前缀，均可用

$$
\boxed{
\|N_{\rm pre}(h_u)-N_{\rm pre}(h_v)\|_2
\le1268.217\|h_u-h_v\|_2.
}
\tag{R12}
$$

其更精确的有理数常数保存在 [全表结果](ascii_floor_results.json)。
这是通过逐行参数条件覆盖所有候选及全部前缀；没有枚举 143,955 个候选之间的全部配对，更没有枚举所有前缀。

**算术保证。** 为避免对整个表执行大量大整数除法，程序分块精确解码 W8 的二进制有理系数，再对正平方和使用显式 IEEE binary64 相对误差包络。
每个非零中间值均检查为正常有限数；对 1536 维求和，

$$
\gamma_{1552}=\frac{1552u}{1-1552u}<2^{-30},
\qquad u=2^{-53}.
$$

使用远大于所需舍入界的 $2^{-30}$ 并向外取整，得到每行的
$\underline a_v\le\alpha_v^2$ 和 $\beta_v^2\le\overline b_v$。
$a-\sqrt a$ 在 $a>1$ 上单调递增，因此可以用这些单侧包络证明共同下界。
随后对 **全部 143,955 行**执行第 8.1 节的精确有理数多项式核验，认证共同的 $18/5$。

独立核验器再次检查完整候选覆盖、全部有理数不等式及分母下界，
并从模型重新读取极端/分布式若干行，用精确有理数检查源包络。
这些源行抽查只检查实现，其他行的包络正确性仍由明确的 IEEE 误差推导保证。
所以这是**在声明算术模型下的计算辅助证明**，不是声称整个提取过程都使用了任意精度整数。

文件：

- [全表证书生成](certify_ascii_floor.py)
- [完整系数包络证书](ascii_floor_certificate.npz)
- [独立核验程序](verify_ascii_floor.py)
- [独立核验结果](ascii_floor_verification.json)

这个推广并没有证明 FFN 整体可逆、后续各层都有同样的分母下界，或实际量化执行能够恢复全部 token。
它完成的是原始 $\rho$ 推导里的一处实际参数约束，后续仍需保留 FFN、PLE、attention 与最终 KV 投影的成对相关性。

## 9. FFN 和 PLE 的成对关系可以显式保留

对 FFN，令 $n_u=N_{\rm pre}(h_u)$、$n_v=N_{\rm pre}(h_v)$，
$g_i=W_gn_i$、$t_i=W_un_i$，$G$ 为实际参考 GELU。
定义对角割线矩阵 $D_G$，其第 k 项为
$(G(g_{u,k})-G(g_{v,k}))/(g_{u,k}-g_{v,k})$；
两端相等时取导数。
由乘积差的精确恒等式，

$$
G(g_u)\odot t_u-G(g_v)\odot t_v
=\left[
\operatorname{diag}\left(\frac{t_u+t_v}{2}\right)D_GW_g
+\operatorname{diag}\left(\frac{G(g_u)+G(g_v)}{2}\right)W_u
\right](n_u-n_v).
$$

将两次 RMSNorm 都写成 (R7)，即可把 FFN 残差的差写成

$$
\Delta h_{\rm afterFFN}
=
\left(I+
S_{\rm post}W_d
\left[
\operatorname{diag}(\bar t)D_GW_g+
\operatorname{diag}(\overline{G(g)})W_u
\right]S_{\rm pre}\right)\Delta h_{\rm beforeFFN}.
\tag{R9}
$$

这些矩阵取自同一对端点，不能把它们当成彼此独立的自由矩阵。
它允许研究实际候选差方向上的变化，不要求整个 FFN 连续空间可逆。

PLE 也必须保留 token 自身的差异。
若 gate 输出是 $G(W_ph_i)$、token 对应的 PLE 输入为 $w_i$，则其乘积差为

$$
\operatorname{diag}(\bar w)D_GW_p\Delta h
+\operatorname{diag}(\overline{G(W_ph)})\Delta w.
$$

第二项由候选的 PLE 特征差决定，不能被省略为“共同前缀会抵消”。

这些恒等式给出了继续认证 $\rho$ 的具体表达式，但不是已认证的 $\rho$ 数值。
下一处实质缺口是：对实际前缀产生的成对端点，认证这些相关量的联合范围，再传播到目标 KV 投影。
如果再次把所有相关量拆成独立全局最大值，本轮已经看到的过松问题会重新出现。

后续已进一步把 FFN 的前后归一化、以及最终 KV 的 input/head norm 精确合并，
见 [合并归一化与成对 rho 表达式](../gemma_fused_rho/README.md)。
该推导保留两路候选的共同尺度与 Gram 交叉项，直接给出原目标 $\|r_{uv}(S)\|^2$ 的表达式；
同时认证了 34 个实际矩阵的合并缩放范数界。它没有完成联合可达域上的最大值认证。

## 10. 复现

在工作区根目录运行：

    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_shared_prefix_rho/certify_attention.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_shared_prefix_rho/check_arithmetic.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_shared_prefix_rho/certify_ascii_floor.py
    kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python kvidence/mathematical_analysis/gemma_shared_prefix_rho/verify_ascii_floor.py
