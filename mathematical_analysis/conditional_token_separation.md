# 条件性 token 分离定理：输入相差多少，模型保留多少，误差允许多少

日期：2026-10-17。本文是当前证明主线的统一入口。

## 1. 研究目标与当前结论

我们要证明的是：**给定模型参数和允许的 token 特征，当一对 token 的间距、模型对其差异的保留程度以及数值误差满足明确条件时，这两个 token 能被指定的 KV 匹配规则完全区分。**

定理可以对整个声明前缀域成立，同时仅认证其中满足条件的 token 对或候选子集。无需先假定实际 Gemma 的全部 token 都能通过同一个阈值。

本文把已经完成的局部证明接入这一目标，并补齐逐算子偏移递推和 FFN 参数条件的表达。以下条件定理已经给出数学证明；本地 Gemma 4 E2B W8 的若干局部参数条件已有证书，深层完整条件及自然 BF16/KV8/NPU 执行预算仍待认证。

**2026-10-18 更新：** [完整 p 参数计算器](parameter_p/README.md)现已将 attention、FFN、PLE、层缩放、最终 KV 和共享关系接通。它固定一个明确的残差线性基准 A，从 checkpoint 自动输出对整个词表和全部声明前缀成立的保守 $\rho$ 上界；不需要先证明 $\rho$ 小。恢复条件是否通过、上界是否足够紧以及硬件执行误差仍是另外的问题。

直观的证明链是：

$$
\text{有限 token 特征差}
\longrightarrow
\text{模型基准保留的差异}
\longrightarrow
\text{扣除已认证的非线性／前缀偏移}
\longrightarrow
\text{与计算误差比较}.
$$

## 2. 先把对象、单位和量词固定

固定模型参数 $\theta$、候选 ID 集合 $\mathcal V$、观测层、位置及允许前缀状态域 $\mathcal S$。
一个前缀状态 S 包含参考模型需要的历史 KV、位置和 mask；同一候选比较使用相同 S。
不同长度可以分别建域，或在明确的最大长度下认证一致常数。

令 $z_v$ 表示 token v 的完整输入特征。对本地 Gemma，它包含缩放主 embedding 和目标 KV 之前各层实际使用的 PLE 特征。
所有特征块的缩放须固定：改变坐标尺度会改变下文的输入距离和保留系数，但不会凭空改善实际可分性。

令

$$
F_S(v)=F_\theta(S,z_v)\in\mathbb R^{d_{KV}}
$$

为指定深层的干净参考 KV，统一使用同一种 L2 度量。
模型使用固定 affine W8 系数和声明的实数参考非线性；实际反量化、BF16、KV8、NPU 运算及重建历史偏差通过误差预算进入。
不能把不同的实际数值历史当作“相同 S”而忽略差异。

观测和重算候选记为

$$
y=F_S(u)+e_T,\qquad
\widehat F_S(v)=F_S(v)+e_{C,v},
$$

其中真 token 为 u，且在整个声明域内有

$$
\|e_T\|_2\le b_T,\qquad
\|e_{C,v}\|_2\le b_{C,v}.
$$

这些预算必须包含传播到**当前被比较向量**的误差，不能只放入某个内部算子的误差。
误差可以相关，无需独立、随机或零均值；一个经过证明的外包络可用于下面的充分条件。
观测到最大坐标差约 ±4 LSB，本身还不提供上述全域 L2 上界。

我们研究的规则是：在实际查询的候选中，选取 $\|y-\widehat F_S(v)\|_2$ 最小者。真候选必须被包含且被查询。

## 3. 主定理：从特征差和模型参数得到分离条件

### 3.1 参数条件

选择一个由模型参数和声明域确定的固定线性基准 A。它可以来自局部算子的基准矩阵乘积，或经过认证的线性化；不能只用拟合误差冒充全域保证。
对每一对不同候选定义

$$
q_{uv}=z_u-z_v,\qquad
D_{uv}=\|Aq_{uv}\|_2,
$$

$$
r_{uv}(S)=F_S(u)-F_S(v)-Aq_{uv}.
$$

假设参数认证给出

$$
\|r_{uv}(S)\|_2\le\rho_{uv}
\qquad\text{对所有 }S\in\mathcal S.
\tag{C1}
$$

则干净候选间距满足

$$
\boxed{
\|F_S(u)-F_S(v)\|_2
\ge\max\{0,D_{uv}-\rho_{uv}\}.
}
\tag{C2}
$$

### 3.2 精确恢复条件与证明

对每个错误候选 v，如果

$$
\boxed{
D_{uv}-\rho_{uv}
>2b_T+b_{C,u}+b_{C,v},
}
\tag{C3}
$$

则最近邻匹配一定严格优先选择真候选 u。

**证明。** C2 是反三角不等式。正确候选的距离满足

$$
\|y-\widehat F_S(u)\|_2\le b_T+b_{C,u}.
$$

错误候选的距离满足

$$
\|y-\widehat F_S(v)\|_2
\ge D_{uv}-\rho_{uv}-b_T-b_{C,v}
>b_T+b_{C,u}.
$$

因此不存在与错误候选的距离并列或反转。证毕。

若统一使用 $b_{C,v}\le b_C$，充分条件化为

$$
\boxed{D_{uv}>\rho_{uv}+2(b_T+b_C).}
\tag{C4}
$$

两侧各有误差上界 b 时阈值包含 4b；只有一侧误差为 b、另一侧精确时才是 2b。
这只是充分条件，未通过或等号成立都不证明实际匹配失败。

### 3.3 从一对 token 推广到候选集合

如果集合 $\mathcal V_*$ 内的每一对不同 token 都满足 C3，且真实 token 和全部竞争候选都在 $\mathcal V_*$ 内，那么该集合在声明前缀域上可被保证区分。

只检查真 token 对少数错误候选，不能保证与未检查候选也可分；从词表移除未通过的候选，也必须有输入域依据，确保真实 token 不被排除。
逐对检查有限表行可以用于构造证书，不等于枚举全部可能前缀。

**整段恢复还需要归纳条件：** C3 在每个允许位置的正确前缀路径上成立，缓存误差更新继续满足预算，且每一步都覆盖真候选。
此时从起始状态逐步应用单步定理即可归纳恢复整段。只有当前一个位置的证书，不足以完成这一步。

## 4. 怎样写成“token 至少相距多少”的阈值

令

$$
a_{uv}=\|z_u-z_v\|_2>0,\qquad
\kappa_{uv}=\frac{\|A(z_u-z_v)\|_2}{a_{uv}}.
$$

$\kappa_{uv}$ 表示**模型基准在这一对实际 token 差方向上**保留了多少差异。
若参数分析能够认证

$$
\rho_{uv}\le\beta_{uv}a_{uv}+p_{uv},
\qquad\beta_{uv},p_{uv}\ge0,
$$

并且 $\kappa_{uv}>\beta_{uv}$，那么

$$
\|F_S(u)-F_S(v)\|_2
\ge(\kappa_{uv}-\beta_{uv})a_{uv}-p_{uv}.
$$

于是一个明确的输入间距充分条件为

$$
\boxed{
a_{uv}>
\frac{p_{uv}+2(b_T+b_C)}{\kappa_{uv}-\beta_{uv}}.
}
\tag{C5}
$$

这里 $p_{uv}$ 是模型参考差的加性偏移预算，不是观测误差；其前缀影响已在整个 $\mathcal S$ 上取界。
$\kappa,\beta,p$ 必须由固定模型、候选对和状态域认证，不能为了得到正结论倒选数值。
它们可以随 token 对变化，所以 C5 通常是**逐对阈值**，不是“任何 embedding 只要距离大于某个数都行”。
若能对一整个候选集合认证统一 $\kappa>\beta$ 和 p，再用该集合最小输入距离，就可得到统一阈值。

原始 embedding 距离相同的两对 token，可能沿着模型完全不同的压缩方向。矩形 A 有核空间也不妨碍某个有限候选集合通过条件；需要检查有限差向量，而非要求整个连续空间可逆。
若两个 ID 的完整特征完全相同且没有其他 ID 依赖分支，它们在这个参考模型下无法区分，应单独处理，不能除以 $a_{uv}=0$。

### 4.1 一个纯数学示例

下面数字只用于说明条件，不是 Gemma 的测量值。假定已认证

$$
\kappa=0.8,\quad\beta=0.2,\quad p=1,
\quad b_T=b_C=0.5.
$$

则输入距离需严格大于 $[1+2(0.5+0.5)]/(0.8-0.2)=5$。

| 输入距离 a | 认证的干净 KV 间隔下界 $0.6a-1$ | 与阈值 2 比较 | 结论 |
|---:|---:|---|---|
| 6 | 2.6 | 严格大于 | 保证这对候选正确区分 |
| 5 | 2 | 等于 | 这份条件不能保证严格唯一 |
| 4 | 1.4 | 小于 | 未获得保证，不等于实际失败 |

## 5. 怎样从模型算子得到 rho，而不是把它留作空假设

### 5.1 一条可检查的逐算子构造

将固定位置的参考计算写成 $G_L\circ\cdots\circ G_1$。
必要时使用增广状态：同时携带当前隐藏、后续需要的 token 特征和共同历史 S，直到最终输出所选 KV。
这样 PLE 的 token 依赖和 attention 的历史依赖都是算子输入的一部分；中间未使用的特征通过恒等分量传递。

初始两路状态差为 $B_0q_{uv}$，其中 $B_0$ 将完整 token 特征插入对应坐标，而共同历史的差为零。
这里初始特征已经定义为模型实际使用的缩放值；不能省略缩放再假设 $B_0$ 不变。

对算子 $G_\ell$，选固定矩阵 $M_\ell$。要求有经认证的域 $\Omega_\ell$，覆盖所有允许前缀、这对候选进入该算子的两个端点及其连接线段，并有

$$
\sup_{X\in\Omega_\ell}
\|JG_\ell(X)-M_\ell\|_2\le\epsilon_\ell.
\tag{C6}
$$

域可以是严格包含可达集合的包络。无需每个点都可达，但必须确实覆盖这些线段；只有真实端点的界不足以支持积分。
对固定历史的子算子，也可以直接对所有 S 一致认证隐藏导数偏差；若使用增广状态，则包含全部需要的导数块。

定义基准差和误差递推

$$
c_0=B_0q_{uv},\qquad E_0=0,\qquad
c_\ell=M_\ell c_{\ell-1},
$$

$$
\boxed{
E_\ell=(\|M_\ell\|_2+\epsilon_\ell)E_{\ell-1}
+\epsilon_\ell\|c_{\ell-1}\|_2.
}
\tag{C7}
$$

则可以取

$$
\boxed{
A=M_L\cdots M_1B_0,\qquad\rho_{uv}=E_L.
}
\tag{C8}
$$

**证明。** 令真实进入第 $\ell$ 个算子的两路差为 $\Delta_{\ell-1}$。
沿两端连线积分，由 C6 得

$$
\Delta_\ell=M_\ell\Delta_{\ell-1}+\xi_\ell,
\qquad\|\xi_\ell\|_2\le\epsilon_\ell\|\Delta_{\ell-1}\|_2.
$$

若 $\|\Delta_{\ell-1}-c_{\ell-1}\|\le E_{\ell-1}$，则

$$
\begin{aligned}
\|\Delta_\ell-c_\ell\|_2
&\le\|M_\ell\|_2E_{\ell-1}
+\epsilon_\ell(\|c_{\ell-1}\|_2+E_{\ell-1})\\
&=E_\ell.
\end{aligned}
$$

从 $E_0=0$ 归纳即得 C8。证毕。

这个构造只要求可认证的导数**偏差上界**，不要求每个矩阵方阵、可逆，也不要求每层都是收缩映射。
通过与否由有限候选的 $\|Aq_{uv}\|-E_L$ 决定。
不过算子范数包络可能很松；本地 Gemma 全部相关算子的 C6 尚未完成足够紧的认证，不能把公式存在当成证书已经通过。

若各 $\epsilon_\ell,M_\ell$ 对声明候选域统一，可以进一步用 C7 给出比例型界 $E_L\le\beta\|q_{uv}\|$；代入 C5 时取 p=0。
若在某个中间位置开始，且已有初始误差界 $E_0>0$，同一递推仍成立，该初始误差还须按递推传播。

### 5.2 FFN 的条件具体落在哪些模型参数上

对当前本地 dense FFN 分支，定义

$$
n=N_{\gamma_p}(h),\quad a=W_gn,\quad b=W_un,
\quad u=W_d[G(a)\odot b],\quad f(h)=N_{\gamma_f}(u).
$$

FFN 残差算子为 $T(h)=h+f(h)$。链式法则给出

$$
Jf(h)=JN_{\gamma_f}(u)\,W_d
\left[
\operatorname{diag}(b\odot G'(a))W_g
+\operatorname{diag}(G(a))W_u
\right]JN_{\gamma_p}(h).
\tag{C9}
$$

这里对 $r(x)=\sqrt{\|x\|_2^2/d+\varepsilon}$，

$$
JN_\gamma(x)=D_\gamma
\left(\frac I{r(x)}-\frac{xx^T}{d\,r(x)^3}\right).
\tag{C10}
$$

所有矩阵、gamma、epsilon 和 G 都由固定模型决定。
在覆盖所需输入线段的状态域上，认证

$$
\|Jf(h)-B\|_2\le\beta_f
$$

后，就可在 C6–C7 中令 $M=I+B$、$\epsilon=\beta_f$。
若紧接 scalar s，则对应乘上 s，偏差界乘 $|s|$；若中间还有 PLE，必须保留其算子顺序或使用第 6 节的投影结构，不能跨过去合并成遗漏 PLE 的计算。

一种完全明确的认证方法是：根据状态域、已认证的 norm 分母下界和实际权重，向外传播 C9–C10 的矩阵区间。
若最终有逐元素包络 $|Jf(h)-B|\le R$，其中 $R_{ij}\ge0$，则

$$
\|Jf(h)-B\|_2\le\|Jf(h)-B\|_F\le\|R\|_F,
$$

可取经过向上舍入认证的 $\beta_f\ge\|R\|_F$。
这给出了可核验的参数条件，但没有保证这种粗矩阵区间会足够紧。实际全域 FFN 包络目前尚未算到能通过最终条件。
先前的全空间非单射和方向收缩见证说明不能无条件假设正保留率；它们不妨碍某些有限候选对、明确状态域通过 C3。

### 5.3 更紧的已有结果怎样替换粗界

前述 C7 是一种完整、保守的构造，并非必须在每层使用。

目前可直接运行的完整实现是 [parameter_p](parameter_p/README.md)：使用 post norm 全域幅度包络闭合各层，再将真实 KV 相对明确线性 A 的归一化偏移完整计入。它是 C1 的一个具体参数证书，不要求全部 C6 导数偏差先得到紧界。本地默认 A 下，第 10、15 层 K+V 的统一上界分别不超过 55.12、45.36（实值标准 L2）；当前取最小值的是输出范围包络，不能解释成实测误差或紧的前缀敏感度。

当前针对 p／rho 本身的推进见 [FFN 的全前缀参数递推](gemma_prefix_rho_ffn/README.md)：没有选择 token 对，已认证第 1–14 层实际 FFN 的导数上界系数，并将共同位置变化、输入差偏移和输出幅度上限合成一条 p 递推。全空间常数仍很松；这不是最终深层小 rho 已通过的结论。数字 1、2 附近输入球的另一次旁支尝试未被作为 p 的认证依据。

- [共同位置／差值递推](gemma_joint_prefix_geometry/README.md)：若分别认证均值偏移 C、差值误差 E 和路径上的 Jacobian/Hessian 界，可用 $E'\le LE+H\|d_0\|C$ 保留两路相关性。它依赖不同的参考对及路径条件，不能与 C7 的符号未经转换直接混用。
- [共享前缀 attention 界](gemma_shared_prefix_rho/README.md)：利用相同历史 KV、softmax 分数差跨度和概率重叠，直接约束两候选的 attention 差，减少把共同历史算成两次独立最坏变化。
- [合并归一化](gemma_fused_rho/README.md)：用完整 FFN／KV 恒等式保留抵消与 Gram 交叉项；已认证的矩阵范数上界可以作为区间和敏感度界的常数来源。
- [最终 KV 的联合能量条件](gemma_joint_prefix_geometry/README.md)：可直接从分母范围、共同位置和候选差的能量交叉项给出最终 $\rho$，不必把每个粗导数上界连乘。

替换后的每个界仍须覆盖同一声明域、同一输出度量和同一对候选。

## 6. PLE 投影作为另一条可用的分离证书

最近一层 PLE 的已证结构是 $h=s[x+Cz]$，其中 x 是该层 FFN 残差后的状态。
目标 KV head 在 RoPE 前为 $U(h)=Th/t(h)$。
令 Q 为 $TC$ 列空间的正交补投影，$L=QT$，则

$$
QU(h)=\frac{sLx}{t(h)}.
$$

当 $s>0,Lx\ne0$ 时，其单位方向就是 $Lx/\|Lx\|$。
本地第 10、15 层 K、V 各有 256 维这样的投影，已由实际参数的精确秩证书核验。

若在整个声明域、对当前候选对认证

$$
\|Lx_u\|,\|Lx_v\|\ge\mu>0,\qquad
t(h_u),t(h_v)\le\bar t,
$$

以及

$$
\left\|\frac{Lx_u}{\|Lx_u\|}-\frac{Lx_v}{\|Lx_v\|}\right\|\ge\delta,
$$

那么

$$
\boxed{\|U(h_u)-U(h_v)\|\ge\frac{s\mu\delta}{\bar t}.}
\tag{C11}
$$

证明：投影端点幅度至少 $r=s\mu/\bar t$，且
$\|r_un_u-r_vn_v\|^2=(r_u-r_v)^2+r_ur_v\|n_u-n_v\|^2\ge r^2\delta^2$；原向量距离至少为正交投影距离。
K 应撤销共同位置 RoPE 或同步旋转投影。多个不重叠 K/V 分块的平方下界可以相加。

将 C11 的下界代替 C3 中的 $D-\rho$，便得到同样的匹配保证。
这条路线允许差向量随前缀改变方向，不要求贴近固定 A；但必须另外认证 $\mu,\delta,\bar t$。
它只消去紧邻前一层 PLE 的直接增量对投影方向的影响，早先 PLE 的间接作用和分母幅度仍在。

若两种路线都对相同 KV 度量给出下界，可以取较大的一个；不能将它们相加，也不能把一个已知局部间隔直接乘投影维数。

## 7. 已有证明对应当前链条的什么位置

| 已有结果 | 严格已经得到的结论 | 在本定理中的用途／尚缺内容 |
|---|---|---|
| [数字初始残差证书](gemma_digit_parameter_certificate/README.md) | 实际 45 对数字在第一 attention 残差处，普通 L2 统一下界大于 1.05756；全参考前缀有效 | 提供候选相关的初始域和分离约束；尚非深层 KV 的 $D-\rho$ |
| [第一 attention 方向证书](gemma_joint_prefix_geometry/README.md) | 同样 45 对的固定加权单位方向距离统一大于 0.0436009 | 初始方向约束；不能直接沿用到 FFN 后，也不是普通 KV 距离 |
| [ASCII 分母证书](gemma_shared_prefix_rho/README.md) | 143,955 个明确 ASCII 当前候选的第一 pre-FFN 分母统一大于 0.0918613 | 可收紧局部敏感度；连接线段是否同样满足该下界需单独检查 |
| [合并 norm 参数证书](gemma_fused_rho/README.md) | 34 个实际矩阵有经独立核验的合并缩放范数上界 | 是 rho／导数包络的参数输入，不是完整 rho 已通过 |
| [PLE 固定投影](gemma_first_block_joint/README.md) | 11 个非零子式认证；目标层 K/V 各保留 256 维投影 | C11 的结构已成立；深层 $\mu,\delta,\bar t$ 联合条件待认证 |
| FFN 与共同偏移见证 | 连续放松中不能自动保持原方向下界；真实前缀可达性未证 | 约束定理假设，提醒保留有限候选与可达域；不否定条件定理 |
| [长度一致误差传播](length_uniform_proof.md) | 在其声明算子假设下，按网络深度建立误差递推 | 用于提供 $b_T,b_C$ 与整段归纳；实际硬件的全部局部误差界待认证 |

这张表区分了“数学条件定理已证”“实际局部条件已核验”和“深层数值条件尚未认证”。
目前不能给出 Gemma 某一深层实际 token 对已经通过 C3 的新名单，也不能把 ±4 LSB 填成未认证的 $b_T=b_C=4$。

## 8. 从 LSB 到预算时必须保留哪些信息

若已在声明域上证明两条对应向量每坐标的**实值**差满足 $|e_j|\le c_j$，则可取 $\|e\|_2\le\sqrt{\sum_jc_j^2}$。
这个外包络不要求实际误差能在每维同时取端点；它可能很保守，条件未通过时不能据此断言自然误差会失败。

对于共同 scale $s_j$、共同偏置的对应量化码，码差至多 $r_j$ 给出 $c_j=s_jr_j$。
不同 scale／bias 时应比较完整反量化值；还需区分“两个执行结果之间的差”与“它们各自到同一参考的误差”，前者不能自动拆成两个更小的单侧预算。
若另有经过认证的相关结构或整体 L2 上界，应直接使用更紧预算。
逐维 RMS 与标准 L2 相差 $\sqrt{d_{KV}}$，最大单维误差也不能当作标准 L2。

## 9. 最终可向导师表述的结论

**中文：** 对已知模型和有限 token 特征集，本文建立候选对的条件性稳健分离定理：由模型基准映射保留的特征差，减去经参数及状态域认证的非线性／前缀偏移，得到干净 KV 间隔下界；当该下界超过目标和重算候选所需的误差阈值时，L2 最近邻能够精确区分该候选对。给出逐算子认证构造、真实 Gemma FFN 的参数表达，以及消去邻接 PLE 直接影响的投影结构。本地 Gemma 已通过若干局部参数证书；完整深层条件和自然执行误差预算尚未全部认证。

**English:** For a known model and a finite token-feature set, we establish a conditional robust-separation theorem for candidate pairs. A parameter- and domain-certified bound on nonlinear and prefix-dependent deviations is subtracted from the feature difference retained by a fixed model-based reference map, yielding a lower bound on clean KV separation. If this bound exceeds the required target and candidate error budget, L2 nearest-neighbor matching identifies the pair correctly. We provide an operator-wise certificate construction, the parameter-level Jacobian expression for the actual Gemma FFN, and a projection that eliminates the adjacent PLE branch's direct contribution to projected directions. Several local parameter certificates have been verified for the local Gemma checkpoint; the complete deep-layer conditions and natural execution-error budgets remain unverified.

这些定理采用反三角不等式、线段积分、链式法则和归纳法；本文不声称这些数学工具本身是新的。
可用于实际模型的紧证书和明确适用域，是后续需要完成的工作。
