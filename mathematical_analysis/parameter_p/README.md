# 从模型参数直接计算完整的 p 上界

日期：2026-10-18。对应当前讨论的 p，即条件性分离定理中的 $\rho$。

**已经实现：提供 `config.json` 和 safetensors 参数，即可计算指定 KV 层、全部当前 token、全部声明前缀上的 p 上界。** 不需要 prompt、token 对、前缀样本、旧实验结果或模型前向推理。

当前自动读取器支持 **Gemma 4 dense text decoder 的 MLX affine W8 checkpoint**，包括本地 E2B、PLE、GQA、全局／滑动注意力、层缩放和 KV sharing；维数、词表大小、层数、group size 从模型读取，没有锁死为 E2B 的数值。不支持的 MoE、K=V、不同量化格式等会报错。其他架构需要对应的参数读取和计算图适配，不能只凭一堆没有结构说明的矩阵自动证明。

本次完成的是**完整上界的可计算性**。不要求 p 小，不检查候选恢复条件，也不把这个保守上界称为真实最小 p。

## 1. 直接运行

在仓库根目录，使用已有环境：

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/mathematical_analysis/parameter_p/compute_p.py \
  --model-dir kvidence/experiments/kv_guided_search_gemma4_int8/model \
  --layers 10,15 \
  --output-dir kvidence/mathematical_analysis/parameter_p/selected_run
```

换模型时更换 `--model-dir`。该目录需要 `config.json`、`model.safetensors`，或 safetensors 分片及 `model.safetensors.index.json`。不需要 tokenizer、下载模型代码、加载模型执行框架或联网。

省略 `--layers` 会计算全部生产层，并将共享层映射到实际 KV 生产层。层号**从 1 开始**。当前模型的第 16–35 层复用第 14 或 15 层的 KV，不会被重复计入联合输出。

`--anchor-scale` 是精确非负有理数，默认 `1`，也可以写 `1/2`。它改变下文定义中的 A，因此也改变 p，不能把不同 A 的 p 数值当作同一问题直接比较。当前没有自动优化 A。

输出：

- `results.json`：严格有理数上界、显示近似值、各层贡献、A 的完整构造、各 K/V head 的候选上界、参数来源哈希。
- `report.md`：便于阅读的结果表。
- `verification.json`：运行复算工具后生成。

复算：

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/mathematical_analysis/parameter_p/verify.py \
  kvidence/mathematical_analysis/parameter_p/run/results.json
```

可以给 `verify.py` 追加 `--model-dir /new/checkpoint/path`，检查移动后的同一份参数。程序重新读取参数、重新计算并逐字段比较，不只检查 JSON 内部算术。它与生成器共享实现，**不是独立证明系统**。

新环境依赖见 [requirements.txt](requirements.txt)。已有环境无需重新安装。

## 2. 本次计算的 p 到底是什么

延续 [条件性分离定理 C1](../conditional_token_separation.md)，固定模型参数、基准 A 和观察位置：

$$
p_*^{(A)}=
\sup_{u,v\in\mathcal V,\ S\in\mathcal S}
\left\|F_S(u)-F_S(v)-A(z_u-z_v)\right\|_2.
$$

工具输出满足

$$
\boxed{p_*^{(A)}\le\widehat p.}
$$

这里 $\mathcal V$ 默认是模型的**整个词表**；$\mathcal S$ 覆盖声明因果文本模型的全部有限参考前缀，比较两候选时使用相同 S、相同位置。覆盖整个词表自然也覆盖任何英语／数字子集；不需要先挑数字或执行它们的前向计算。

$F_S$ 是所选层的实值 K/V 拼接。K 撤销已知位置的 RoPE，V 保持原样，使用普通 L2，不是逐维 RMS，也不是 INT8 整数码或 LSB。若希望保留原 K，只需同步旋转 A 的 K 块；正交旋转不改变 p 上界。

$z_v$ 包含缩放主 embedding 与实际 PLE 特征。本计算器将原先没有数值指定的 A **明确固定**为下面的残差线性基准，保留主 embedding 通路，PLE 的列设为零；PLE 的完整作用纳入余项 p，绝未假设 PLE 不存在或与 token 无关。

因此 p 同时包含前缀和非线性相对该 A 的偏移，**不是“运行误差”或纯粹两个前缀之间的距离**。它也不是条件定理 C5 在额外选择 $\beta$ 后的某个唯一加性参数。若使用本工具的统一值，可直接作为 C1 的 $\rho_{uv}=\widehat p$。

## 3. 所有层如何接入

### 3.1 三个分支的全域幅度界

Gemma 的实际执行顺序为 attention 残差、FFN 残差、PLE 残差，然后乘层标量。将实际分支输出记为 $a_\ell,f_\ell,g_\ell$：

$$
h_\ell=s_\ell(h_{\ell-1}+a_\ell+f_\ell+g_\ell).
$$

它们在各自真实的中间状态上计算，并没有被当作彼此独立的函数。
对 RMSNorm，有

$$
N_\gamma(x)=\frac{D_\gamma x}{\sqrt{\|x\|_2^2/d+\varepsilon}},
\qquad
\|N_\gamma(x)\|_2\le\sqrt d\|\gamma\|_\infty.
$$

证明是 $\|D_\gamma x\|\le\|\gamma\|_\infty\|x\|$，而分母至少为 $\|x\|/\sqrt d$；$x=0$ 单独成立。

三个分支末尾都有各自的 post RMSNorm，因此直接从其参数得到

$$
B_{A,\ell}=\sqrt d\|\gamma_{A,\ell}\|_\infty,\quad
B_{F,\ell}=\sqrt d\|\gamma_{F,\ell}\|_\infty,\quad
B_{P,\ell}=\sqrt d\|\gamma_{P,\ell}\|_\infty.
$$

没有 PLE 时 $B_{P,\ell}=0$。这覆盖内部 Q/K、softmax、GQA、历史 cache、门控 FFN、PLE 投影及它们的相互作用，因为不论内部实值输出是什么，都经过相应的 post norm。

**内部矩阵没有被假设为零。** 其结构和量化元数据被检查；本版幅度界对任何有限内部权重都成立，所以数值只依赖 post norm 参数。它不利用共同前缀相关性和内部权重方向，因而保守。此前 FFN Jacobian/Hessian 等较细的证明仍可用于未来收紧，但本工具不依赖尚未闭合的细化路径，也不冒充已经将那些路径全部紧化。

### 3.2 隐藏偏移递推

设 $h_0(v)=\sqrt d\,e_v$，$\sigma_\ell=\prod_{j=1}^{\ell}s_j$，固定残差锚点

$$
\bar h_\ell(v)=\sigma_\ell h_0(v).
$$

用参数递推：

$$
R_0=0,\qquad
\boxed{
R_\ell=|s_\ell|
\left(R_{\ell-1}+B_{A,\ell}+B_{F,\ell}+B_{P,\ell}\right).
}
$$

从 $h_\ell-\bar h_\ell$ 相减、应用三角不等式并归纳，得到

$$
\forall S,v:\quad
\|h_\ell(S,v)-\bar h_\ell(v)\|_2\le R_\ell.
$$

标量可以是负数或零；传播半径时用绝对值，构造 A 时保留其真实符号。

**第 m 层 KV 来自 $h_{m-1}$，所以使用 $R_{m-1}$，不是 $R_m$。**
晚于 m 的层不会反过来影响这份 KV。共享层返回先前生产层的证书。

### 3.3 从 embedding 参数得到全部 token 的输入域

每个 affine W8 参数为

$$
e_{v,j}=q_{v,j}s_{v,g}+b_{v,g},\qquad q_{v,j}\in\{0,\ldots,255\}.
$$

因此该 group 的每个元素均满足

$$
|e_{v,j}|\le m_{v,g}
=\max\{|b_{v,g}|,|255s_{v,g}+b_{v,g}|\}.
$$

若 group 大小为 G，则

$$
\|h_0(v)\|^2
\le dG\sum_gm_{v,g}^2,
\qquad
H_0^2=d\max_v\left(G\sum_gm_{v,g}^2\right).
$$

这只扫描 embedding 的量化元数据，直接包住整个词表；没有执行候选对攻击，也没有枚举前缀。主 embedding 实际码值无需逐个展开。PLE 表通过其分支 post norm 整体包住。

目标层前有

$$
\bar H=|\sigma_{m-1}|H_0,\qquad H=\bar H+R_{m-1}.
$$

## 4. 最后接到 K/V：不能漏掉两次归一化

以下对一个 head 证明，维数为 k。令 $\gamma_i$ 为目标层 input norm 的缩放，$\gamma_o$ 为 head norm 的缩放；V 的 $\gamma_o=\mathbf1$。

定义

$$
B=WD_{\gamma_i},\qquad T=D_{\gamma_o}B.
$$

将 input RMSNorm、投影和 head RMSNorm 精确合并：

$$
U(h)=\frac{Th}{t(h)},\qquad
t(h)^2=\frac{\|Bh\|^2}{k}
+\frac{\varepsilon}{d}\|h\|^2+\varepsilon^2.
$$

因此不是把最终 KV 简化成一次线性投影。
参数给出经过认证的 $b\ge\|B\|_2$、$\tau\ge\|T\|_2$，以及

$$
M=\sqrt k\|\gamma_o\|_\infty,\quad
\|U(h)\|\le M,\quad
L=\frac{\|\gamma_o\|_\infty b}{\varepsilon}.
$$

L 是 $U=N_{\gamma_o}\circ B\circ N_{\mathbf1}$ 的全空间 Lipschitz 上界：每个 RMSNorm 的导数范数至多其最大缩放除以 $\sqrt\varepsilon$。

对于 $\|h\|\le X$：

$$
t_- =\varepsilon\le t(h)\le
t_+(X)=\sqrt{(b^2/k+\varepsilon/d)X^2+\varepsilon^2}.
$$

### 4.1 固定线性基准 A

工具默认 $\eta=1$，并指定

$$
\boxed{A_{m,\text{head}}z_v
=\eta T\sigma_{m-1}h_0(v).}
$$

这等价于以固定分母 1 选定线性参照，**不是假定真实 norm 分母等于 1**。两者的差会在下一步完整计入 p。A 对 PLE 特征坐标的系数为零，所有 head 堆叠后得到整体 A。

### 4.2 一次归一化偏离线性参照的界

因为 $Th=t(h)U(h)$，所以

$$
U(h)-\eta Th=[1-\eta t(h)]U(h).
$$

设

$$
D(X)=\min\left\{
M\max\{|1-\eta t_-|,|1-\eta t_+(X)|\},
M+\eta\tau X\right\}.
$$

则 $\|h\|\le X$ 时，$\|U(h)-\eta Th\|\le D(X)$。第一项来自上一恒等式，第二项来自三角不等式。

### 4.3 三条完整 p 上界，取最小者

记 $R=R_{m-1}$。对所有 S、u、v，所需余项满足

$$
\boxed{
\rho_{m,\text{head}}\le
\min\left\{
2LR+2D(\bar H),\quad
2D(H)+2\eta\tau R,\quad
2M+2\eta\tau\bar H
\right\}.
}
$$

三项的证明分别是：

1. 把两个真实隐藏分别移到残差锚点，各花费至多 LR；再将两个锚点的真实 KV 改为线性 A，各花费至多 $D(\bar H)$。
2. 直接在两个真实隐藏上减去 $\eta Th$，各花费至多 $D(H)$；剩余真实隐藏差相对锚点差的偏移至多 $2R$，经 $\eta T$ 后至多 $2\eta\tau R$。
3. 两个 KV 的范数至多 M，两个线性锚点的范数至多 $\eta\tau\bar H$，四项直接相加。

以上每条都包含完整非线性、前缀和 PLE 的作用；没有遗漏“参考输出到原 A”的差异。每个 head 取较小的合法上界，然后对拼接输出使用

$$
\boxed{\widehat p=\sqrt{\sum_{m,\text{head}}\rho_{m,\text{head}}^2}.}
$$

求和只计实际选中的、不重复的 KV 坐标。不能对同一生产层的共享别名重复计数后仍称为原输出度量。

## 5. 数值为何是上界，而非普通浮点估计

- BF16/FP16/FP32 元数据按精确二进制有理数读取，W8 按 $q\cdot scale+bias$ 精确整数解码。
- B 的谱范数平方使用 $\min(\|B\|_F^2,\|B\|_1\|B\|_\infty)$；T 使用 $\min(\|T\|_F^2,\|\gamma_o\|_\infty^2 b^2)$。不是将幂迭代近似值当作上界。
- 矩阵平方与求和在 int64 前检查溢出，必要时转为 Python 大整数。
- 后续运算采用 `Fraction`；根号以整数平方根得到 $2^{-96}$ 网格上的上下包络；半径和最终上界向上舍入。
- `*_display` 是便于阅读的浮点近似，证明依据是 JSON 的有理数字段。
- 模型参数范围或格式不满足读取器支持条件时直接失败，不继续生成证书。

RMSNorm 的 $\varepsilon$ 使用 config 中十进制数的精确有理数解释，主 embedding 的 $\sqrt d$ 为实数参考缩放。本证书针对这一明确参考，不把实际 BF16／NPU 的舍入也当作已证明。

## 6. 本地 Gemma 4 E2B W8 的实际结果

完整结果见 [run/report.md](run/report.md)、[run/results.json](run/results.json)，复算记录见 [run/verification.json](run/verification.json)。该目录仍是此前下载的 `mlx-community/gemma-4-e2b-it-8bit` checkpoint；实际参数绑定以 config 和相关张量哈希为准。

默认 $\eta=1$，全词表、全部声明前缀、K+V 标准 L2：

| 观察层（从 1 起） | KV 前隐藏偏移上界 R | 完整 p 上界，向上保留两位小数 |
|---:|---:|---:|
| 10 | 202.400148874 | **55.12** |
| 15 | 129.736301664 | **45.36** |
| 第 10 与 15 层拼接 | — | **71.39** |

全部 35 个 decoder 层的组成均已读取检查，15 个独立 KV 生产层均有输出；源参数重新读取复算通过。后续共享层不生成新的独立 KV。

**这两层当前取到的最小项都是第三项 `output_range`。** 这是 KV 范围与固定 A 的直接包络；前两条逐层传播的合法界也计算并保存在 JSON 中，但更松。不能将数值描述为已经精确测得了前缀影响、已经利用共同前缀抵消，或已经获得紧的深层证书。

实现验证有 5 项：精确根号／溢出回退、矩阵范数的独立有理数核对、小型独立完整前向图诊断、无 PLE／不同 A／篡改检测、不支持架构拒绝。小型前向图包括多 head、GQA、滑窗、RoPE、负／零 scalar、共享 KV、PLE 和双宽 FFN；它仅用于发现实现错误，不是全域证明的来源。真实 Gemma 计算的前向推理次数和候选对计算次数均为 0。

测试命令：

```bash
cd kvidence/mathematical_analysis/parameter_p
../../experiments/kv_guided_search_gemma4_int8/.venv/bin/python -m unittest -v test_compute_p
```

## 7. 与已有证明及下一步的关系

本工具闭合了 [参数分离证明的幅度包络](../parameter_separation_proof.md) 到 [固定 A 的 C1](../conditional_token_separation.md) 的完整上界计算。此前的 FFN 全域导数、共同位置／差值递推、共享前缀 attention、PLE 投影和融合 norm 结果没有失效；它们可以用来收紧这个基线。

当前已经回答“给定这些模型参数，能不能算一个完整的 p 上界”：**能。** 是否需要更紧的 A／状态域／方向界是另一个问题。只有进一步把 $\|Aq\|-\widehat p$ 与执行误差预算比较，才涉及恢复保证。

本工具不计算硬件执行预算，不把 ±4 LSB 当作标准 L2 的 4，不认证 KV8／BF16 舍入；当前 token 的音频／图像编码器输入也不属于此文本域。要扩展这些对象，需要增加其明确输入域、参考计算图或误差界。

## English definition

For a supported Gemma 4 dense text checkpoint, the calculator returns a conservative parameter-derived bound

$$
\sup_{S,u,v}\|F_S(u)-F_S(v)-A(z_u-z_v)\|_2\le\widehat p.
$$

The reference map A is explicitly fixed by the residual scalar path, the fused input/head-normalized KV projection coefficients, and the chosen constant scale eta. All PLE contributions are enclosed in the remainder. The calculation includes every decoder branch and respects the KV production point and cache-sharing graph. It needs no inference, token-pair search, or prefix enumeration. It computes an upper bound, not the exact supremum or a recovery guarantee. The reference uses exact affine W8 coefficients and real arithmetic; hardware and KV8 execution errors are separate.
