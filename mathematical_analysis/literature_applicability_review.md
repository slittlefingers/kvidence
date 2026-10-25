# 理论论文能否直接证明 KVidence 的问题一

日期：2026-10-17。对象：已知固定 Gemma、正确 token 前缀、深层 KV、实际量化和浮点执行差异下的稳健恢复。本轮阅读定理、相关证明与模型假设，没有新增模型实验。

## 1. 结论

**五篇论文均不能直接给出当前 Gemma 的全域恢复结论。** 可以引用部分数学引理、借鉴差分验证框架，但必须补齐实际模型的条件与算子证明。不能把“已有理论方法”写成“Gemma 已满足定理”。

尤其需要区分：隐藏表示可逆不等于 KV 可逆；误差上界不等于候选分离下界；固定句子附近的输入区域不等于所有允许英文前缀；对称定点量化不等于当前 affine KV8/BF16 混合执行。

## 2. 原文核对结果

| 论文与核对位置 | 原文的核心条件／结论 | 对我们的直接适用性 |
|---|---|---|
| [Invertible Residual Networks](https://proceedings.mlr.press/v97/behrmann19a.html)，Theorem 1、Lemma 2、补充材料 Appendix A | 同维残差映射，各分支 Lipschitz 常数小于 1，得到可逆性与逆稳定性 | 局部引理可引用；Gemma 的分支条件、PLE 和 KV 投影未被覆盖 |
| [Globally Injective ReLU Networks](https://www.jmlr.org/papers/v23/21-0282.html)，Definition 1、Theorem 2/8、Remark 5、Appendix A.3 | ReLU 层满足 DSS 权重几何条件，给出全空间注入性和输出距离下界 | 不能直接套用 Gemma；其全空间条件也强于有限 token 集合需要的条件 |
| [ReluDiff](https://arxiv.org/html/2001.03662)，Sections 3.1、4.4、5、6 | 对仿射＋ReLU 前馈网络，联合传播两路计算的差异界并细分输入域 | 可借鉴共同前缀的关系分析；原算子规则不直接覆盖 Gemma，差异接近的性质也不是候选分离性质 |
| [QEBVerif](https://faculty.sist.shanghaitech.edu.cn/faculty/songfu/publications/CAV23a.pdf)，Sections 2–3、Theorem 1；对应 arXiv 版本 Theorem 3.1 | ReLU、对称均匀定点量化及规定的舍入／裁剪语义；证明差分分析有声性 | 可借鉴实际误差的差分分析；不能直接认证 affine KV8、动态 BF16 元数据或 GPU/NPU 浮点归约 |
| [Robustness Verification for Transformers](https://arxiv.org/html/2002.06622)，Sections 3.1–3.3、4.1、Appendices B/C/E | 为 attention 的跨位置依赖与非线性建立线性界；主要验证固定输入附近的分类稳健性 | 可借鉴 attention 界；不是当前 Gemma 的 KV 恢复定理。主实验还修改了 LayerNorm，移除了标准差项 |

已阅读 iResNet 的反三角不等式证明、ReLU 逆稳定性证明中的激活区域划分与线段论证、ReluDiff 的差分传播有声性及细分原则、QEBVerif 的量化语义与证明覆盖条件，以及 Transformer 的乘除与归一化边界构造。此表是适用性审查，不是声称读完每篇的所有实验与附录。

ReluDiff 使用 outward rounding 保护验证程序的区间计算；这不能直接解释为它已经建模了目标 NPU 算子的所有浮点执行语义。QEBVerif 的完备性同样限于其正式定义的网络和量化模型，不能迁移为 Gemma 验证器的完备性。

## 3. 对 Gemma 的三个具体缺口

以下判断依据本地 [Gemma 计算图](../experiments/kv_guided_search_gemma4_int8/.venv/lib/python3.13/site-packages/mlx_lm/models/gemma4_text.py) 和 [配置](../experiments/kv_guided_search_gemma4_int8/model/config.json)，不是论文已经完成的结论。

### 3.1 普通残差不自动满足 iResNet 条件，PLE 还会随 token 改变

对满足 $F=I+g$、$\mathrm{Lip}(g)=L<1$ 的参考块，Lemma 2 的下界可直接引用：

$$
\|F(x)-F(y)\|_2\ge(1-L)\|x-y\|_2.
$$

但本地 Gemma 并没有已认证的分支收缩条件；已有粗参数界失败不证明真实 Lipschitz 常数一定大于 1。前缀应作为共同条件变量，常数还必须在允许前缀域上统一。

另外，两个 token 的 PLE 不同。不能将“固定某个 PLE 后对隐藏向量可逆”直接当成两种 token 可区分。一个简单反例是 $T(h,a)=h-a$：固定 a 时修正分支对 h 的 Lipschitz 常数为 0，但两种候选若分别有 $(h,a)=(0,0)$ 和 $(1,1)$，输出都为 0。

### 3.2 隐藏可逆仍不足以保证 KV 分离

第 10、15 层的隐藏输入为 1536 维，K/V 投影各输出 512 维。堆叠矩阵 $[W_K;W_V]$ 为 $1024\times1536$，必有非零核空间；投影相同后，head norm 和 RoPE 无法恢复被抹掉的差异。因此不能用隐藏网络的全空间逆稳定性直接推出 KV 下界。

这不证明实际 token 碰撞：有限候选的隐藏差方向仍可能全部避开核空间。需要的是允许候选及前缀共同决定的差方向上的受限下界。JMLR 论文也明确区分全局注入与受限域注入，不能用其全空间条件未满足来宣称有限集合不可恢复。

### 3.3 实际数值误差必须另行建模

本地实现包括 GELU 近似与门控乘法、RMSNorm、softmax、RoPE、PLE、层缩放，以及 W8/KV8 的 affine 载荷和 BF16 元数据。它们与 ReLU 定点网络不同。

用户所述“最大单维约 4 LSB、整体约 0.9”的实际误差，不能换成各维独立任意取 ±4 的盒子后，再把盒子反例当作实际执行反例。0.9 的指标和单位仍待明确；真实量化舍入、反量化、浮点归约顺序和前缀重建偏差都应进入适当的执行模型。

## 4. 可在这些理论之上补出的条件性引理

下面是反三角不等式的直接推导，**不是原论文已经证明了 Gemma/PLE**。

设某参考块为 $T_p(h,a)=s(h+g_p(h,a))$，a 为当前 token 的辅助输入（例如 PLE）。若在所需区域内、对所有允许 p 有

$$
\|g_p(h,a)-g_p(h',a')\|_2
\le L_h\|h-h'\|_2+L_a\|a-a'\|_2,
\qquad L_h<1,
$$

则

$$
\boxed{
\|T_p(h,a)-T_p(h',a')\|_2
\ge |s|\max\{0,(1-L_h)\|h-h'\|_2-L_a\|a-a'\|_2\}.
}
$$

证明：将输出差写为主残差差加修正差，应用反三角不等式，再代入所声明的修正差上界；范数非负给出截断。该形式同时保留主 embedding 差、PLE 差和前缀一致性。

它是一条可能的条件性证明路线，但必须实际认证每个块的常数、保持区域覆盖，并补上 KV 映射的受限分离与实际数值误差界。目前这些条件没有在 Gemma 上完成认证。

## 5. 更贴近任务的迁移方式：直接验证匹配差距

不必要求整个网络全局可逆。对固定的真候选 u 和错误候选 v，构造三路共享同一符号前缀 p 的计算：目标执行得到 $y_u(p)$，攻击者执行得到 $c_u(p)$、$c_v(p)$。三路各自的数值路径不同，但其前缀 token、位置和模板受同一约束。

定义

$$
G_{u,v}(p)=\|c_v(p)-y_u(p)\|_2^2-\|c_u(p)-y_u(p)\|_2^2.
$$

若对全部允许候选对，能认证

$$
\inf_{p\in\mathcal P}G_{u,v}(p)>0,
$$

则错误候选在任何允许前缀下都比真候选远，因而最近邻正确。若允许多种执行顺序，还需对这些允许执行选择一并取下确界。这是由最近邻定义直接得到的充分条件，不是上述论文已完成的 Gemma 结论。

ReluDiff 的关系分析、Transformer 的 attention 松弛、QEBVerif 的数值差分建模分别提供可参考的组件。迁移需要新的 Gemma 算子规则与有声性论证，尤其是门控、RMSNorm 和动态 affine 量化，不能直接调用原工具就宣称证明。

输入域应保留允许 embedding/PLE 行的离散对应关系，或者使用覆盖它的可靠超集。超集上认证成功足够；超集中出现失败点若不可达，则不是实际输入反例。细分能收紧界，但不保证在大型模型上可行，也不保证存在正下界。

若先分析参考模型而非实际执行，还需单独证明实际误差预算；若直接分析执行图，则需忠实建模其量化和浮点语义。这两条路线不能混用假设。

## 6. 当前可以写入论文与不能写入论文的结论

- 可以引用残差逆稳定性引理，并明确我们的模型需满足的附加条件。
- 可以提出基于同一前缀的关系分析和匹配差距认证问题，借鉴上述验证方法。
- 不能宣称 Gemma 已满足残差收缩、全域 KV 分离或真实数值误差上界。
- 不能宣称五篇论文简单组合就构成我们的完整证明，也不能把纯数学引理包装为新发现。
- 界过松、验证资源不足、找到真实反例，以及完成认证，必须分别报告。

因此，目前最准确的判断是：**现成论文提供可复用的局部数学工具与验证框架，但没有提供可直接套用的 Gemma 深层 KV 稳健恢复定理。最值得继续的是保留共同前缀和实际数值关联的任务级差分验证，而非要求整个模型全局可逆。**

## 原文留档

原始 PDF、iResNet 证明补充材料及便于检索的提取文本保存在 `literature_sources/`；[manifest.json](literature_sources/manifest.json) 记录下载 URL、大小和 SHA-256。版本编号差异已标明，引用论文定理时应使用对应版本。

本地主要文件：[iResNet 正文](literature_sources/behrmann2019_invertible_residual_networks.pdf)、[iResNet 证明补充](literature_sources/behrmann2019_invertible_residual_networks_supplement.pdf)、[ReLU 注入性](literature_sources/puthawala2022_globally_injective_relu_networks.pdf)、[ReluDiff](literature_sources/paulsen2020_reludiff.pdf)、[QEBVerif](literature_sources/zhang2023_qebverif.pdf)、[Transformer 验证](literature_sources/shi2020_transformer_verification.pdf)。
