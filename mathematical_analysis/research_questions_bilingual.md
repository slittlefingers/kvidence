# KVidence 研究问题：深层 KV 的稳健 token 恢复
# KVidence Research Questions: Robust Token Recovery from Deep-Layer KV Caches

日期 / Date: 2026-10-17

本文统一研究问题与边界，不宣称以下问题已经解决。

This document aligns the research questions and their scope. It does not claim that these questions have been resolved.

**当前证明目标的进一步明确 / Clarification of the current proof target:**

中文：问题一的目标是给出条件定理——在什么可核验的模型参数与状态域条件下，哪些 token 对的特征间距足以抵抗多大的数值误差，从而被指定的 KV 匹配规则精确区分。允许只认证满足条件的候选对或候选子集；不预设实际 Gemma 的全部 token、全部前缀都通过同一个条件。条件仍须覆盖所声明的整个前缀域，不能用采样成功代替。统一证明见 [条件性 token 分离定理](conditional_token_separation.md)。

English: Question 1 seeks a conditional theorem: under which verifiable model-parameter and state-domain conditions does the feature separation of a token pair suffice to tolerate a specified numerical error and permit exact discrimination by the stated KV matching rule? Certification may apply to qualifying pairs or candidate subsets; it does not presume that every token and prefix of the actual Gemma model satisfies one universal condition. The conditions must still cover the entire declared prefix domain, rather than only sampled successes. See the [unified conditional separation proof](conditional_token_separation.md).

## 共同研究设定 / Shared Research Setting

### 中文

- **模型已知且固定：** 使用指定版本的 Gemma，主 embedding 表、per-layer embedding（PLE）、其余模型参数及 tokenizer 已知；不通过重新训练或修改词表改变模型。分析可限制允许的输入集合，但必须保证真实 token 被该集合覆盖。
- **观测来自内存：** 从内存读取指定深层的 KV 及解读它所需的元数据，不依赖应用事先记录的 token ID 或对话历史。研究不以首层 KV 的直接恢复为重点。
- **恢复标准：** 恢复原始 token ID；语义相近、拼写相似或生成内容合理均不等于精确恢复。
- **输入域：** 以英语、数字及明确允许的符号为研究对象。字符范围、文本规则、最大长度和实际可产生的 token 集合需正式定义。“正常英语”不是已完成的数学定义；若要覆盖密码或随机字符串，必须明确纳入。
- **误差模型：** 研究实际量化舍入、浮点计算及执行路径差异产生的数值偏差，并检查这些偏差如何作用于缓存重建。不能直接将观测到的最大单维偏差，替换为所有维度都可独立取极值的对抗性噪声模型。
- **度量仍需明确：** 已观察到个别维度可能相差约 2–4 LSB，而整体误差较小；“约 0.9”究竟指标准 L2、逐维 RMS 还是其他指标，其单位与参考 scale 尚待明确。上述数值目前是观测，不是整个输入域的已证明上界。比较不同量化载荷时，必须考虑各自 scale/bias，不能只比较整数码差。

### English

- **Known, fixed model:** Use a specified Gemma version with known main embeddings, per-layer embeddings (PLE), remaining model parameters, and tokenizer. Do not retrain the model or alter its vocabulary. The analysis may restrict the admissible input set, but that set must include the true tokens.
- **Memory-based observations:** Read KV data from specified deep layers and the metadata needed to interpret it, without relying on token IDs or conversation histories previously recorded by the application. Direct recovery from the first layer is not the primary focus.
- **Recovery criterion:** Recover the original token IDs exactly. Semantic similarity, similar spelling, and plausible generated text do not count as exact recovery.
- **Input domain:** Focus on English text, digits, and explicitly permitted symbols. The character set, text constraints, maximum length, and reachable token set must be formally specified. “Normal English” is not yet a mathematical domain definition. Passwords and random strings must be explicitly included if they are intended targets.
- **Error model:** Study numerical discrepancies arising from actual quantization rounding, floating-point computation, and execution-path differences, including their effects on cache reconstruction. An observed maximum coordinate discrepancy must not automatically be replaced by an adversarial model in which every coordinate can independently attain its extreme value.
- **Metrics to be clarified:** Individual coordinates have been observed to differ by approximately 2–4 LSB, while aggregate error is smaller. Whether “approximately 0.9” denotes the standard L2 norm, per-coordinate RMS, or another metric, and its units and reference scales, remain to be specified. These values are observations, not proven bounds over the entire input domain. Comparisons between quantized payloads must account for their respective scales and biases, rather than integer-code differences alone.

## 问题一：正确前缀下的稳健可辨识性
## Question 1: Robust Identifiability Given a Correct Prefix

### 中文

**研究问题：** 在此前 token 已正确恢复的条件下，实际量化和浮点执行差异是否会破坏当前深层 KV 对不同候选 token 的区分能力？能否从允许的 embedding/PLE 表行、固定模型参数及实际数值误差机制，推导对声明输入域成立的精确恢复条件？

正确的前缀 token 并不意味着攻击者重建的缓存与目标缓存数值相同；两条执行路径之间的差异必须纳入分析。

需要区分两个层次：

1. **观测可辨识性：** 给定相同正确前缀，两个不同 token 在允许的实际执行条件下，是否可能产生无法区分的 KV 观测？如果允许观测集合重叠，只依赖该观测的算法无法对两种情况同时保证正确。
2. **匹配规则的稳健性：** 在 token 可辨识的前提下，指定的距离和匹配规则能否始终将真 token 与错误候选区分开？最近邻选错不自动证明所有恢复算法都不可能成功。

**期望结果：** 给出以有限表行、模型参数和有效误差界为条件的定理，明确适用输入域、观测层、数值格式与匹配规则；或者构造符合该同一设定的反例，精确说明哪个保证不成立。有限案例成功率不能替代全域证明。

### English

**Research question:** Assuming that all preceding tokens have been recovered correctly, can discrepancies from actual quantization and floating-point execution compromise the ability of the current deep-layer KV observation to distinguish candidate tokens? Can exact-recovery conditions valid over the declared input domain be derived from the admissible embedding/PLE rows, fixed model parameters, and the actual numerical error mechanism?

A correct token prefix does not imply that the attacker's reconstructed cache is numerically identical to the target cache. Differences between the two execution paths must be included in the analysis.

Two levels must be distinguished:

1. **Observational identifiability:** Given the same correct prefix, can two different tokens produce indistinguishable KV observations under the admissible execution conditions? If their admissible observation sets overlap, an algorithm using only that observation cannot guarantee correctness in both cases.
2. **Robustness of the matching rule:** When tokens are identifiable, does the specified distance and matching rule reliably distinguish the true token from incorrect candidates? A nearest-neighbor error does not by itself establish impossibility for all recovery algorithms.

**Desired result:** A theorem with conditions expressed in terms of finite table rows, model parameters, and valid error bounds, with an explicit input domain, observed layers, numerical format, and matching rule; alternatively, a counterexample satisfying that same setting and identifying precisely which guarantee fails. Success rates on finitely many cases do not replace a domain-wide proof.

## 问题二：缓存重建误差的传播与整段恢复
## Question 2: Propagation of Cache-Reconstruction Error and Sequence Recovery

### 中文

**研究问题：** 即使此前恢复的 token ID 全部正确，攻击者重建缓存与目标缓存之间的数值差异，会如何通过后续 attention、残差、FFN、PLE 和量化操作，影响未来位置的深层 KV？这种影响在什么条件下仍能保留问题一要求的区分能力，从而保证整段精确恢复？

需要分析：

1. **数值状态的传播：** 在输入 token 始终正确时，给出逐位置或逐层误差界，明确其对上下文长度、网络深度、滑动窗口、共享 KV 和执行路径的依赖。不能预设误差必然累积，也不能预设其必然衰减或与长度无关。
2. **恢复归纳的闭合：** 证明若此前 token 正确且状态误差满足规定条件，当前 token 也能正确恢复，并且更新后的状态继续满足下一步所需条件。
3. **首次误判后的行为：** 若某一步恢复了错误 ID，应将由此产生的离散输入错误与此前的纯数值误差分开，研究其可能造成的级联错误或可纠正性；不能默认之前的正确前缀误差界继续适用。

**期望结果：** 一个与实际缓存更新规则一致的传播定理，并与问题一的条件结合形成整段恢复保证；若不能形成保证，应明确失效机制和适用边界。最终单个 KV 向量的扰动容忍度不能直接当作每一步内部误差传播后的容忍度。

### English

**Research question:** Even when every previously recovered token ID is correct, how do numerical differences between the attacker's reconstructed cache and the target cache affect deep-layer KV at future positions through attention, residual connections, FFNs, PLE, and quantization? Under what conditions does this propagation preserve the distinguishability required in Question 1, enabling exact recovery of an entire sequence?

The analysis should address:

1. **Propagation of numerical state error:** With correct token inputs throughout, derive per-position or per-layer bounds, making explicit their dependence on context length, network depth, sliding windows, shared KV, and execution paths. Do not assume in advance that errors must accumulate, decay, or remain independent of length.
2. **Closure of the recovery induction:** Show that a correct prefix and an admissible state-error bound imply correct recovery of the current token, and that the updated state continues to satisfy the conditions needed at the next step.
3. **Behavior after the first incorrect token:** Separate the discrete input error caused by an incorrect recovered ID from purely numerical discrepancies, and investigate possible error cascades or correction. Bounds conditional on a correct prefix must not automatically be carried over after a token error.

**Desired result:** An error-propagation theorem consistent with the actual cache-update rules, combined with Question 1 to establish a sequence-level recovery guarantee. If that guarantee cannot be established, identify the failure mechanism and scope. Robustness to perturbations of one final KV vector must not be treated as robustness to internally propagated errors at every step.

## 问题三：有限查询下的搜索覆盖与恢复效率
## Question 3: Search Coverage and Recovery Efficiency Under a Limited Query Budget

### 中文

**研究问题：** 假设当前 token 在问题一的意义下可可靠区分，且前缀状态满足问题二的条件，我们的 KV 方向引导搜索，能否在有限真实前向查询预算内找到真 token，并保持精确恢复？

这里的搜索方法以已查询的 embedding/PLE 特征及其真实 KV 响应估计候选方向，再选择后续真实前向查询；其具体代理模型、特征和选择规则必须明确。代理预测的 KV 不是额外的真实观测。

需要分析：

1. **候选覆盖：** 在哪些几何或模型条件下，有限预算内的查询集合必然包含真 token？若只能给概率保证，必须明确输入或随机化分布、量词和失败概率。
2. **查询复杂度：** 为达到指定恢复保证，需要多少次真实前向查询？相较穷举的优势依赖哪些条件？有限样本拟合误差不能直接充当全域代理误差上界。
3. **搜索与匹配的分离：** 真 token 未被查询属于搜索覆盖失败；真 token 已被查询但匹配错误属于可辨识性、数值稳健性或匹配规则问题。两者不能用同一个成功率掩盖。

**期望结果：** 搜索覆盖定理或明确定义的概率保证、查询复杂度界，以及与前两个问题相接的完整恢复条件。查询次数与实际耗时应分别报告；硬件吞吐量的变化不能代替算法复杂度分析。

### English

**Research question:** Assuming that the current token is reliably distinguishable in the sense of Question 1 and that the prefix state satisfies the conditions in Question 2, can our KV-guided search find the true token within a limited budget of actual forward evaluations while preserving exact recovery?

The search estimates candidate directions from queried embedding/PLE features and their actual KV responses, then selects subsequent forward queries. Its surrogate model, features, and selection rule must be specified. Surrogate-predicted KV values are not additional actual observations.

The analysis should address:

1. **Candidate coverage:** Under what geometric or model conditions must the queried set contain the true token within the budget? If the guarantee is probabilistic, specify the input or randomization distribution, quantifiers, and failure probability.
2. **Query complexity:** How many actual forward evaluations are required for the stated recovery guarantee, and under what conditions is this fewer than exhaustive search? A fitting error measured on finitely many samples must not be treated as a uniform surrogate-error bound.
3. **Separation of search and matching:** Failing to query the true token is a coverage failure. Querying it but selecting an incorrect token is an identifiability, numerical-robustness, or matching-rule issue. A single aggregate success rate must not obscure this distinction.

**Desired result:** A coverage theorem or explicitly defined probabilistic guarantee, a query-complexity bound, and complete recovery conditions connecting this question to the first two. Query counts and wall-clock time should be reported separately; hardware throughput does not replace algorithmic complexity analysis.

## 三个问题的关系与当前优先级 / Relationship and Current Priority

**中文：** 问题一回答“能否可靠区分”，问题二回答“重建过程中这种能力能否保持”，问题三回答“能否以更少查询找到真候选”。整体恢复保证需要三者衔接，但当前首先研究正确前缀下、符合实际数值机制的问题一。任何结论均应区分：已证明的条件性保证、符合设定的反例，以及尚未解决的部分；不能把某个证明方法的界过松，等同于问题无解。

**English:** Question 1 asks whether tokens can be reliably distinguished; Question 2 asks whether this ability is preserved during reconstruction; Question 3 asks whether the true candidate can be found with fewer queries. A complete recovery guarantee must connect all three. The current priority is Question 1 under a correct prefix and an error model faithful to actual numerical execution. Conclusions must distinguish proven conditional guarantees, counterexamples satisfying the stated assumptions, and unresolved questions. A loose bound from one proof method does not establish impossibility of the underlying problem.
