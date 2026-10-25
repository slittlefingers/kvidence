# Collision+ 官方源码核对（CPA 移植前的审阅记录）

后续已增加 `collision_plus_cpa`，当前实现与验证请看 [CPA_BASELINE.md](CPA_BASELINE.md)。本页所称“当前实现”专指仍保留的旧 `collision_plus` L2 适配基线。

核对日期：2026-10-19。**当前实现的候选搜索顺序与官方接近，但不等同于官方 Collision+：没有 chosen-plaintext 阈值校准，而且匹配判据不同。**

## 来源与版本

官方仓库：[SiO-2/kvcloak](https://github.com/SiO-2/kvcloak)。本地 checkout：`../references/kvcloak`。固定 commit 为 `6b40f36edb2f337557543e7e60b10022308883d4`，提交日期 2026-02-28。保留 Apache-2.0 LICENSE，没有修改上游代码、安装依赖或执行攻击脚本。

[下载及哈希记录](../references/kvcloak_source.json)。旧实验 `experiments/ciphertext_collision_comparison/reference/collision.py` 与此次官方文件 SHA-256 完全相同：`7b9dd072e2d28d0f2e24b2b0dd06bd2da35c994da6198e8a1282a6eb24655b9a`。所以不是旧 reference 下载错了；框架的 `kvrecon/search.py` 是另一个适配实现。

## 对照结果

| 项目 | 官方 | 当前 KVidence |
| --- | --- | --- |
| 概率排序 | 根据已恢复前缀的 logits，softmax 后排序 | 直接稳定排序 logits；数学上保序，但浮点下的并列情况不能保证逐位相同 |
| 通过者选择 | 完成 logical batch，接受概率顺序中第一个通过者 | 相同基本选择原则；不选距离最小者 |
| 距离 | K、V 分别计算 Frobenius/L2 距离 | 拼接 K+V 后计算单一 L2 |
| Collision+ | `--enhance` 加载已知明文校准统计，计算分量阈值；当前主入口 rank guess=8 | 没有这一步，使用此前指定的固定默认 epsilon=4 |
| 普通 Collision | 无校准阈值时，用各分量距离的批均值减 target_gap 倍标准差 | 没有该异常值判据 |
| 阈值接受 | K、V 条件同时成立；指定阈值乘 1.01 | 单一距离严格 `< epsilon`，无额外 1% |
| 搜索范围 | 主入口 stop_partition 默认 1；独立 collision.py CLI 默认 3 | 默认全原始词表；可显式限制 ID 集或预算 |
| 初始前缀 | 从模型 BOS 开始；无 BOS/logits 时初始按 ID 排序 | 使用输入给定的非空公开前缀；深层目标需紧接其后 |
| 推进前缀 | 复用被接受候选那一路的 cache/logits | decode commit 重算并追加 KV，额外检查 commit 距离 |
| 执行 | PyTorch/Transformers 真正 batch 复制前缀；含 CUDA 操作 | LiteRT NPU 隔离候选槽位、物理拆批、ADB 传输 |
| 精度 | 主入口 float16/bfloat16/float32 | Gemma 4 INT8 KV 反量化比较；Gemma 3 float32 KV |

上游源码定位（固定版本）：

- [排序与搜索范围](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/collision.py#L201-L242)
- [分别计算 K/V 距离](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/collision.py#L298-L310)
- [阈值、首个通过者和 cache 推进](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/collision.py#L344-L394)
- [enhance、rank guess 与配置加载](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/attacks.py#L193-L253)
- [主入口范围默认值](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/attacks.py#L479-L495)；[独立入口默认值](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/collision.py#L426-L447)。截断按 batch 起点判断，最后一批可能越过精确分数边界。

本地实现：[search.py](kvrecon/search.py)、[engine.py](kvrecon/engine.py)、[npu.py](kvrecon/npu.py)。

## “+”具体增强了什么

概率排序在普通 Collision 中已经存在，不是“+”本身。增强项是读取已知明文校准样本里的真 token / 其他 token 距离统计，并选择阈值。

若统计样本的真 token 最大距离小于其他 token 最小距离，函数取二者中点；否则使用均值、标准差建立正态近似并优化阈值。来源：[get_collision_threshold](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/collision.py#L101-L168)、[校准脚本](https://github.com/SiO-2/kvcloak/blob/6b40f36edb2f337557543e7e60b10022308883d4/attack/get_collision_threshold.py)。这不是在攻击过程中直接读取待恢复秘密的标签。

本次是静态核对，没有验证校准脚本在当前 Transformers 下能否直接运行。当前 checkout 的跟踪文件未包含预生成 `attack/config/` 阈值资产，不能假设下载后即可用于 Gemma NPU。

## 为什么判据不等价

令 d_K、d_V 为 K、V 的独立 L2 距离，官方给定阈值时要求：

$$
d_K<1.01\tau_K\quad\text{且}\quad d_V<1.01\tau_V.
$$

当前框架要求：

$$
\sqrt{d_K^2+d_V^2}<\epsilon,\qquad\epsilon=4\text{（默认）}.
$$

假设 tau_K=tau_V=4，某候选 d_K=d_V=3：官方条件通过，而我们的拼接距离约 4.24，不通过。这是说明判据差异的例子，不是作者的实测阈值。两种接受区域不同，不能仅调整一个常数就普遍等价；也不能从这个例子推断谁实际更快或更准确。

## 框架及论文中的定位

为兼容旧命令保留 `--method collision_plus`。当前结果应标为 **probability-ordered collision baseline with a shared L2 criterion**，或 **Collision+ search-order adaptation (without CPA calibration)**。此前“Collision+ (L2-adapted)”结果只支持这个范围。

该适配对照用于比较：相同模型、词表、设备、已知前缀和 K+V L2 标准下，候选提议是否比概率排序有效。它不能直接证明优于作者完整的 Collision+。

若论文要对比官方完整方法，仍需另设基线：使用独立校准集，实现 K/V 双阈值和 1% 容差，明确 rank 参数、搜索范围、logical/micro batch，并披露 NPU commit 重算等移植差异。不能直接套用 Llama 的阈值到 Gemma，也不能用测试秘密拟合阈值。

本次没有把 epsilon=4 替换为官方阈值，没有改写历史实验结果，没有声称完成官方 Collision+ 的 NPU 复现。

## 统一对比入口的范围

新增 `compare` 对比当前 L2 适配基线与 guided，各自在新进程中建立公开前缀；重复时交替顺序。两者都用 K+V，不自动混入第一层 V-only 查表。

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
"$PY" "$APP/run.py" compare --baseline collision_plus --model-name gemma4-e2b \
  --observation /path/to/observation --output /path/to/new_comparison \
  --epsilon 4 --batch-size 128 --repeats 2
"$PY" "$APP/run.py" evaluate-comparison \
  --comparison /path/to/new_comparison --truth /path/to/private_eval/truth.json
```

恢复过程不读取真值，单独评价后才报告正确数。输出含逐次结果、日志、`comparison.json`、`summary.json`、`SUMMARY.md`。记录候选数、模型执行时间、在线时间和整条命令时间，不自动生成速度优势结论。34 项主机测试通过；这个新成对入口尚未执行真实设备对比实验。
