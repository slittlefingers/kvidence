# 第二轮手机端优化与精确候选选择

本轮保持全部原 token ID、原模型/数值精度、候选种子配额、ridge 评分函数和阈值，仅改变等价排序、内存复用与缓存淘汰。内置默认模式是 `optimized_v2`；`optimized` 保留上一轮执行方式，`reference` 保留最初逐元素复制的参考方式。源码、模式和 APK 身份写入实验记录。

## 完整排名仍然完整

手机使用稳定 LSD radix sort 排列有限 IEEE float32 logits。先把浮点位模式转换成按数值降序单调的整数键，再按 4 个字节稳定排序。`+0` 和 `-0` 归一成同一键；相同数值保留原始 ID 升序。这与原来的 `stable_sort(scores[a] > scores[b])` 输出定义相同，并不截成 top-50，也不减少词表。

非有限值依然拒绝；完整排名仍返回 262144 个 int32 ID，传输字节数仍是每位置 1 MiB。通过改变排序实现减少 CPU 比较，不宣称减少这部分通信量。

## embedding 批量输入与缓存

- 将每次查询临时构造/释放的批量 embedding/PLE 字节数组改为 worker 内的 scratch buffer，按固定槽位覆盖写入。所有有效和填充槽位每次都重写，padding 仍重复首个候选；不改变 NPU 图的物理 batch。
- embedding 缓存容量仍为 1024 项。原先满了整张清空；现在淘汰最久未访问的一项。`reset` 清空缓存、LRU 索引及 scratch。缓存只复用由同一模型、同一 token ID 计算的 embedding，不缓存依赖前缀的候选深层 KV。
- 不做全词表 embedding/PLE 预计算，不增加大型持久查表资产。scratch 增加常驻工作内存，但避免反复分配同尺寸数组，仍受编译图尺寸限制。

## 主机只精确选出所需的前 k 项

ridge 仍计算全词表相同分数；改动在 `lowest_scored_ids`，替代把全部可用候选 lexsort 后截前 k 的过程。

先用 partition 找到第 k 小分数作为边界；取全部低于边界的 ID，再按原始 ID 升序补齐边界上的并列项；最后只对选中的 k 项按 `(score, ID)` 排序。这样选出的有序列表与原先完整 lexsort 的前 k 项相同。候选提议预算、接受候选或真实 KV 距离没有变化。所有输入分数仍须有限，候选 ID 来自已排序的 `flatnonzero(available)`。

独立主机测试：262144 个随机 float32 分数中选 20 个、20 次运行中位数，完整排序约 21.025 ms、部分选择约 2.276 ms。结果相同。这只测选择步骤，不代表 ridge 全词表矩阵计算也快了相同比例。

## 验证

50 项主机测试通过。新增原生排序测试有 100 组数组，包括全词表大小、重复值、正负零、极小非零值、正负有限最大值及非有限值拒绝；主机部分选择比较 float32/float64、稀疏候选池、边界并列、k=0/1/20/全量等场景与原 lexsort 输出。

设备对照：[native_v2_validation_20261019](results/native_v2_validation_20261019/validation.json)。同 APK 的 `optimized`/`optimized_v2` 比较完整排名、不同候选数量与重复查询、CPA 第二个物理批候选的保留提交、后续查询、等长不同前缀 reset，以及 1100 个不同候选触发缓存淘汰后再次查询原候选。所有 KV 和完整排名一致，也与旧 APK 保存的共享结果一致。

短对照中 1100 个候选的 bulk 查询：worker 3.427 → 1.258 秒，embedding 准备与拼装 2.699 → 0.634 秒；两次完整排名 worker 合计 0.313 → 0.220 秒。两者均为固定顺序短样本，不能将这些比例推广为完整恢复速度。

完整样本已完成：109/109 正确，16,530 候选、388 次物理查询，查询顺序、全部已测距离和阶段记录与上轮相同；在线 117.365 → 95.636 秒（本次耗时减少 18.51%）。详见 [完整报告](results/stored_short0001_layer5_guided_native_v2/REPORT.md)。

当前 APK/optimized_v2 的 CPA pilot 已独立重采集：[cpa_gemma4_pilot_native_v2](assets/cpa_gemma4_pilot_native_v2/calibration.json)，阈值与旧资产一致。仍为 8×1024 pilot，要求 `--allow-pilot-calibration`；不能改写旧资产哈希或视作正式全词表校准。

## 当前版本的完整成对对比

当前优化版本已完成第 5 层样本的两方法交替重复（各两次）：CPA 每次 108/109，guided 每次 109/109；平均在线时间 205.948 / 95.372 秒，本样本 guided 约 2.16×。CPA 仍使用 8×1024 pilot 校准，两者验收规则不同。见 [最新成对对比报告](results/stored_short0001_layer5_cpa_vs_guided_native_v2/REPORT.md)。旧的单方法耗时记录保留供版本追踪，当前两方法速度比以这次成对结果为准。
