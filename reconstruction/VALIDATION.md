# 初版验证记录 — 2026-10-19

pilot_100_short 全部 100 条明文的真实 NPU KV 已采集并提取第 5 层，见 [数据集](../datasets/shadow_chat/plaintext_chat_100/README.md)。采集前抽定的 `0076` 做两方法各两次验证，均 15/15 正确；平均在线 CPA 422.984 秒、guided 20.461 秒。仅验证一条恢复，CPA 仍使用 pilot 校准，见 [报告](results/plaintext_short100_0076_layer5_cpa_vs_guided_native_v2/REPORT.md)。

同一样本未加密原文的第 5 层对比已完成（21 token，各两次交替）：guided 21/21、平均 14.300 秒；CPA 20/21、平均 49.543 秒。CPA 将 ` an` 接受成 ` a`。仍为小规模 pilot 校准与单样本结果，见 [明文报告及密文对照](results/plaintext_short0001_layer5_cpa_vs_guided_native_v2/REPORT.md)。

当前优化版本已完成第 5 层样本的两方法交替重复（各两次）：CPA 每次 108/109，guided 每次 109/109；平均在线时间 205.948 / 95.372 秒，本样本 guided 约 2.16×。CPA 仍使用 8×1024 pilot 校准，两者验收规则不同。见 [最新成对对比报告](results/stored_short0001_layer5_cpa_vs_guided_native_v2/REPORT.md)。

## 第二轮手机端优化

当前默认 `optimized_v2`：复用 embedding 批缓冲区、1024 项 LRU、精确完整 radix 排名；主机相同 ridge 分数用精确部分选择替代全量排序。50 项主机测试及同 APK 的缓存淘汰/重置/CPA 提交/KV/排名一致性检查通过。

完整既有第 5 层样本 109/109 正确，16,530 候选/388 次物理调用，所有已测候选的 ID、距离和阶段记录与上轮一致；在线 117.365 → 95.636 秒。详见 [报告](results/stored_short0001_layer5_guided_native_v2/REPORT.md)。当前独立 pilot 为 [cpa_gemma4_pilot_native_v2](assets/cpa_gemma4_pilot_native_v2/calibration.json)，仍未完成正式全文全词表校准。

## 手机端执行优化更新

48 项主机测试通过（含原生 C++ 的 2000 种 tensor copy 布局/局部写入对照）。同 APK 的 reference/optimized 路径在公开固定输入下，完整排名、候选 KV、跨物理批 CPA 保留提交、下一位置查询、等长不同前缀 reset 均一致，且匹配前版 APK 保存的共享结果。

完整数据库 0001 第 5 层 guided 重跑：109/109 正确，16,530 候选、388 次物理调用，候选序列与 KV 同前；在线 222.327 → 117.365 秒。详细分解见 [完整报告](results/stored_short0001_layer5_guided_native_optimized/REPORT.md)，实现见 [手机端优化](NATIVE_OPTIMIZATION.md)。当前 APK/optimized 模式的独立 pilot 校准是 [cpa_gemma4_pilot_native_optimized](assets/cpa_gemma4_pilot_native_optimized/calibration.json)，下文先前版本资产保留为历史记录。

## 持久连接更新

默认 native NPU 通信改为持久 socket，旧文件通信通过 `--transport file` 保留。47 项主机测试通过；同 APK 两种 transport 的完整词表排名、候选 KV、跨物理批的 CPA 保留提交与后续查询均逐项一致。

既有数据库 0001 第 5 层的 guided 完整重跑：109/109 正确，16,530 个候选、388 次物理查询，逐位置候选序列及接受 KV 与旧结果一致；在线 610.764 → 222.327 秒，本样本约 2.75 倍。详见 [报告](results/stored_short0001_layer5_guided_socket/REPORT.md)。单次历史对照不代表跨输入普遍倍数。

新 APK 下重新生成独立 8×1024 pilot 校准，阈值与旧资产相同；新资产为 [cpa_gemma4_pilot_socket](assets/cpa_gemma4_pilot_socket/calibration.json)。CPA 三 token 集成样本正常完成，见 [结果](results/cpa_gemma4_holdout_socket_smoke/result.json)。旧资产及历史结果未覆盖，全文全词表校准仍未完成。

## 之前的验证记录

最新实验决策：默认成对对比恢复为 `collision_plus_cpa`（官方规则的 NPU 移植）与 guided，逻辑 batch=256。基线使用独立校准双阈值；guided 使用原始数值 K+V 拼接 L2<4，这里的 4 不是 LSB。统一 L2 适配基线保留为补充选项。以下 CPA pilot 检查已完成，完整校准及正式评估尚未完成；切换默认方案不产生新的实验结果。

## 2026-10-19：已保存样本 0001，第 5 层完整密文试跑

直接使用 `datasets/shadow_chat/encrypted_chat_100/capture/0001/all_layers.gkv.gz` 的第 5 层（索引 4），覆盖 109 个密文 token，未重新采集目标 KV。完整原词表 262144、公开前缀 11 token，两边独立重建前缀；运行后才单独核对真值。

- CPA NPU 移植：108/109 正确，查询 98304 个候选，在线时间 1718.50 s。
- guided：109/109 正确，查询 16530 个候选，在线时间 610.76 s；解码密文与原 payload 字节完全一致。
- CPA 的第 98 个密文 token 把 `se` 选成 `Se`；后续 11 个 token 正确，但 KV 距离非零。guided 全部接受距离为 0。
- 本次 CPA 使用独立的 8 位置×1024 候选 pilot 校准，尚不是完整校准；两边验收标准不同，单样本固定顺序结果不能推广为正式基准或唯一性证明。

详见 [本条报告](results/stored_short0001_layer5_cpa_vs_guided/REPORT.md)、[独立评价汇总](results/stored_short0001_layer5_cpa_vs_guided/SUMMARY.md) 和逐位置数据。可复查脚本在 `experiments/stored_short0001_layer5/`。

## 2026-10-19：Collision+ CPA 移植验证

新增方法与差异详见 [CPA_BASELINE.md](CPA_BASELINE.md)。41 项主机测试通过；阈值测试通过 AST 提取官方纯函数，对分离、重叠和零 target 方差分支，在 rank=8/64 下逐项比较数值；另覆盖 K/V AND、严格阈值、首个通过者、保留 KV、逻辑批量、截断、校准断点与独立观测限制。native worker 与 instrumentation APK 构建通过。

Pixel 10 / Gemma 4 E2B / 第 5 层：

- 独立已知文本试验校准：`fixtures/cpa_gemma4_known_pilot`，BOS 后 Bitter Lesson 前 8 个文本 token；每个位置 1024 个候选，包含校准真 token，其余为固定种子随机样本。统计记录 8 个真 token 距离和 8184 个其他距离（每个分量）。
- 校准资产：`assets/cpa_gemma4_pilot/calibration.json`。原始 tau_K=0.37839773297309875，tau_V=10.247016906738281；包含源码中的 1% 后有效上界约 0.382182 / 10.349487。8 个真 token 的两个距离均为 0，采用官方分离中点分支。子集最小距离可能比全词表大，因此这些阈值不是正式全量基线阈值。
- 该校准会话（不含设备初始化）32.98 秒；8 次 bulk 的主机耗时合计 23.29 秒，其中模型执行 3.82 秒。这些是短前缀样本，不代表长前缀吞吐。将其线性外推到 113×262144 约 23.4 小时，仅是预算估计，非实测；全文全词表校准尚未执行。
- 独立恢复：`results/cpa_gemma4_holdout_smoke`。独立采集的 3 个 greedy token 全部正确，独立 `evaluation.json` 确认 3/3、0 误接受。每位置完整逻辑批 256，拆成两次物理 128，合计 768 候选；三次 `accept_query` 返回与被选查询逐字节相同的 KV，两分量距离均为 0。
- 这是容易的接口 smoke，不代表随机密文准确率、全局唯一性、长上下文性能或正式 CPA 评估。恢复显式传入 `--allow-pilot-calibration`，默认会拒绝子集校准资产。
- instrumentation APK SHA-256：`05b0b24b48845d0a4614edf43c1e4a9510941d10d6de0f5eae17c06384c868ba`。

成对入口真机检查：[`results/cpa_guided_pair_smoke/SUMMARY.md`](results/cpa_guided_pair_smoke/SUMMARY.md)。CPA 和 guided 均在独立进程中完成 3/3，分别查询 768/150 个候选；仅验证路由和评价流程，不据此给出速度结论。实际 Pixel 校准统计也输入了官方纯阈值函数，输出两个阈值与移植函数完全一致，见 `results/cpa_gemma4_holdout_smoke/official_fit_verification.json`。

## Collision+ 基线与源码审阅（移植前记录）

新增 `compare` / `evaluate-comparison`：在独立进程中运行 guided 与现有概率排序 L2 基线，统一观测、模型、词表、阈值和批量，单独读取真值评价；重复实验交替执行顺序，失败保留日志并停止配对。34 项主机测试通过，覆盖方法路由、参数不一致拒绝、独立评价、失败保留，以及基线不采样随机种子、不调用 ridge。此新入口尚未进行真实设备成对实验。

官方源码已下载至 `../references/kvcloak`，固定 commit `6b40f36edb2f337557543e7e60b10022308883d4`；旧 reference/collision.py 与官方文件哈希一致。静态核对确认：现有框架没有官方 CPA 校准和 K/V 双阈值，不能称为完整官方 Collision+ 复现。历史结果不改写，解释范围限定为统一 L2 条件下的概率排序对照，详见 [源码核对](COLLISION_BASELINE_REVIEW.md)。

## 第一层 V 自动匹配

`run --method auto` 对第一层（producer 0）使用 V 表，其他层继续 guided。第一层不查询 logits、不做 ridge、不提交恢复 token；每个位置独立扫描整张表。多个候选通过时保留全部 ID，输出 null/ambiguous，不把任意最近者当成唯一结果。

- 30 项主机测试通过，包括严格阈值边界、完整扫描/跨块次小值、重复 V 的歧义、未知前缀与非连续位置、K 改变不影响 V 匹配、模型/表完整性检查、外部插件自动分流、local 快照未环绕导入及覆盖/环绕拒绝。
- Gemma 4 原生完整表：[metadata.json](assets/first_v_gemma4_full/metadata.json)。262144 个原始 ID、每行 256 维 INT8，V 数据文件 67108992 字节（64 MiB 加 NPY 头）；构表阶段约 592.53 秒，不含前面的设备加载与模型哈希检查。这里仍运行完整编译图取第一层 V，没有实现裁剪第一层计算图。
- 构表前，抽查 8 个 ID 在空前缀、BOS、改变前缀、长度 16 的前缀下的 V，以及 query/commit 路径；最大原始数值差异均为 0。抽样验证不等于所有上下文的形式化证明。
- 独立从设备逐 token decode 采集 8 个不同 ID 的第一层 KV：[fixture](fixtures/first_v_gemma4_eight/observation/metadata.json)。默认 auto 自动读取完整表，故意传入不存在的 ADB 路径后仍恢复成功，证实缓存匹配阶段未调用设备。
- [独立评价](results/first_v_gemma4_eight/evaluation.json)：8/8 正确、误接受 0、歧义 0、未匹配 0；每个最近距离为 0，默认 V-only 原始 L2 < 4 下各有且仅有一个表内候选通过。匹配循环约 0.982 秒；这不包含模型哈希、表加载/校验、首次构表等成本，也不是性能基准。
- Gemma 3 float32 路径：[小表记录](assets/first_v_gemma3_probe/metadata.json)、[独立评价](results/first_v_gemma3_four/evaluation.json)。4 个 ID 的上下文/位置及 query/commit 检查最大差异 0；独立采集后离线恢复 4/4 正确。此项仅验证限定 4-ID 表的适配，不是完整 Gemma 3 词表恢复评估。Gemma 3 全表尚未预生成，首次默认使用会自动构建。
- 首次完整构表遇到一次 ADB 发布命令报错，但设备随后确认该序号已经执行。失败文件保留在 `assets/first_v_gemma4_full_failed_transport`。RPC 现要求匹配序号的设备确认来处理这种不确定状态，不重复发送状态变更命令；主机测试覆盖确认与无确认两种情况。成功重跑未记录发布异常。

上述 8 个 ID 是受控集成样本，并非秘密文件恢复实验或跨全体输入的唯一性证明。第一层 V-only 阈值与深层 K+V 阈值的数值不能直接视为同等判别标准。详细使用和限制见 [FIRST_LAYER_V.md](FIRST_LAYER_V.md)。

## 可扩展适配器验证

新增统一 `ModelAdapter` / `RecoveryBackend` 接口，现有实现命名为 `litert_tensor_gemma`。外部插件支持 `module:factory`，无需修改搜索代码；模型专属 manifest 解析和观测写入由适配器负责。旧配置与默认搜索参数保留。

- 25 项主机测试通过。包含独立串行 CPU 合成后端，经外部插件配置执行 fixture、恢复及 `check-adapter`；验证不同物理批量下 Collision+ 与 guided 的候选顺序、距离、选择结果一致，并能检出隐藏缓存被查询修改、commit 数值不一致等错误。
- Gemma 3 第 6 层：[接口验证](results/adapter_contract_gemma3/validation.json)。候选 `[0,1,1437,9307]`、公开前缀 `[2]`；重复查询、重排、批量对比独立单候选、commit 对比 query、commit 对比前缀重放均通过，原始数值最大差异 0。
- Gemma 4 第 5 层：[接口验证](results/adapter_contract_gemma4/validation.json)。相同候选与前缀，所有检查通过，最大差异 0。两者均使用严格零容差。
- Gemma 4 实际自动拆批：[独立评价](results/adapter_split_gemma4/evaluation.json)。逻辑批量 8、`verify` 物理槽位 4，每个位置通过两次查询合并候选结果。三 token 恢复 3/3 正确，24 次候选查询，接受 L2 为 0；记录保留物理槽位上限 4。

原生 worker 与主 APK 本次未改动。这些是有限接口样本及受控恢复验证，不是任意模型兼容性证明；串行 CPU 合成后端不计入真实模型恢复结果。扩展步骤与限制见 [ADAPTERS.md](ADAPTERS.md)。

## 模型选择验证

新增 `--model-name gemma4-e2b|gemma3-1b`，默认仍为 Gemma 4 E2B；`models` 列出本机可选配置与资产状态。选择会联动 tokenizer、embedding、设备模型路径、编译尺寸和默认采集层。主 APK 和原实验代码没有修改。

- 18 项主机测试通过，新增模型选择、Gemma 3 实际尺寸、float32 KV 保存/读取与接受值不截断检查。
- Native worker 和 instrumentation 测试 APK 构建成功。Gemma 3 使用无 PLE 路径及 float32 KV；Gemma 4 保留 PLE 与 INT8 KV。
- Gemma 3 embedding 从现有 Tensor G5 `.litertlm` 内提取，8 个抽样 ID 与设备实际 embedder 输出比较，最大绝对误差为 0：[资产验证](results/gemma3_assets/validation.json)。此抽样不等于验证全部 262144 行。
- Gemma 3 第 6 层受控三 token 恢复，独立核对 3/3 正确、无误接受，150 次候选查询、接受距离均为 0：[evaluation.json](results/model_gemma3_guided/evaluation.json)。在线约 8.10 秒，其中 decoder 约 0.737 秒；SentencePiece 解码通过。
- Gemma 4 第 5 层旧观测回归测试，独立核对 3/3 正确、无误接受，150 次候选查询、接受距离均为 0：[evaluation.json](results/model_gemma4_regression/evaluation.json)。在线约 9.60 秒，其中 decoder 约 0.435 秒；兼容旧观测中未显式记录 dtype 的 INT8 元数据。

上述两项恢复都在前 50 个概率候选中接受，未触发 ridge。这是模型适配与流程验证，不是长密文成功率、跨模型速度比较或任意模型兼容性的证明。Gemma 3 权重为量化版本，但该编译模型的 KV 为 float32；默认原始 L2 < 4 的判别能力需要按模型另做研究评估。

## 参数化更新验证

新增候选总数、预测种子数、模型资产配置、自定义原始 ID 词表及模型编译签名选择。默认配置仍为 Gemma 4 E2B、完整 262144 词表、200 个初始候选（50 概率 + 150 随机）、后续提议 20、128 槽位。

- 15 项主机测试通过，包括默认资产/参数不变、尺寸跟随 manifest、零预测种子、非连续 ID 词表、候选池先过滤再概率 top-k、非法 ID/计数和设备 batch 上限检查。
- 默认参数设备复跑：[config_defaults/evaluation.json](results/config_defaults/evaluation.json)，3/3 正确，接受 L2 全为 0。
- 自定义参数设备复跑：[config_custom/evaluation.json](results/config_custom/evaluation.json)，使用初始 8、预测 0、显式 8-ID 词表、`verify` 4 槽位、有效 batch 4；3/3 正确，接受 L2 全为 0。轨迹核对每个位置只执行两个四候选随机子批，native ready 报告 4 槽位、原始模型词表仍为 262144，手机实际模型路径由命令参数传入。
- 这份 8-ID 词表特意包含受控样本的真值，只用于参数传递与 ID 映射检查，**不能作为缩小词表后的攻击效果证据**。
- 本次参数化阶段尚未新增其他模型适配；随后模型选择阶段的 Gemma 3 适配见上节。任意新增模型仍需要兼容当前 native adapter 的编译接口。概率预测来源仍是选定 NPU 模型自身。

## 已验证的能力

这是框架集成验证，不是新的密文恢复实验。原 `experiments/` 代码、原 100 条采集与已有对比结果没有修改。

1. Native worker 构建成功，测试 APK 构建成功；主 APK 的 native bridge 显示 UP-TO-DATE。
2. 八项主机核心测试通过：严格原始 L2、全词表耗尽、显式预算停止、不提交未接受候选、同阶段多解记录、实际 KV 验收、量化尺度、跨 NPU 子批保留阶段最优，以及有 padding 的 K/V stride 读取。其中部分测试同时覆盖多个检查点。
3. Pixel 10 / Tensor G5 上生成受控三 token 观测。公开前缀只有 BOS（ID 2）；目标由同设备模型贪心生成，属于故意容易的集成样本。
4. 概率基线和引导搜索各自从零缓存、公开前缀重新开始，均恢复 3/3 个正确 token，接受距离为 0，没有误接受。引导搜索均在前 50 个概率候选阶段通过，没有触发 ridge，因此不能把此项结果视作 ridge 搜索有效性的验证。
5. 单独对真实保存的 NPU KV 执行 ridge 数值 preflight：96 个锚点、一个未参与拟合的目标，为全部 262,144 个原始候选打分；全部分数有限。84 个点与 float64 直接重建对照通过 `rtol=3e-4, atol=0.05`，预测平方 L2 最大绝对差约 0.021565。这是提议评分的数值检查，不是接受误差或恢复成功率。
6. 设备接口测试验证 1、4、22、50、72、128 个有效候选的补齐，目标行差异均为 0 LSB；全部 128 行重排后结果逐行一致；commit 后下一位置查询与重新构建前缀一致。

## 可核对的结果

| 检查 | 结果文件 |
| --- | --- |
| 三 token 独立采集 | [观测元数据](fixtures/smoke_greedy3_v3/observation/metadata.json) |
| 概率基线独立真值核对 | [evaluation.json](results/smoke_collision/evaluation.json) |
| 引导搜索首次核对 | [evaluation.json](results/smoke_guided/evaluation.json) |
| 最终版本引导搜索复跑 | [evaluation.json](results/smoke_guided_final/evaluation.json)、[逐位置轨迹](results/smoke_guided_final/trace.jsonl) |
| 设备接口补齐、重排、前缀推进 | [validation.json](results/device_contract_retry/validation.json) |
| 原始完整词表 ridge 数值检查 | [validation.json](results/surrogate_preflight/validation.json) |

首次成功 smoke 的概率基线查询 384 个候选，在线总耗时约 10.25 秒；引导策略查询 150 个候选，在线总耗时约 7.85 秒。两者仅有 3 个位置、固定运行顺序、没有热稳态控制，不能作性能对比结论。decoder 时间分别约 0.437 / 0.434 秒，其余包括 ADB、输入准备、logits 传输、CPU 操作及日志。模型/设备初始化与公开前缀预填另行记录。

最终版本在增加安装包校验和检查、明确跨子批阶段语义后，重新完成三 token guided smoke：3/3 正确、150 次有效候选查询、接受和 commit 距离均为 0，在线约 8.73 秒，decoder 约 0.453 秒。此复跑仍全部在概率种子阶段接受，未把它计作 ridge 成功案例。

## 联调中修复的问题

- Android scoped storage：`run-as` shell 不能用与正常应用相同的权限读外部模型目录。模型校验和由 ADB shell 读取，模型实际打开仍在应用进程内。
- ADB 文件 RPC：`exec-out` 不能可靠传回缺失文件的失败状态，`exec-in` 的退出也不足以保证远端小文件上传完成。改为带远端退出状态的 `shell -T`，保留原始二进制字节；命令与响应通过 rename 发布。
- 重复安装测试 APK 超时：已安装版本相同时改为校验后直接使用，`--no-install` 也验证 APK 一致性。失败检查目录 `results/device_contract` 未写入有效检查结果；重试结果在 `device_contract_retry`。
- 保留原阶段语义：150 个随机候选拆成 128+22 时，仍完整查询该阶段再选最优，新增单元测试覆盖这一点。

## 尚未验证的研究结论

- 没有用此框架完成 100 条密文、1000-token 长链或跨 512-window 边界的实际恢复评估。
- 没有验证量化误差在长恢复链上的累积、误接受后回溯或多模态前缀。
- 默认提议 embedding 来自已有 MLX 8-bit 资产，没有宣称其与 Tensor G5 编译 embedding 逐位一致。实际候选验收全部在 NPU 上计算。
- 通过 L2 阈值不构成唯一性证明；同阶段多个候选通过会被记录，词表未查询部分仍可能含有其他匹配。
- 上述历史结果使用 ADB 文件 RPC，存在明显开销；新版默认持久 socket 的验证另见 [通信优化](TRANSPORT.md)。约 60 ms 的主 decoder 调用结果仍不能代替框架端到端时间。
