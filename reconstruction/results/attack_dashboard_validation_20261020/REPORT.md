# Ghost Attack：单条真实 NPU 验证

本次只验证同一条明文样本的 guided 与 Collision+ CPA，未启动 100 条明文＋100 条密文批量实验。Pixel 10、Gemma 4 E2B Tensor G5、原始 INT8 KV、第 5 层，完整原始词表。目标为已有 after-response 快照中的 21 个 user-prompt token，不包含回复 token。

## 结果

| 方法 | 实际正确 token | 误接受 | 评估候选数 | 在线恢复时间 | 主模型调用时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| guided | 21/21 | 0 | 1,820 | 23.493 s | 4.211 s |
| Collision+ CPA 官方规则 NPU 移植 | 20/21 | 1 | 22,784 | 46.769 s | 13.678 s |

这是启用实时事件和手机监控的单次功能验证，不是重复、随机交替顺序的性能基准；不据此估计普遍加速比。在线时间不含模型初始化、公开前缀处理和独立校准。

两种方法与各自先前的 `plaintext_short0001_layer5_cpa_vs_guided_native_v2/001_*` 逐位置核对，接受的 token ID、候选 ID 顺序、候选距离及提交距离均完全相同。遥测没有改变本样本的搜索决策。

## CPA 规则与误接受

CPA 使用 `kvrecon/cpa.py` 中按 Shadow in the Cache / kvcloak commit `6b40f36edb2f337557543e7e60b10022308883d4` 移植的统计和验收逻辑。按模型概率顺序查询，选择首个同时通过 K/V 阈值的候选；逻辑 batch=256，物理 NPU batch=128。它不使用 guided 的拼接原始 KV `L2 < 4`。

测试 APK 哈希已变化，因此重新采集独立的 8 位置×1,024 候选 pilot 校准，保存在 `../../assets/cpa_gemma4_pilot_dashboard_20261020/`。新旧校准候选数值文件哈希及拟合阈值完全一致。**这仍是 pilot 子集校准，不是完整论文规模校准。**

含官方 1% 容差的严格阈值：

- K L2 < 0.38218171030282977
- V L2 < 10.349487075805664

第 2 个目标 token（绝对位置 5）真实为 ` an`（ID 614），CPA 接受了 ` a`（ID 496）。候选的 K/V 距离分别为 0.3461188972 / 7.8659653664，均通过阈值；拼接 L2 为 7.8735766605。后续位置使用 CPA 自己接受的前缀继续恢复。

因此 UI 的 **Accepted 21/21** 只表示全部位置通过相应验收规则。独立评价结果是 **Verified 20/21，False acceptances 1**，不能把前者称为全部正确。

## 手机实时验证

- 手机运行时已观察到 CPA `Reconstruction in progress`、第 1 个目标 token、第 25 批、6,400 个候选，证明实际运行中的批次已同步到 UI。
- 最终读取手机 `files/attack_live.json` 确认：`collision_plus_cpa`、`complete`、22,784 个候选、独立评价 20/21、误接受 1。
- guided 的位置历史页显示第 21 个目标 token ` goa`：第 8 批接受 ID 228911，该候选为该位置第 286 次查询，原始 KV L2=0。
- 对应完整记录仍保存在主机 `events.jsonl` / `trace.jsonl`；手机档案保留，可离线查看。
- 两套原始输出：[guided](guided/) 与 [collision_plus_cpa](collision_plus_cpa/)。各自的 `evaluation.json`、`result.json` 和 `recovered.txt` 已生成。

## 代码验证

Android debug APK 编译并安装成功。框架测试共 80 项，78 项通过、2 项跳过；新增测试覆盖候选阶段选择不受遥测影响、提交不通过时不标成已恢复、异常终止事件以及跨位置候选计数。CPA 本次未修改搜索代码。
