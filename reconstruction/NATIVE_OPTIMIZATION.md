# 手机端张量复制与上下文复用

本文记录第一轮优化（模式 `optimized`）。当前默认模式为 `optimized_v2`，新增改动及当前校准见 [第二轮优化](NATIVE_OPTIMIZATION_V2.md)，本文的完整结果与校准资产保留为历史记录。

本次优化保留模型、完整词表、候选搜索与阈值。已有 KV cache 原本就常驻设备；这里进一步减少执行器在历史 KV 之外的重复准备和逐元素复制。

## 改动

1. `TensorCopy` 根据真实 shape 和 stride 找到连续的逻辑后缀，将原来逐元素 `memcpy` 合并成连续块。整个张量连续时直接整块复制；带 padding 或非连续布局按连续块分段，最差回退为单元素。部分 KV 写入同样按块处理，保持 padding 和其他位置不变。
2. 同一 position 的 RoPE 和 mask 只生成一次；候选 batch 的固定 mask/RoPE 输入只写入一次。成功 commit 后 position 改变即重新生成；`reset` 强制清空两种缓存，即使新前缀长度与旧前缀相同也不能复用。
3. 候选 embedding 的缓存命中返回引用，拼装批输入预留空间，避免额外 Data/Bytes 副本。原有 1024 项 embedding 缓存上限不变，仍保留所有原 token ID。
4. CPA 保存候选所有层 KV 后，回传目标层时直接使用已保存的切片，避免再次拆出全部层。CPA 的选中规则、保留 KV 和额外 decode logits 流程不变。

`--native-mode optimized` 为内置后端默认；`--native-mode reference` 保留逐元素复制、每批重建上下文和旧式临时复制，用于同 APK 的受控对照。reference 是加了计时的参考路径，不声称编译产物与历史 APK 完全相同。两种模式均可配合 `--transport socket` 或 `file`。外部插件仅在显式指定时收到 native_mode 选项。

设备记录和成对实验一致性检查包含 native_mode，CPA 校准身份也绑定 native_mode。新 APK 下需要重新运行校准，不能直接替换旧校准文件的哈希。

## 细分计时

每次 RPC 的 `profile_seconds` 与 `profile_counts` 随响应写入 `device/rpc.jsonl`。

| 字段 | 范围 |
| --- | --- |
| `phase_query_context` | RoPE/mask 生成、构造批量输入及写入；命中缓存时仅检查 |
| `phase_query_embedding` | embedding/PLE 查询、缓存访问及候选批拼装 |
| `phase_query_write` | 候选 embedding 输入张量写入 |
| `phase_query_read` | 读取所选层输出，或 CPA 所有层输出 |
| `phase_query_pack` | 拆分候选、保存 CPA 切片和打包返回值 |
| `phase_commit_inputs` | decode 上下文与 embedding 准备、写入 |
| `phase_commit_outputs` | decode 输出/目标 KV、logits 读取与打包 |
| `phase_commit_append` | 向 decode/prefill KV 缓冲区追加已选位置 |
| `model_<signature>` | 对应 LiteRT compiled model 运行调用 |
| `buffer_read_*` / `buffer_write_*` | 更低层的分配、锁定、复制、解锁计时 |

这些字段存在**嵌套**：例如 `model_decode_embedder` 和部分 buffer 时间已包含在 `phase_query_embedding` 中，不能全部相加。高层 phase 与主 decoder/prefill 调用可用于主要步骤的分解；低层 buffer 指标用来解释其中的内存操作。`phase_commit_append` 直接锁定并局部写缓存，没有重复计入通用 buffer 读写指标。`worker_seconds` 仍包含其余管理、分配释放和协议准备开销。

`decoder_seconds` 的历史定义不变：只统计主 decoder/prefill 运行调用，不是全部模型子图，更不是纯 NPU kernel 计时。

## 有限等价性检查

48 项主机测试通过，其中 C++ 原生复制测试覆盖 2000 个布局：INT8/float32 元素宽度、连续/带 padding/置换 stride、singleton 维度、部分范围写入，并和独立逐元素实现比较所有字节（包括 padding）。

设备检查：[native_validation_20261019](results/native_validation_20261019/validation.json)。同一 APK、相同公开前缀与固定候选，先 reference 后 optimized。检查完整词表排名、多次重复、不同候选数量、第二个物理批候选的 CPA 提交、提交后下一位置，以及换成不同但等长前缀再 reset 的缓存失效。共享检查还和前一次 APK 保存的字节结果比较，全部一致。检查不读取待恢复真值。

九次候选查询混合普通/CPA 场景，worker 总时间 reference 4.054 秒、optimized 1.474 秒。嵌套的通用 buffer 读写复制总计从 3.154 秒降到 0.141 秒；输出读取阶段从 2.661 秒降到 0.161 秒。这是一次固定顺序的小型检查，不能推广为每个位置的固定加速比例。其他阶段受运行状态影响也有波动。

完整既有样本已完成：109/109 正确，逐位置查询序列与接受 KV 均与前两版一致；在线 222.327 → 117.365 秒，16,530 个候选、388 次物理查询不变。完整计时和限制见 [报告](results/stored_short0001_layer5_guided_native_optimized/REPORT.md)。

当前 APK/optimized 模式的独立 pilot 校准为 [cpa_gemma4_pilot_native_optimized](assets/cpa_gemma4_pilot_native_optimized/calibration.json)，仍是 8 位置 × 1024 候选，需要显式 `--allow-pilot-calibration`；没有替代正式全词表校准。旧校准资产保留作历史记录。
