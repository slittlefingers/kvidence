# 第一层 V 预计算与直接匹配

`run` 默认 `--method auto`：观测的 `layer_index == 0` 时走第一层 V 查表，其他层继续用原 guided 搜索。`--method first_v` 可显式选择；`guided` / `collision_plus` 仍可显式用于原流程。

## 实际执行方式

1. 从配置选择的同一个设备模型生成完整原始词表的第一层 V 表。使用 NPU 原生输出，不用主机权重投影近似设备结果。当前实现构表仍运行完整编译图并只保存第一层 V；没有裁剪出仅第一层计算的 NPU 图。
2. 构表前对选定 token 检查不同前缀、位置、query/commit 的一致性。适配器还必须声明第一层文本 V 在模型架构上与上下文无关。有限设备样本不构成所有输入的数值等价证明。
3. 表绑定模型文件 SHA-256、manifest、适配器、编译签名、原始 ID 集合及量化参数，记录设备运行库信息。保存完整性哈希；不完整表或不匹配表不会用于恢复。
4. 对每个观测位置，忽略 K，只比较 V。按块遍历表中全部候选，以原始数值计算 L2，严格小于 `epsilon` 才通过；默认仍为 4。
5. 一个候选通过：记录其 ID。多个通过：标记 `ambiguous` 并保留全部通过者。没有通过：标记 `unresolved`。最接近者与距离始终单独记录，不自动当成正确答案。

表生成后直接匹配只用主机 NumPy，不调用 NPU，不需要概率预测或 ridge，不重新推理已恢复的前缀。各位置独立处理；某位置有歧义时会继续处理后面的位置。

这里的度量是 **V-only 原始 L2**，不是深层流程的 K+V L2，也不是 LSB/RMS。相同的数值阈值不代表相同的误接受率。完整扫描只能列全当前表中通过阈值的候选，不能单凭阈值证明真实身份。

## 用法

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction

# 可提前构建，也可让首次第一层恢复自动构建；输出需为新目录。
"$PY" "$APP/run.py" build-v-table --model-name gemma4-e2b \
  --output "$APP/assets/first_v_gemma4_full"

# 输入第一层观测后自动使用 V 表；已有表时无需连接手机。
"$PY" "$APP/run.py" run --model-name gemma4-e2b \
  --observation /path/to/first_layer_observation \
  --output /path/to/new_recovery
```

Gemma 3 对应 `gemma3-1b` 和 `assets/first_v_gemma3_full`。当前 profile 用 `first_v_table` 指定默认缓存目录。未配置时按身份生成缓存路径。`--v-table /path/to/table` 可以显式选择已有表；显式路径不存在会报错，避免误以为已经完成预计算。

默认表覆盖 262144 个原始 token ID。`build-v-table --vocabulary ids.json` 可以构造受限表，但恢复时必须传入相同的 `--vocabulary`；该结果只说明指定集合中的匹配，不代表全词表结论。候选池不重编号。

第一层模式不使用概率种子数、随机候选数、ridge 提议数或 `--max-queries`；它完整扫描所选表。`--max-positions` 仍可限制本次处理的位置数。

构表需要模型和设备，时间单独计为 `build_seconds`，不能把离线匹配时间当成包含构表的端到端成本。失败目录与日志会保留；需修复原因后用新输出目录重建，不能将残缺文件当作完整表。更换运行库、编译版本或量化方式后，应重新验证、构表。

## 第一层观测与滑动窗口

标准观测仍是有序 K+V，匹配时仅取 V 部分。第一层允许空公开前缀以及非连续但严格递增的已知位置；深层仍要求连续位置紧跟公开前缀。

```bash
# empty_prefix.json 内容为 []；--layer 0 表示第一层。
"$PY" "$APP/run.py" prepare --model-name gemma4-e2b \
  --snapshot /path/to/all_layers.gkv.gz \
  --prefix-json /path/to/empty_prefix.json \
  --start 0 --count 100 --layer 0 \
  --output /path/to/first_layer_observation
```

现有 GHOSTKV1 导入器只接收**未环绕且逻辑位置仍能直接对应物理槽位**的第一层 local cache。快照声明 ringbuffer，或活跃位置数超过 local 缓冲区容量时，会拒绝导入，要求显式槽位映射。查表无法恢复已经覆盖掉的 V，也不能推测其原来位置。

这个模式针对文本 token；图片或音频的 adapter embedding 不一定属于文本 embedding table。不能将这些行套用文本词表并声称恢复了原内容。

## 输出与适配器扩展

- `result.json`：按观测顺序保留 token ID；歧义/未解决位置为 null，后续位置不会错位。
- `trace.jsonl`：最近候选、最小/次小距离、通过者数量、观测位置、实际比较的表行数。`queries=0` 表示在线没有模型调用。
- `candidates/000000.npy` 等：该位置全部通过阈值的原始 token ID，包含空集合。
- `protocol.json`：表身份、观测元数据、配置、阈值及匹配范围。
- `evaluate`：独立读真值进行核对，null 不算误接受；`decode` 对含 null 的结果拒绝直接拼接，避免丢失位置。

外部适配器可以增加可选 `first_v_spec(metadata, profile)` 方法，返回 `layer:0`、有序 K/V 的 `quantization`、构表公开 `prefix`、`context_independent_by_architecture:true` 及架构适用范围。若不提供该能力，第一层自动模式明确报错；不会默默套用 Gemma 假设。

后端可提供 `first_v_table(ids)` 批量返回原生 V codes，形状 `[len(ids), V_width]`。否则框架通过已有 `query` 接口拆批构表。内置 NPU 接口在设备端合并多个候选批次，每次请求最多 4096 个 ID，以减少 ADB 往返；实际并行槽位仍由编译图决定。
