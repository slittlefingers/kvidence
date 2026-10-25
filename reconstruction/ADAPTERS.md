# 模型适配器扩展指南

框架将搜索策略、模型执行和观测导入分开。新增模型不应在 `search.py` 中增加模型名称判断。

## 现有实现与边界

内置 `litert_tensor_gemma` 适配 Pixel Tensor G5 上当前 Gemma 3/4 的 LiteRT 分段计算图。张量名称、RoPE/mask、K/V 排列、可选 PLE 等假设封装在该适配器与 `npu.py` / `native/worker.cc` 中，并未宣称任意 LiteRT 模型可直接使用。

内置后端默认使用持久 socket，CLI 的 `--transport file` 可保留旧文件通信。这个选项仅在显式设置时传给外部插件；插件可自行实现或拒绝它。内置 `probabilities()` 在手机稳定排序，仍返回完整原 ID 排名，接口语义不变；详见 [通信优化与验证](TRANSPORT.md)。

内置后端默认 `native_mode=optimized_v2`，启用张量块复制、位置上下文复用、embedding scratch/LRU 和等价 radix 排名；`optimized` 保留上一轮，`reference` 保留逐元素复制参考路径。该选项仅在显式指定时传入外部插件。上下文缓存必须在 `reset` 时失效，不能只以 position 判断两个不同前缀。模式写入设备记录并绑定 CPA 校准身份，见 [第二轮手机端优化](NATIVE_OPTIMIZATION_V2.md)。

- 接口和计算语义兼容的新模型：添加模型配置与匹配资产即可，通过 `--model-config` 加载，不要求修改命名模型列表。
- 不同张量布局、计算图或运行时：实现新适配器，复用搜索及结果记录。
- 新适配器在能正确查询候选 KV 的前提下，允许先实现串行版本；不要求支持 128 并行。
- 目前研究任务仍是连续文本 token 恢复；新增适配器不自动解决多模态、缺失位置、滑动窗口历史已丢失等问题。

## 配置与插件加载

现有配置增加：

```json
{
  "adapter": "litert_tensor_gemma",
  "adapter_options": {}
}
```

旧配置省略 `adapter` 时仍选择上述内置实现。外部实现可使用 `"adapter": "my_backend:Adapter"`：框架从 Python 导入路径中加载 `my_backend`，调用无参工厂 `Adapter()`。用 `PYTHONPATH=/path/to/plugins` 或安装本地包即可，不需要编辑框架注册表。

插件是可执行 Python 代码，只加载自己信任的实现。`adapter_options` 是传给适配器的自定义配置对象，其中路径由插件自行解析；公共资产路径仍相对模型配置 JSON 解析。

公共必填项：`name`、`model`、`manifest`、`tokenizer`、`embeddings`、`vocabulary`（全词表为 null）。`default_layer` 默认 4，使用从零开始的 producer 编号。LiteRT 专属项 `device_model`、`batch_signature`、`mask_token_id` 可供该适配器使用，其他插件可以忽略；当前默认值仅为旧配置兼容，不是通用模型约定。

目前模型身份采用单一 `model` 文件的 SHA-256；多文件运行时应由适配器额外校验、记录各权重分片。框架默认提议器仍使用完整 float32 embedding 表及现有 PyTorch/MLX ridge 实现，新增后端不自动改变主机评分依赖。

## ModelAdapter 接口

定义见 [adapters.py](kvrecon/adapters.py)。工厂返回的对象实现以下方法：

| 方法 | 职责 |
| --- | --- |
| `read_manifest(path)` | 读取自有 manifest，返回适配器理解的元数据；无需采用 LiteRT schema |
| `dimensions(metadata, profile)` | 返回 `vocab_size`、`embedding_dim`、`context_capacity`、`batch_capacity` |
| `capabilities(metadata, profile, layer)` | 返回 `Capabilities`：上述尺寸、可输出 producer 编号、选定层 KV dtype/宽度 |
| `open(metadata, profile, layer, output, model_hash, **options)` | 构建支持上下文管理的后端；负责加载、核对模型身份、写设备记录和关闭资源 |
| `import_observation(snapshot, output, metadata, model_hash, prefix, start, count, layer)` | 从该运行时快照导入统一观测；不支持的格式明确抛错 |
| `save_fixture(output, vectors, positions, prefix, layer, metadata, model_hash, source)` | 将独立采集阶段输出写成统一观测 |

`open` 当前接收公共执行选项 `adb`、`serial`、`install`；CPU 等后端可忽略，自己的选项从 `profile["adapter_options"]` 读取。`capabilities.layers` 表示图可输出的 producer，不代表所有层都保存了完整历史。

## RecoveryBackend 接口

定义见 [backend.py](kvrecon/backend.py)。保持原始 token ID，不重编号：

| 方法/属性 | 必须满足的行为 |
| --- | --- |
| `reset(prefix)` | 替换已有前缀，`position == len(prefix)`；清除旧缓存状态 |
| `probabilities()` | 返回按下一 token 概率降序排列的**完整 ID 序列**，不是概率数值；并列稳定排序 |
| `query(ids)` | 返回 `[len(ids), kv_width]` 原生 KV codes；候选共享同一前缀，互相不可见；不改变 position、缓存和下一 token 分布 |
| `values(codes)` | 返回相同形状的原始数值 K+V，供 CPA、原始数值模式和接口检查；整数 LSB 搜索绕过此方法，直接安全扩展原始 codes |
| `commit(id)` | 返回 `[1, kv_width]` 并将该 token 加入前缀，position 恰好加一 |
| `vocab_size` / `batch_capacity` | 原始词表大小 / 单次真实候选调用上限；串行后端上限为 1 |
| `setup_seconds` / `decoder_seconds` | 初始化时间 / 累计模型执行时间，不把通信时间计为 decoder 时间 |
| `__enter__` / `__exit__` | 资源创建与释放；异常也需要清理 |

K+V 必须按与观测相同的展平顺序、同一位置编码约定返回。物理 KV 查询若会更新底层缓存，适配器必须隔离或恢复状态。复制 ID 到一个普通 prefill 序列不等于隔离候选。

`CandidateBatchBackend` 将逻辑候选批次拆成底层支持的大小，再按输入顺序合并。它本身不修改搜索选择规则：Collision+ 保留概率顺序下首个通过者；guided 保留完整阶段的最小距离。串行退化意味着多次真实推理，不会产生虚假的并行加速。`physical_slots` 记录物理上限，`valid_candidates` 可能是多个底层调用合并后的逻辑批次数。

## 统一观测与实验记录

适配器可调用 `save_canonical_observation`，保存有序 K+V、连续位置、公开前缀、模型哈希、量化元数据与来源。当前标准观测支持 INT8 或 float32；其他原生表示需由适配器显式转换并记录，不能直接重解释字节。

恢复运行仍禁止读取 `private_eval` / 原始 `datasets`。目标 KV 仅供主机评分，不得作为执行器前缀缓存。`protocol.json` 新增适配器名称与能力描述，并保留配置、资产、源码哈希。外部插件不在框架目录的源码哈希扫描范围内，发布实验时需另外固定插件版本或记录源码哈希。

## 新适配器验收

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
"$PY" "$APP/run.py" check-adapter --model-name gemma3-1b \
  --prefix-json "$APP/examples/public_prefix.json" \
  --candidate-ids-json "$APP/examples/contract_candidates.json" \
  --output /path/to/new_contract_result
```

外部模型将 `--model-name` 换成 `--model-config /path/to/profile.json`。检查包括重复查询、候选重排、批量对比从相同前缀开始的单候选查询、commit 对比 query、commit 后对比重建整个前缀。还检查 position 与概率排序保持不变。设备打开或接口检查失败会写 `passed:false` 并以错误退出；配置与输入预检查错误直接退出。

默认 `atol=rtol=0`；如硬件路径存在数值差异，可显式设置容差，报告同时记录实际最大差异。该容差用于接口验证，与攻击接受阈值 `epsilon` 是不同参数。应在不同前缀长度、候选批量、目标层和精度下执行；有限测试不能证明对所有输入都相同。

主机测试提供一个不依赖 LiteRT 的串行 CPU 合成适配器，通过外部插件配置完成 fixture、恢复、验收命令，以验证扩展入口。它不是真实语言模型，也不构成恢复成功率证据。参考 [support_cpu_adapter.py](tests/support_cpu_adapter.py)。

第一层 V 查表另有可选的 `first_v_spec` / `first_v_table` 能力，不要求所有适配器实现。声明要求、原生量化对齐与自动路由见 [FIRST_LAYER_V.md](FIRST_LAYER_V.md)。

## CPA 可选扩展

`collision_plus_cpa` 还要求后端实现 `begin_candidates()` 与 `accept_query(token)`。前者清空本逻辑批的候选缓存，之后多次 `query` 保留各候选所有层的 KV；后者提交所选候选的缓存，返回与查询字节一致的目标层 K+V。`CandidateBatchBackend.query_retained` 自动跨物理批保留这些状态。未实现此能力的插件明确报错，不静默退回 decode 重算 KV。

可选 `query_bulk(ids)` 用于校准，最多 4096 个原 ID、不能推进前缀；不实现则自动经普通物理拆批。Gemma NPU 用设备内部循环减少 ADB 往返。`cache_acceptance_policy` 字符串记录额外 logits 重算等差异；详见 [CPA 基线](CPA_BASELINE.md)。

整数 LSB 恢复要求后端提供 `quant`（与 observation.quantization 相同布局的量化元数据），以核对宽度、dtype、scale、zero-point。`query`/`commit` 必须返回 native 整数代码，不能返回预先反量化的浮点值。`SearchConfig.distance_space` 控制搜索、拟合和 commit 的单位；直接调用 API 时，targets 必须处于同一空间（可用 `load_observation(..., distance_space="auto")`）。旧 `load_observation()` 默认仍返回原始数值以兼容历史分析脚本；新 CLI 显式选择单位。
