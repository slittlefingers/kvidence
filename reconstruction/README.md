# KVidence reconstruction

新增终端交互工作台：项目根目录运行 `./reconstruct`，支持采集控制、上下文切换和恢复进度查看，详见 [终端使用说明](TERMINAL.md)。

Local case assistant: describe `create case1` in the workspace. A local MLX model asks for missing metadata and schedules registered ridge-guided, Collision+, and exhaustive recovery templates. See [Case workflow](CASES.md).

新增 Ghost **Attack** 实时页面：启用 `--live-events` 并运行独立 ADB 同步程序，可查看每批候选、位置历史、接受结果与独立评价。使用方法及单条验证见 [Attack Dashboard](ATTACK_DASHBOARD.md)。

pilot_100_short 全部 100 条明文的真实 NPU KV 已采集并提取第 5 层，见 [数据集](../datasets/shadow_chat/plaintext_chat_100/README.md)。采集前抽定的 `0076` 做两方法各两次验证，均 15/15 正确；平均在线 CPA 422.984 秒、guided 20.461 秒。仅验证一条恢复，CPA 仍使用 pilot 校准，见 [报告](results/plaintext_short100_0076_layer5_cpa_vs_guided_native_v2/REPORT.md)。

同一样本未加密原文的第 5 层对比已完成（21 token，各两次交替）：guided 21/21、平均 14.300 秒；CPA 20/21、平均 49.543 秒。CPA 将 ` an` 接受成 ` a`。仍为小规模 pilot 校准与单样本结果，见 [明文报告及密文对照](results/plaintext_short0001_layer5_cpa_vs_guided_native_v2/REPORT.md)。

当前优化版本已完成第 5 层样本的两方法交替重复（各两次）：CPA 每次 108/109，guided 每次 109/109；平均在线时间 205.948 / 95.372 秒，本样本 guided 约 2.16×。CPA 仍使用 8×1024 pilot 校准，两者验收规则不同。见 [最新成对对比报告](results/stored_short0001_layer5_cpa_vs_guided_native_v2/REPORT.md)。

面向 Pixel 10 的 KV 引导文本恢复框架，可选择 Gemma 4 E2B 或 Gemma 3 1B。这里将已有搜索方法与 NPU 候选接口连接起来，独立于 `experiments/` 中的原始实验。

初版已完成主机测试、设备接口测试和三 token 端到端 smoke，详见 [验证记录](VALIDATION.md)。这些检查不等同于长密文恢复评估。

扩展入口现已独立为 `ModelAdapter` 与 `RecoveryBackend`，详见 [模型适配器扩展指南](ADAPTERS.md)。兼容模型可以只增加配置；不同计算图通过 `module:factory` 插件接入，无需修改搜索算法。新增 `check-adapter` 命令验证接口一致性。

已增加按恢复出的媒体开始 token 分派到 `image.py` / `audio.py` 的框架入口，Gemma 4 默认规则由 tokenizer 解析。文本搜索保留；支持媒体的后端完成恢复和重放后返回文本循环。**当前内置 NPU 后端尚未实现媒体优化/重放接口，实际遇到媒体会明确停止为 `unresolved_media`。** 接口、已知区间条件、验证范围见 [多模态分派](MULTIMODAL.md)。

两个媒体模块现已包含旧 Gemma/MLX pilot 的实际反演算法：联合优化整块 adapter 输入、固定模型和 adapter、匹配第 5 层 KV，并做完整模型重放。图片与音频分别选择视觉/音频 adapter，支持独立从旧公开观测运行；这不等同于 NPU 媒体重放已经完成。

已完成图片、音频各两次 3000 步迁移验证及新旧 forward/梯度/更新核对，见 [算法迁移报告](results/media_algorithm_port_20261020/REPORT.md)。

第一层观测现在自动使用预计算的 V 表直接匹配；深层继续使用 guided 搜索。首次构表从实际设备输出生成，后续匹配不需要 NPU。用法、歧义处理和滑动窗口限制见 [第一层 V 查表](FIRST_LAYER_V.md)。

当前对比基线采用按官方规则移植的 **Collision+ CPA**：`compare` 默认使用 `collision_plus_cpa`，要求独立校准资产，K/V 分别通过校准阈值及官方 1% 容差；默认 rank=8、成对运行逻辑 batch=256（NPU 物理 128）。我们的 guided 方法对整数 KV 默认使用**整数刻度（LSB）空间中，拼接 K+V 的 RMSE 严格小于 `epsilon`**；不反量化。当前 CLI 默认 `epsilon=0.125`、`distance-metric=rmse`。RMSE = L2 / √d，d 为匹配的 K/V 坐标总数；1024 维时与旧 L2 < 4 等价。本轮尚未重新运行手机实验。

研究定位是基于已有 KV 碰撞攻击改进候选搜索。主对比保留双方各自的验收规则，同时报告准确率和成本；统一 RMSE 的概率排序适配版通过 `--baseline collision_plus` 显式选择，用作补充对照。官方规则的 NPU 移植差异、校准命令及已完成范围见 [CPA 基线](CPA_BASELINE.md)。现有校准资产仅为 pilot，正式的全文全词表校准尚未完成。

NPU 通信默认使用 ADB 转发的持久 socket，避免每次查询启动多条 ADB 命令；`--transport file` 可选择旧文件通信。候选选择与阈值不变，完整词表排序在手机执行，目标层当前位置的候选 KV 返回主机评分。实现及验证见 [通信优化](TRANSPORT.md)。

手机端默认 `optimized_v2` 启用连续块复制、位置上下文复用、embedding scratch/LRU 和等价基数排序；主机按相同分数精确选出前 k 个候选。`--native-mode optimized` 保留上一轮，`reference` 保留原始执行方式作对照。见 [第二轮优化](NATIVE_OPTIMIZATION_V2.md) 与 [第一轮细分计时](NATIVE_OPTIMIZATION.md)。

## 可配置参数（guided 默认保持原设置）

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--method` | `auto` | 第一层用 `first_v`，其他层用 `guided`；也可显式选择 `collision_plus_cpa` 或旧 `collision_plus` |
| `--v-table` | 模型配置的缓存位置 | 第一层已生成 V 表；省略且无默认表时自动构建 |
| `--initial-candidates` | 200 | 引导搜索初始种子候选总数，非整个搜索的查询上限 |
| `--prediction-candidates` | 50 | 初始候选中来自模型条件概率排序的数量 |
| 随机候选数 | 150 | 自动计算：初始总数减预测候选数 |
| `--proposals` | 20 | ridge 每轮提出的候选数 |
| `--batch-size` | guided/旧基线 128；CPA 256 | 逻辑候选批量；自动按后端物理上限拆分，不改变编译图槽位数 |
| `--transport` | 内置 NPU 后端为 `socket` | 持久连接；`file` 选择旧 ADB 文件通信，用于复核 |
| `--native-mode` | 内置 NPU 后端为 `optimized_v2` | `optimized` 保留上一轮；`reference` 保留逐元素复制参考路径；主机精确前 k 选择对各模式共用 |
| `--model-config` | [configs/gemma4_e2b.json](configs/gemma4_e2b.json) | 模型名称、模型/手机路径、编译签名、tokenizer、embedding 及可选词表 |
| `--model-name` | `gemma4-e2b` | 命名选择：`gemma4-e2b` 或 `gemma3-1b`，与 `--model-config` 二选一 |
| 配置 `adapter` | `litert_tensor_gemma` | 执行及观测导入适配器，也可指定外部 `module:factory` |
| `--model` / `--device-model` | 现有 Gemma 4 E2B IT Tensor G5 模型 | 本机模型与手机实际加载路径，SHA-256 必须一致 |
| `--manifest` | 现有 NPU 探针 manifest | 与选定编译模型对应的签名、尺寸与量化参数 |
| `--batch-signature` | `prefill_128` | 候选接口；Gemma 4 也可使用 `verify`（4 槽位），Gemma 3 不提供该签名 |
| `--tokenizer` | 当前 Gemma 4 tokenizer | 文本解码词表；路径与哈希写入协议 |
| `--embeddings` | 当前完整 MLX 8-bit embedding 表 | 供 ridge 提议使用，行数/维度必须匹配模型，原 token ID 不变 |
| `--vocabulary` | 不传：完整模型词表 | 可指定原始 token ID 的 JSON 列表，限制候选范围 |
| `--vocab-size` | 自动读取，当前 262144 | 可指定预期大小做校验；必须等于模型 logits 维度，不用于截断或伪造词表 |
| `--epsilon` | 0.125 | guided/适配基线拼接 K+V RMSE 阈值，单位由 distance-space 决定；CPA 忽略此参数 |
| `--distance-metric` | rmse | 默认按维度归一化；l2 用于复现旧距离，CPA 保留官方 L2 判定 |
| `--distance-space` | auto | 整数 KV 使用 lsb；浮点 KV 使用 original。original 可显式重跑旧指标；CPA 固定保留原指标 |
| `--seed` | 20261019 | 随机种子，每个位置加其相对索引 |
| `--max-queries` | 无 | 可选显式资源上限，默认没有 600 次限制 |

例如把初始候选改成 400 个，其中 100 个来自模型预测，剩余 300 个随机抽样：

```bash
"$PY" "$APP/run.py" run \
  --observation "$APP/observations/sample0001" \
  --initial-candidates 400 --prediction-candidates 100 --proposals 32 \
  --output "$APP/results/sample0001_400_100"
```

允许 `--prediction-candidates 0`（全部随机种子），也允许预测数等于初始总数（无随机种子）。预测数不能超过初始总数；guided 初始总数至少为 2。候选池较小时自动截到可用数量，实际数量逐位置记录。若概率阶段已经通过阈值，仍会提前停止，不强制凑满初始候选总数。`collision_plus` 沿用全候选概率排序，不使用 guided 的种子配额。

自定义候选词表的文件格式：

```json
{"token_ids": [1437, 9307, 21029, 236746]}
```

用 `--vocabulary /path/to/token_ids.json` 传入。ID 必须唯一且位于模型原始词表内。概率 top-k、随机采样与 ridge 选择均限制在这个集合中；embedding 表仍保留完整原始行，**不重新编号**。`protocol.json` 记录模型词表大小、候选池大小/哈希、种子配额和最终解析的模型配置。

模型配置 JSON 中的相对路径以 JSON 所在目录为基准；命令行同名参数覆盖配置文件。更换模型时应成套提供匹配的 `.litertlm`、manifest、tokenizer、embedding 和 Android 路径。模型词表大小、embedding 维度、上下文容量与候选槽位从编译签名读取；手机启动后再次核对。**当前 native adapter 要求兼容的 LiteRT 分段模型、decode/embedding/aux 接口及 INT8 或 float32 KV 布局；PLE 按模型是否提供决定加载。配置化不代表任意模型已经适配。** 模型预测候选目前来自所选 NPU 模型自己的 logits，尚未接独立的外部预测模型。

## 选择模型

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
"$PY" "$APP/run.py" models

"$PY" "$APP/run.py" run --model-name gemma4-e2b \
  --observation /path/to/gemma4_observation --output /path/to/new_gemma4_result
"$PY" "$APP/run.py" run --model-name gemma3-1b \
  --observation /path/to/gemma3_observation --output /path/to/new_gemma3_result
```

| 模型名称 | KV 存储 | 词表 / embedding 维度 | 编译上下文容量 | 默认采集层（从 1 开始） |
| --- | --- | --- | ---: | ---: |
| `gemma4-e2b`（默认） | INT8 | 262144 / 1536 | 4096 | 第 5 层 |
| `gemma3-1b` | float32 | 262144 / 1152 | 1280 | 第 6 层 |

模型选择同时切换本机/手机模型路径、编译签名、tokenizer、embedding 资产和默认采集层。`--layer` 仍可覆盖默认采集层。观测绑定模型 SHA-256；不能用 Gemma 4 的观测配 Gemma 3 恢复。默认候选配额和 `RMSE < 0.125` 不随模型自动改变；不同模型中这个阈值的实际判别能力仍需实验评估。

Gemma 3 的 tokenizer 和 embedding 资产从本机已有的 Tensor G5 `.litertlm` 提取，无需另下模型：

```bash
"$PY" "$APP/prepare_gemma3.py"
```

本工作区已完成提取；该命令拒绝覆盖已有资产目录。提取记录在 `assets/gemma3_1b/provenance.json`。它解包 SentencePiece tokenizer，并对模型内 INT8 embedding 按行 scale/zero point 反量化，再应用模型内 embedding scale，得到用于候选提议的 float32 表。设备端另行抽样检查与实际 embedder 输出的差异。

## 第一版范围

- 目标：连续文本 token，对比第 5 层等全局 KV producer 的 K+V。
- 观测：已有 `GHOSTKV1` 真实运行时 INT8/float32 缓冲区快照，或单独生成的设备端受控测试数据。
- 前缀：深层仅接受明确已知的公开 token ID，随后由恢复出的 token 自行推进。第一层 V 查表不需要公开前缀。不会把观测中的真实前缀缓存加载到恢复执行器。
- 候选：默认完整原始词表 `0..262143`，保留特殊 token；显式传 `--vocabulary` 才限制集合。
- 查询：默认 Pixel NPU `prefill_128`，共享前缀、重复位置编码、mask 隔离。单次物理调用有效候选最多 128 个；逻辑批次超过后端上限时自动拆分。不足时复制一个有效候选填充额外槽位，屏蔽跨槽位 attention，丢弃填充行。没有将“不存在的 padding 行”视作一个额外待接受候选。
- 拟合、评分、commit 验收：整数 KV 直接使用存储整数，先扩展到 float64 防止 INT8 减法溢出，不乘 scale、不减 zero-point。计算拼接 K+V 的 **LSB RMSE < epsilon**，即 L2 除以 √d。模型输出仍为真实整数 KV；拟合预测允许非整数，不取整。目标与候选必须具有相同的对应量化网格，包含类型、符号、scale 和 zero-point。浮点 KV 单独使用原始数值距离，不称为 LSB。
- 默认无查询次数上限。全词表耗尽未通过就标记 `unresolved`；显式资源预算耗尽也标记未解决，不接受“最接近但不过阈值”的 token。

这不是完整的取证采集工具。深层流程不支持缺失位置、旋转缓存地址映射、多模态前缀重建或超过所选编译模型上下文容量的输入（默认模型为 4096），也没有加入 beam search / 回溯。通过阈值表示候选被接受，不代表数学上的唯一性或已验证的真实 token 身份。

## 结构

```text
GHOSTKV1 / 受控 NPU 数据
    ↓ prepare（只提取目标层、目标位置）
observation/{observed.npz, metadata.json}
    ↓
主机：概率排序 → 提出候选 → NPU 查询 → KV 整数刻度 RMSE → 接受/继续搜索
                              ↑                    ↓
手机常驻模型：共享已重建前缀 ← decode 提交已接受 token
    ↓
trace.jsonl + result.json
    ↓ 独立 evaluate 命令才能读取 private_eval/truth.json
evaluation.json
```

| 文件 | 用途 |
| --- | --- |
| `kvrecon/observation.py` | 按真实 stride 读取 K/V、保留量化元数据、移除其他位置和层 |
| `kvrecon/search.py` | 概率基线与分阶段 KV 引导策略；实际查询后才验收 |
| `kvrecon/surrogate.py` | 复用已有稳定 QR ridge 提议器 |
| `kvrecon/engine.py` | 按位置恢复、接受后提交、记录失败/歧义与计时 |
| `kvrecon/npu.py` | ADB 通信、设备和模型校验、常驻 worker 生命周期 |
| `native/worker.cc` | 底层 LiteRT C API；常驻 NPU 模型；独立 query 与 commit |
| `tests/test_core.py` | 验收边界、耗尽、歧义、错误提议与快照布局测试 |

主 APK 的交互聊天代码和原有运行库没有修改。新增的 `ReconstructionDeviceTest` 与 JNI `.so` 仅打包进配套 instrumentation 测试 APK。运行新 worker 时不要同时使用主 APK 推理；共享进程/硬件上的多个任务不是本框架已验证的并发模式。

## 搜索方法

官方源码已按固定 commit 下载核对，详见 [Collision+ 基线审阅](COLLISION_BASELINE_REVIEW.md)。当前 `collision_plus` 保留概率搜索顺序，但没有官方的 CPA 阈值校准，且使用不同的 K+V 拼接 RMSE 判据；应作为统一 RMSE 下的适配对照，不能标作完整官方复现。论文对齐版本及新的 `compare` 入口见 [CPA 基线](CPA_BASELINE.md)。

`guided` 默认沿用已有方法：先查概率前 50 个，再查其余词表均匀无放回抽样的 150 个；仍未通过时，使用已查询的 embedding→实测 KV 对拟合 ridge，按预测距离提出 20 个未查询候选，再交给 NPU 实测。一个阶段超过 128 时拆批，阶段间保留提前停止；不把“200 个初始候选”改成新的硬预算。`--proposals` 可调整后续提议数量。

实际复用的是 `../experiments/ciphertext_full_vocabulary/stable_surrogate.py`。最多 512 个回归锚点是拟合规模限制，不是搜索空间或查询上限。

Gemma 4 默认 embedding 资产是以前保存的 MLX 8-bit 完整原始表，**没有被宣称与 Tensor G5 编译模型逐位一致**；Gemma 3 使用从其 Tensor G5 模型提取的 embedding 表。两者都只影响候选提议，最终验收与前缀推进使用所选模型的 NPU 实际计算。资产哈希写入协议。第一版不启用 HNSW。

`collision_plus` 是概率排序策略的 RMSE 验收适配基线：按完整模型 logits 排序全部原始 token，默认每批 128，接受该顺序下第一个通过阈值的 token。它不是原论文 CPA 验收或 CUDA 代码的原样复现。`guided` 保留原方法的阶段语义：例如随机 150 个即使拆成 128+22，也全部查询后才选通过阈值且距离最小的 token；同阶段多个通过会记录在 `passing_ids_in_accepting_stage`，不声称未查询空间没有其他解。

## 环境与构建

以下命令均在仓库根目录运行。可复用现有实验环境：

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
bash "$APP/native/build.sh"
```

编译脚本使用与现有运行库一致的 LiteRT revision `26895c9fbcc25c43faa8c1a98cd1fd28951602c3`。通过 `LITERT_HEADERS`、`JSON_HEADERS` 和 `ANDROID_NDK_HOME` 覆盖当前工作站路径。构建测试 APK：

```bash
(cd annoymous && env JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home \
  ./gradlew :app:assembleDebugAndroidTest)
"$PY" -m unittest discover -s "$APP/tests" -v
```

主机使用 NumPy；guided 另需现有环境中的 PyTorch 和 MLX。模型在手机现有应用数据目录中，runner 会核对本机与手机模型 SHA-256，并记录实际运行库和测试 APK 哈希。手机需连接并授权 ADB；多设备时显式传 `--serial`。默认仅在配套测试 APK 校验和不同时安装它；`--no-install` 禁止安装，但仍验证已安装 APK 与本地构建一致。

## 导入已有观测

`public_prefix.json` 是明确已知的公开前缀 token ID 数组，不能放入待恢复部分的真值。第一次目标位置必须等于公开前缀长度。比如已知前 11 个 token，恢复随后 100 个：

```bash
"$PY" "$APP/run.py" prepare \
  --snapshot kvidence/datasets/shadow_chat/encrypted_chat_100/capture/0001/all_layers.gkv.gz \
  --prefix-json /path/to/public_prefix.json \
  --start 11 --count 100 --layer 4 \
  --output "$APP/observations/sample0001"
```

`--layer` 是从 0 开始的 runtime producer 编号，`4` 表示第 5 层；默认模型的全局 producer 为 `4/9/14`，导入时按实际快照检查其是否为全局层。manifest 默认读取已验证 NPU 探针的实际编译签名；快照与 manifest 必须来自同一模型。导入过程会记录快照哈希，但快照本身不包含完整模型身份证明，采集来源仍需研究者核对。

```bash
"$PY" "$APP/run.py" run \
  --observation "$APP/observations/sample0001" \
  --method guided --output "$APP/results/sample0001_guided"
```

概率基线只需改成 `--method collision_plus` 并使用不同输出目录。可用 `--max-positions` 做部分恢复；`--max-queries` 是显式资源限制，耗尽会失败停止，默认不启用。输出目录必须是新目录。

## 受控端到端 smoke

为了验证管线，不需要重新运行原有 100 条采集。下面单独生成三步贪心 token 的小样本：

```bash
"$PY" "$APP/run.py" fixture \
  --prefix-json "$APP/examples/public_prefix.json" --greedy-tokens 3 \
  --output "$APP/fixtures/new_smoke"
"$PY" "$APP/run.py" run \
  --observation "$APP/fixtures/new_smoke/observation" \
  --method collision_plus --output "$APP/results/new_smoke_collision"
"$PY" "$APP/run.py" evaluate \
  --run "$APP/results/new_smoke_collision" \
  --truth "$APP/fixtures/new_smoke/private_eval/truth.json"
"$PY" "$APP/run.py" decode --run "$APP/results/new_smoke_collision"
```

`fixture --token-ids-json FILE` 可改用指定秘密 token 数组。fixture 属于采集/测试阶段，会写入独立的 `private_eval`；恢复命令通过读文件审计拒绝访问 `private_eval` 和源 `datasets` 目录。三步贪心样本是故意容易的集成测试，不能作为加密文本恢复成功率证据。

`decode` 默认跟随恢复结果记录的 tokenizer：Gemma 4 JSON tokenizer 使用 `tokenizers` 并保留特殊 token；Gemma 3 `.model` 使用 `sentencepiece` 原生解码（可能省略控制 token）。输出 `recovered.txt` 并记录处理方式与 tokenizer 哈希。token ID 序列是权威输出；文本解码不参与搜索或正确性验收。

## 输出与计时

- `protocol.json`：方法参数、完整词表范围、观测/模型/embedding/源码哈希、阈值和数据边界。
- `device/device.json`：手机模型、运行库、测试 APK 的哈希与 instrumentation 命令。
- `trace.jsonl`：每个位置的候选 ID、实测距离（由 distance_metric 标识）、各阶段、接受值、同批歧义、commit 一致性。
- `result.json`：恢复出的 token ID、停止原因、公开前缀初始化、在线总耗时与 decoder 调用时间。
- `device/rpc.jsonl`：逐次调用的 host、worker、decoder 时间，区分 NPU 与通信开销。
- `evaluation.json`：独立真值对照。只有评价阶段才能报告 token 正确率。

第一版的串行 ADB 文件 RPC 已由持久 socket 默认替代，旧路径通过 `--transport file` 保留。模型常驻避免重复加载，CPU 输入准备、张量读写和主机评分仍可能成为瓶颈。原先 59–65 ms 的 decoder 数字不能替代端到端耗时。手机私有任务目录在退出后保留用于审计，worker 收到 stop 后退出；socket 断开时失败退出，不重试提交。文件模式保留 15 分钟空闲退出机制。详见 [通信优化](TRANSPORT.md)。

## LSB 指标迁移

整数 KV 的 guided、`collision_plus` 适配对照和第一层 V 查表现在默认使用 LSB 空间的 RMSE。第一层预计算表仍保存原始整数，可直接复用。官方 `collision_plus_cpa` 校准和接受规则保留原始数值空间，不随此更改。

- `protocol.json`、`result.json` 的配置和实时 batch 事件标明 `distance_space` 和 `distance_metric`；Ghost Attack 页面显示对应单位。旧事件没有这一字段时仍显示 Raw KV L2。
- 旧观测保留了 native `codes`，可直接用于新指标；旧 metadata 中的 metric 文字不是新 run 的判定依据。旧实验结果不重写，也不能把旧距离改名成 LSB。
- `--distance-space original --distance-metric l2 --epsilon 4` 可复现旧原始数值 L2 度量；旧原始数值阈值与新 LSB 阈值没有统一换算关系。
- 浮点 KV 默认保留原始数值距离；对其显式要求 `--distance-space lsb` 会报错。
- 既有 image/audio 连续优化试验使用数值 KV/NMSE，不自动转换为 LSB。接入整数 LSB 目标时，媒体后端必须声明并实现 `media_distance_space="lsb"`，否则返回 unresolved_media，防止不同单位混算。
- 当前只迁移实现和定义，未重跑 NPU 恢复性能/准确率实验；既有报告仍代表旧指标。

### RMSE 输出字段

新运行的 `accepted_distance`、`commit_distance`、候选 `distances` 和 `minimum_distance` 使用配置中的距离指标。`accepted_l2` 和 `commit_l2` 仍保存真正的 L2，不能直接用于检查 RMSE 阈值。旧结果文件不会改写。第一层 V 查表只用 V 的坐标数归一化。不同模型仍需独立验证阈值。

```bash
bash kvidence/reconstruction/run_plaintext21_lsb.sh
```

脚本显式使用 `--distance-space lsb --distance-metric rmse --epsilon 0.125`。
