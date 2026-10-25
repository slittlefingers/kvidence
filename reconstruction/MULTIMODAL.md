# 按恢复出的媒体 token 分派

保留现有文本恢复算法。深层恢复确认媒体开始 token 后，才加载对应的 Python 处理模块；成功重放恢复出的媒体表示后，回到同一个文本恢复循环。

## 当前完成范围

- Gemma 4 E2B 默认规则，按所选 tokenizer 的特殊 token 字符串解析原始 ID，不硬编码数字 ID。
- `image.py` 与 `audio.py` 的独立入口，可通过模型配置替换为可信本地 `module:function`。
- 已知媒体区间输入、按 token 分派、连续特征优化接口、真实后端复核与提交检查、返回文本循环。
- 原有搜索方法、候选配额及文本验收规则保持不变；原始实验脚本保持不变。
- 合成后端测试覆盖两种媒体、多个媒体块、后续文字、区间缺失、错误提交、位置预算及位置对齐评价。
- 图片与音频的实际 Gemma/MLX 反演算法已从旧 pilot 迁入：独立的 `build_problem`、共享可微第 5 层映射、原生 MLX Adam 更新、完整模型重放；两个模块均可独立从旧公开 KV 观测运行。

迁移验证已完成：70 项常规测试、2 项真实模型核对，以及图片/音频各两个起点的 3000 步优化。数值、输入范围与验证限制见 [迁移验证报告](results/media_algorithm_port_20261020/REPORT.md)。

**当前内置 Pixel NPU 后端仍是文本后端，没有实现 `media_problem`、`query_media`、`commit_media`。因此，当前实际 NPU 运行遇到媒体开始 token 会以 `unresolved_media` 停止，原因写入 trace。不能把本次框架适配称为真实 NPU 多模态恢复已验证。** 旧 MLX 实验仍可独立运行；也没有自动把旧 BF16 forward 当作当前 INT8 NPU 的 forward。

## 图片与音频的具体算法

两者共用同一个优化过程，区别在于所用 adapter、媒体 token 类型、输入维度和由模型生成的 attention mask。

| 项目 | 图片 | 音频 |
| --- | --- | --- |
| 模块 | `media/image.py` | `media/audio.py` |
| 固定的 Gemma adapter | `model.embed_vision` | `model.embed_audio` |
| E2B 待优化变量宽度 | 768 | 1536 |
| 宽度取自 | `vision_config.hidden_size` | `audio_config.output_proj_dims`，缺省时才取 encoder hidden size |
| E2B adapter 输出宽度 | 1536 | 1536 |
| 媒体 token type | 1 | 3 |

音频 encoder 的内部 hidden size 是 1024，但本实验优化的是其输出投影之后、adapter 之前的 1536 维表示，不能误用 1024。维度在代码中从所选模型配置取得。

令整个媒体区间有 N 个位置，U 是 N×d 的未知矩阵，固定 adapter 为 A，固定的前 5 层 KV 映射为 F，观测为 Y：

$$
Z=A(U),\qquad
\mathcal{L}(U)=\frac{\operatorname{mean}((F(A(U))-Y)^2)}{\operatorname{mean}(Y^2)}.
$$

**优化的是整块 U，不是逐个位置独立恢复 KV，也不更新 adapter 或模型权重。** 一张简单图片的 64 个位置对应 64×768 的变量矩阵；音频同理为 N×1536。每轮整块前向、整块 KV 损失、整块反向，各元素的梯度可以不同。媒体位置的依赖由实际模型 mask 决定；整块优化不意味着所有位置都能看见彼此。后续文本恢复仍是逐 token 搜索。

迁入的算法来自 `experiments/gemma4_multimodal_layer5/experiment.py::invert` 与 `runtime.py::Layer5Map`，原文件未修改：

1. 用独立 NumPy RNG 初始化标准正态 U；默认 seed 17，独立运行入口默认试 17、29 两个起点。
2. 将 U 转成 BF16，通过固定 adapter，再经过完整的前 4 个 block 和第 5 层 K/V 投影、归一化与 K 的 RoPE。
3. 对 U 求梯度，使用 Adam（β₁=0.9，β₂=0.999，ε=1e−8）。
4. 默认 3000 步，初始学习率 0.08，余弦下降到初始值的 15%；每次更新后沿各位置的特征维做 RMS 归一化。这是保留旧实验的优化选择，不是逐位恢复或唯一性保证。
5. 以目标 KV NMSE 最小的迭代选择 U。多起点也仅按 KV NMSE 选择，不读取原图、原音频、真实表示或转录。
6. 用恢复的 Z 重放完整 MLX 模型，核对第 5 层 KV 与截断 forward 一致。框架实际提交时仍须由当前后端通过原有验收阈值。

当前迁移限定第 5 层全局 producer。输入结构必须截止到媒体末尾；未知后续文本不进入 forward。此前已恢复的媒体可通过 `prefix_media` 提供替代表示；缺少时拒绝把媒体占位 token 当作词表 embedding 重放。没有假定图片和音频采用同一个 attention mask。

## 独立运行图片或音频模块

下面命令使用旧实验的公开观测，输出必须为新目录。只读取模型、结构和目标 KV；原始媒体与 `private_eval` 读取受保护。

```bash
PYTHONPATH=kvidence/reconstruction \
kvidence/experiments/gemma4_multimodal_layer5/.venv/bin/python \
  -m kvrecon.media.image \
  --observation kvidence/experiments/gemma4_multimodal_layer5/results_long/observed/image01 \
  --output /tmp/new_image_inversion

PYTHONPATH=kvidence/reconstruction \
kvidence/experiments/gemma4_multimodal_layer5/.venv/bin/python \
  -m kvrecon.media.audio \
  --observation kvidence/experiments/gemma4_multimodal_layer5/results_long/observed/audio01 \
  --output /tmp/new_audio_inversion
```

可设置 `--model`（本地 MLX 模型目录）、`--steps`、`--lr`、`--seeds`、`--epsilon`。默认输入格式是旧 MLX BF16 第 5 层观测，不接受把原始 INT8 codes 冒充原始数值。根目录 `features.npy` 为按 KV NMSE 选择的表示；每个 seed 目录保存初始表示、优化后的 U/Z、轨迹、逐位置 L2 和完整重放检查。输出状态 `fitted` 表示拟合完成，不代表媒体语义正确、原文件恢复、后续文字恢复或 NPU 验证。

## 分派流程

```text
现有文本候选搜索 → 实测 KV 通过 → 提交并复核 token
    ├─ 普通 token：继续文本搜索
    ├─ 图片开始 token：media/image.py
    └─ 音频开始 token：media/audio.py
                            ↓
               用当前已恢复前缀构造媒体优化问题
                            ↓
               优化未知表示，通过已知 adapter 投影
                            ↓
               实际后端 query_media，逐位置复核 KV
                            ↓
               commit_media，检查位置和 query 一致
                            ↓
               恢复并检查结束 token → 继续文本搜索
```

“通过”沿用所选方法的验收规则：guided / collision_plus 为逐位置拼接 K+V 的原始数值 L2 < epsilon（默认 4）；CPA 使用其原有独立 K/V 校准阈值。媒体优化损失只用于找表示，不能替代实际后端验收；没有使用平均 NMSE 作为最终接受标准。通过阈值也不是正确性或唯一性证明。

## Gemma 默认表

| 类型 | 开始标记 | 连续媒体占位标记 | 结束标记 | 处理模块 |
| --- | --- | --- | --- | --- |
| 图片 | `<\|image>` | `<\|image\|>` | `<image\|>` | `kvrecon.media.image:recover` |
| 音频 | `<\|audio>` | `<\|audio\|>` | `<audio\|>` | `kvrecon.media.audio:recover` |

具体字符串见 [模型配置](configs/gemma4_e2b.json)。这里只依据本地 tokenizer 验证特殊 token 的存在；真实执行图是否以这些离散标记包围媒体、采用什么 mask，仍需适配器核对实际输入展开。不能从 tokenizer 有这些 token 推断任意媒体输入都能用普通文本查询恢复其边界。Gemma 3 1B 没有启用媒体规则。

开始 token 才触发分派；不能因为概率候选列表里出现媒体 token 就调用模块。媒体 soft token 是连续特征位置的占位符，不能用该 ID 的普通词表 embedding 代替实际媒体特征。如果文本搜索接受了 soft token，框架停止并记录问题。

## 媒体边界

第一版与旧实验一样，**区间位置和长度是已知条件**。开始 token 不包含长度，边界表也不是可以在 KV 字节中直接搜索的字符串。未知区间长度不猜测，停止为 `media_boundary_unknown`。

通过 `prepare --media-spans-json spans.json` 导入结构。例如以下为示意位置，不能直接套用到已有数据集：

```json
[
  {
    "modality": "image",
    "marker_position": 4,
    "start": 5,
    "end": 271,
    "source": "Known expanded input layout from acquisition instrumentation"
  }
]
```

- 绝对位置从 0 开始；媒体特征覆盖 `[start, end)`，`end` 是随后结束标记的位置。
- `marker_position` 必须包含在恢复目标中，且 `start = marker_position + 1`。
- 输入保留该层的开始标记、全部媒体位置、结束标记和后续待恢复文字；不把媒体区间压缩成一个 token。
- 区间必须按位置排列、不重叠。多块媒体可按顺序出现。
- 只允许结构字段，不允许传入原始图片、音频、真实特征、真实后续 token。恢复阶段仍受现有 private/dataset 读取保护。
- 若公开 token 前缀已经包含媒体开始/占位 token，当前框架拒绝纯 token 重放，要求将开始标记包含在恢复区间。
- 不支持未知长度推断、缺失位置或复杂非连续切块结构；这些需要后续适配器扩展，不能用错误偏移继续执行。
- 第一层 V 字典仍只适用于文本位置，不替代媒体连续表示反演。

## 后端需要实现的接口

保持已有 `reset/probabilities/query/commit/values` 文本接口。媒体接口为可选能力，只有分派时才要求存在：

```python
problem = backend.media_problem(modality, positions, targets)
codes = backend.query_media(features, modality)
codes = backend.commit_media(features, modality)
```

Gemma 后端也可以只提供模型上下文，直接使用本次迁入的算法，无需重写优化：

```python
context = backend.media_context(modality, positions)
# 必需键：model、input_ids、token_types
# 可选键：steps、lr、seed、layer_index、prefix_media
# input_ids 只含自行恢复/公开前缀和媒体占位符，截止媒体末尾。
# image.py/audio.py 自动调用 build_problem(..., targets=观测媒体KV, **context)。
```

若存在 `media_context` 则优先使用对应模块的 Gemma `build_problem`；否则保留通用 `media_problem` 插件接口。两者都还需要真正的 `query_media` / `commit_media`。MLX 算法会以 `fit_native` 保留原有 MLX 更新算术；没有将每一步转换成 NumPy 更新。

`media_problem` 只能利用当前自行重建的前缀、目标层媒体 KV、公开模型参数和结构。返回对象提供：

- `initial`：待优化变量的初始浮点数组，不能是真实媒体特征。
- `loss_and_grad(u)`：返回有限的标量损失及同形状梯度。
- `project(u)`：通过已知 adapter 得到可送入执行器的媒体表示。
- 通用 NumPy fallback 的可选 `steps`、`lr` 默认 3000、0.05；`normalize_latent` 默认 false。Gemma 专用算法使用上面旧实验的 0.08 与 RMS 归一化。

公共 solver 为其他后端保留 NumPy Adam fallback；本次迁入的 Gemma 路径使用与旧 pilot 一致的 MLX Adam 更新和超参数。后端也可以通过替换 Python handler 使用其他算法。

`query_media` 必须从当前恢复前缀执行候选媒体，不改变任何持久前缀、位置或 logits。`commit_media` 只提交一次，更新后续文本查询所需的全部状态；返回与 query 相同的 KV codes。二者输出形状为 `[media_positions, K_plus_V_width]`，`values` 转回原始数值。还需正确实现模型的媒体 attention mask、位置、PLE 与边界嵌入语义。**不能把目标层观测 KV 直接灌回前缀冒充恢复。**

优化可用可微代理，但最终必须用实际推理后端复核。INT8 NPU 没有在这里被假定支持反向传播。代理与 NPU 的差异若导致不通过，返回未解决，不能继续后面的文本。

## 输出和评价

- 文本仍写入 `trace.jsonl`、`recovered_token_ids`，另记录 `recovered_token_positions`，避免媒体跳转后错位。
- 媒体记录在 `media/<marker_position>/`：恢复特征、优化过程、逐位置实测 L2 与提交结果；`result.json` 中的 `media_segments` 列出成功处理的媒体区间。
- `attempted_positions` 计文本搜索的位置数；`covered_positions` 计成功提交的文本及媒体位置。媒体迭代成本另看媒体记录；不能把文本候选查询数当成全部媒体优化成本。
- 多模态文本评价的真值文件必须同时包含 `token_ids` 和对应绝对 `positions`。缺少位置就拒绝评价，不能把媒体后的文字与原 token 数组错误地 zip。
- 恢复媒体表示的文件不是原图片/音频文件；文本解码输出也不代表完整媒体内容。

测试命令：

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  -m unittest discover -s kvidence/reconstruction/tests -q

# 另需本地模型和 Metal：核对新旧 forward、梯度、3 步更新及全模型重放。
KVRECON_TEST_MLX_MEDIA=1 \
kvidence/experiments/gemma4_multimodal_layer5/.venv/bin/python \
  -m unittest discover -s kvidence/reconstruction/tests -p test_gemma_media.py -v
```
