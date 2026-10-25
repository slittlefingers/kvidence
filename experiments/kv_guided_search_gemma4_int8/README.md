# Gemma 4 E2B：权重 W8 与运行时 KV8 配对实验

2026-10-16。继续 [此前实验](../REPORT.md)，在 10,000 候选、200 次初始查询和最高 600 次预算下，验证 8 位权重模型上的 KV 方向搜索，并比较原生 KV 与运行时 8 位 KV。

## 量化对象与模型来源

使用 [mlx-community/gemma-4-e2b-it-8bit](https://huggingface.co/mlx-community/gemma-4-e2b-it-8bit/tree/03dcf209f3f549b4075e7191e77cf69b3d48e1b2)，固定 revision `03dcf209f3f549b4075e7191e77cf69b3d48e1b2`。这是 Google Gemma 4 E2B IT 的 MLX 8-bit affine 转换，group_size=64；上游 Google revision 为 `70af34e20bd4b7a91f0de6b22675850c43922a03`。

两片权重共 5,866,581,158 bytes，SHA-256 均与 Hugging Face LFS 元数据匹配：

| 文件 | SHA-256 |
|---|---|
| model-00001-of-00002.safetensors | `0e82359bc2f0effdad15168d05a2ae3d46e72cb3296b5658303a2f78eee9926b` |
| model-00002-of-00002.safetensors | `797c7786d137de2b77826ae838ee366eab5a9479dfca91b345f71b7e99d71cfe` |

权重保持 MLX 量化模块，前向使用量化权重计算；没有将完整权重转为浮点模型。归一化等部分参数仍是浮点。这里只加载文本语言模型，未运行图像、音频模块。文本逻辑参数按 packed weight 还原计数为 4,628,569,379，含主 embedding 及 per-layer embedding；不是宣称 E2B 有这么多同时参与每 token 矩阵计算的参数。

“INT8”在本实验中具体指 **8-bit affine 量化**：整数码打包在 uint32 中，每组带浮点 scale、bias；不表示所有计算都是 signed INT8，也不表示激活为 INT8。

| 模式 | 权重 | 实际缓存格式 | 注意力计算 |
|---|---|---|---|
| native | W8 affine | BF16 K/V | 使用 BF16 KV |
| int8 | 相同 W8 affine | 8 位整数码 + BF16 scale/bias | 使用 packed KV 的量化注意力 |

**权重 8 位不会自动使 KV 变成 8 位。** 此模型的原生缓存实测为 BF16。KV8 组在每层生成 K/V 后立即量化保存，后续层及 token 的注意力消费这份量化缓存；不是模型跑完以后对输出加噪或取整。分析时读取实际缓存，再按保存的 scale/bias 解码为 FP32 向量用于拟合和距离计算。分析向量的 FP32 类型不改变原缓存精度。

## Gemma 4 的共享 KV 与额外 embedding

模型有 35 层，只有前 15 层生成独立 KV。后 20 层按注意力类型复用第 14 层的 sliding KV 或第 15 层的 full KV。第 35 层使用第 15 层 KV，不能把它报告成独立的“第 35 层新 KV”。

本次观察第 10、15 层的真实 full-attention KV，均为 1 个 KV head × 512 head_dim × K/V = 1,024 维；没有使用首层。

主 embedding table 为 262,144 × 1,536，额外 per-layer embedding table 为 262,144 × 8,960（35 × 256）。代理只接收主 embedding，与此前思路一致；原模型前向仍使用真实 per-layer embedding。当前未评估把额外 embedding 也作为代理输入的收益，因此与 Qwen 的差异不能单独归因于量化。

## 运行时 KV8 适配

复用 MLX-LM `QuantizedKVCache(bits=8, group_size=64)` 及量化 SDPA。适配在本目录 [runtime.py](runtime.py)，没有修改安装的 MLX-LM、LiteRT-LM 或 Android 应用源码。

Gemma 的共享层通常以 `cache=None` 接收先前层的 KV，而 MLX SDPA 根据 `cache.bits` 选择量化分支。为让共享层正确消费打包 KV，适配提供无缓冲区的 `SharedQuantizedCache` 视图：只返回生产层已有的量化 tuple，不再次追加或保存一份独立 KV。

所有独立生产层在 KV8 模式下都使用量化缓存。sliding 层暂用非环形量化缓存并保留局部注意力 mask；本实验全部前缀及目标长度小于 512，尚未触发窗口淘汰，因此适用于该短上下文条件。没有验证超过 512 时的量化环形缓存行为。

完整字节码、scale/bias 都假定可读。仅获得整数码而缺少量化元数据的恢复不在本实验范围。

## 搜索设计与配对方式

- 从原模型有效英文/数字片段 token 中选 10,000 个候选，保证包含单个大小写字母与数字；token 编解码和全部候选/前缀拼接边界均验证。
- 使用相同 8 个简短英文前缀，补 BOS，未套用对话模板；是原始文本单 token 实验，不是完整聊天交互。
- 正式 seed 为 `20261017`、`20261018`。每个 seed、每个 KV 模式、每种目标分布各 16 个目标，分别观察第 10、15 层。
- 两种 KV 模式使用相同候选池、目标 ID、前缀、初始 200 个候选索引。模型概率目标统一按 W8/native-KV 的参考概率采样，保证两组目标一致；概率基线则各自使用当前 KV 模式完整 35 层的条件 logits。
- 初始 200 次随机查询后每批 20 次，固定预算 200、220、240、300、400、600。四种方法共享初始观测，不提前停止。
- 复用 Qwen 实验的真实前向查询循环、随机/概率/打乱 embedding 对照；ridge=0.01，未离线训练反演器，未调参。
- 拟合解仍在 CPU 求解，候选距离打分改用 MLX FP32 GPU 矩阵运算，数学目标与之前相同。独立数值检查中 CPU/GPU 前 100 名排序一致；跨后端结果不保证每个近似相等距离都逐位一致。
- 最终结果按真实缓存的 KV 距离选最小候选，单独统计“目标被查询”和“最终精确恢复”，避免把量化或低精度数值误差导致的识别失败漏掉。

目标 KV 的生成、已知前缀处理、加载和下载不计入候选查询预算；每种方法最大 600 个不同候选，占候选集 6%。仍会扫描全部候选 embedding 打分，不是完全避开词表规模相关开销。

## 读取与数值核验

KV8 分析使用 MLX 默认反量化：输出精度由 BF16 scale/bias 决定，随后转为 FP32 做统计。未另做 FP32 scale 反量化对照，因此这里评估的是该读取流程下的恢复能力。

[check_oracle.py](check_oracle.py) 在 native、KV8 两种模式都比较了完整输入前向、前缀缓存查询与单独/批量查询，并检查原前缀缓存数值和 offset 未改变。详见 [oracle_validation.json](oracle_validation.json)。

BF16/量化计算随形状存在小差异，核验没有要求逐位相同。4 个独立探针中，cached/full 的归一化 MSE 约为 1.34e-4–3.71e-4，batched/individual 约为 8.51e-5–3.89e-4；所有探针的真实 KV 最近邻仍正确识别。这里的 normalized MSE 不是“错误 token 比例”。

这是运行时真实缓存 API 的实验，尚未接入物理内存 dump、地址扫描或手机 NPU。手机现有 LiteRT Gemma 4 文件使用混合位宽，不能把本次 MLX W8/KV8 结果等同于该文件的实际部署结果。

## 正式结果

两个种子、两种 KV 模式全部完成，每模式每分布每层合计 32 个目标。共 256 个“模式 × 目标 × 层”案例，均通过轨迹审计；两组的目标、候选池、前缀与初始候选逐项配对一致。pilot 不纳入统计。

以下是 KV 代理的精确恢复数：

| 分布 | 生产层 | BF16 KV @220 | KV8 @220 | BF16 KV @600 | KV8 @600 |
|---|---|---|---|---|---|
| 均匀 | 10 | 16/32 | 16/32 | 31/32 | 30/32 |
| 均匀 | 15 | 32/32 | 32/32 | 32/32 | 32/32 |
| 模型概率 | 10 | 15/32 | 16/32 | 29/32 | 30/32 |
| 模型概率 | 15 | 26/32 | 26/32 | 28/32 | 29/32 |

均匀目标第 15 层，两种 KV 模式只需在初始 200 次之后增加 20 次查询，就恢复本次全部目标；随机 @220 为 0/32，语言概率 @220 为 0/32，@600 分别为 2/32、1/32。共同初始 200 次在均匀目标组均未命中。

概率目标 @220，语言概率排序为 31/32，高于 KV 代理的 15–26/32；@600 概率排序为 32/32，代理仍有失败。不能声称代理全面优于语言先验。

配对结果不支持“KV8 一定使搜索失效”：均匀第 15 层全部目标在两组都恢复；均匀第 10 层 @600 有 1 个目标仅 BF16 成功。概率第 15 层 @600 有 1 个仅 BF16 成功、2 个仅 KV8 成功，说明量化路径也会改变排序轨迹，不能把一两个样本差异当成稳定优劣。

在所有已完成轨迹和预算点，“目标已查询但最终真实 KV 最近邻选错”的计数为 0。正式脚本对新运行也独立统计这两类指标，不以两者必然相同为假设。完整曲线与配对表见 [RESULTS.md](RESULTS.md)，机器可读数据见 [combined_metrics.json](combined_metrics.json)。

四轮串行运行时间分别约为：seed 20261017 的 native/KV8 为 86.60/89.10 秒，seed 20261018 为 87.18/87.62 秒。这些是整批测试时间，不能当作单目标恢复延迟或跨模型量化加速倍数。

## 复现与文件

```bash
cd /path/to/litertlm
EXP=kvidence/experiments/kv_guided_search_gemma4_int8
python3.13 -m venv "$EXP/.venv"
PY="$EXP/.venv/bin/python"
"$PY" -m pip install -r "$EXP/requirements.txt"
"$PY" "$EXP/download_model.py"
"$PY" "$EXP/check_oracle.py"
# 为保留报告所引结果，重跑时使用不同输出目录。
"$PY" "$EXP/search.py" --trials-per-scenario 16 --seeds 20261017 20261018 --output "$EXP/results-new"
"$PY" "$EXP/validate.py" --results "$EXP/results-new"
# 以下汇总脚本默认读取本次正式 results/。
"$PY" "$EXP/summarize_results.py"
```

原始正式结果在 `results/seed_{seed}_{native|int8}/`：各含 `metrics.json`、`candidates.json`、`trials.json`。`run.log` 保存运行日志；`model_provenance.json` 保存来源、hash 及版本；`model_metadata.json` 保存 embedding 与共享 KV 映射；`requirements.lock.txt` 为完整依赖锁定文件。

模型权重和独立虚拟环境放在忽略提交的 `model/`、`.venv/`。搜索脚本复用相邻 `kv_guided_search_large/search.py` 的公共逻辑，复现时需保留该目录。

## 解释边界

两组都使用相同 W8 权重，所以配对差异只考察这套运行时里启用 KV8 的影响，不能衡量权重 W8 相对于未量化 Gemma 4 的损失。KV8 模式还切换到量化注意力内核，因此不能进一步把差异归结为整数存储误差这一单独因素。

候选仍是 10,000 个英文/数字 token 子集，正确前缀已知、上下文短、单未知 token、量化 metadata 完整。不能据此承诺完整 262,144 行词表、未知前缀、长密码、缺失 dump 或 NPU 场景的成功率。32/32 若出现，也只是有限样本结果。

不同层观察同一目标，不能将两层作为独立样本相加。不同模型的架构、tokenizer、embedding 特征及计算后端不同，不能把与 TinyStories/Qwen 的差异当成量化的因果效果或端到端速度提升。
