# Pixel NPU：共享前缀的候选并行接口探针

## 结论（2026-10-19）

**Gemma 4 E2B 的 `verify`（4 槽位）和 `prefill_128`（128 槽位）接口均已在 Pixel 10 NPU 上验证：共享前缀、彼此隔离的候选 token 可以并行计算。** 在已测样本中，全部 30 个 INT8 K/V 输出与逐个 `decode` 的结果逐字节一致。

这是通过底层 LiteRT C API 调用已编译的模型签名，重复位置编码并自定义 attention mask，实现同一位置的候选分支。它说明不能从公开 Session API 的 batch=1 限制推断底层模型无法并行计算候选；它尚未证明多个独立对话请求可以直接 batch。

当前模型的最大已编译入口为 128 槽位，这不是 Pixel NPU 的硬件上限。输入尺寸固定，更大的候选集合需要分组调用；尚未实现恢复算法中的分组接入。

## 128 候选结果（设备重新连接后完成）

使用原始词表中的 128 个不同 ID，共享同一快照的前缀，分别测试长度 32 和 600。每组比较 1,179,648 个 INT8 元素，两组合计 2,359,296 个，所有元素差异均为 0。第 5 层（runtime index 4）的整数 L2 与反量化 L2 也均为 0。

| 可见前缀长度 | 逐个 `decode` 128 个候选（ms） | 一次 `prefill_128`（ms） | 调用耗时比 | 最大 INT8 差异 |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 7591.182 | 59.052 | 128.550× | 0 LSB |
| 600 | 7719.017 | 65.422 | 117.988× | 0 LSB |

数值通过 [独立二进制复核](results_prefill128_retry/analysis.json)，对应 [设备报告](results_prefill128_retry/data/report.json)、[日志](results_prefill128_retry/data/native.log) 与 [执行及校验和记录](results_prefill128_retry/submission.json)。设备 instrumentation 测试通过，全流程约 214.6 秒，包括大量重复基准调用、输入准备、隔离检查和文件输出。

固定行 0、1、63、127、改变其他候选后，这四行的 KV 差异为 0。额外开放跨候选 attention 的反向对照产生 195 / 189 LSB 的最大差异。全部实际声明的前缀 KV 输入在测试结束后与初始数据逐字节一致。

**耗时比不是端到端恢复加速比，也不是相同输出工作量下的纯 batch 加速比。** `decode` 输出 logits 等额外结果，`prefill_128` 只输出 KV；两者编译图不同。它证明的是：对本实验需要的候选 KV，一次 `prefill_128` 能以较短的 decoder 调用时间得到与 128 次 decode 一致的输出。embedding、数据搬运、mask 准备和评分不在表中计时。同步 API `LiteRtRunCompiledModel` 的五次调用均值用于比较，不是异步提交耗时。

## 已完成的实验

- 设备：Pixel 10，Tensor G5 NPU。
- 模型：本机现有 `gemma-4-E2B-it_Google_Tensor_G5.litertlm`。
- 前缀来源：`../../datasets/shadow_chat/encrypted_chat_100/capture/0001/all_layers.gkv.gz`，从同一快照选择不同可见前缀长度；不是四条独立抽样的文本。
- 候选：`the` (1437)、`a` (236746)、`cat` (9307)、`dog` (21029)。
- 比较范围：runtime KV 编号 0–14 的 K 与 V，共 30 个输出张量。模型后续共享 KV 的层不是独立的额外 KV 输出。
- 每个前缀比较 36,864 个 INT8 元素，四个前缀共 147,456 个元素。
- decoder 以 NPU 模式编译执行；embedding、per-layer embedding 和辅助位置/mask 运算在 CPU 上准备。设备日志记录 DispatchDelegate 和 Tensor TPU 加载。

| 可见前缀长度 | 逐个计算 4 个候选（ms） | `verify` 一次计算 4 个（ms） | 调用加速比 | 最大 INT8 差异 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 240.107 | 59.627 | 4.027× | 0 LSB |
| 32 | 244.175 | 59.281 | 4.119× | 0 LSB |
| 128 | 243.295 | 60.605 | 4.014× | 0 LSB |
| 600 | 242.882 | 62.656 | 3.876× | 0 LSB |

每条执行路径预热一次、测量五次。serial 是四个候选各自五次调用均值之和，batch 是五次调用均值。**仅计同步 `LiteRtRunCompiledModel` 调用**，不含模型加载、embedding、mask/RoPE 准备、KV 拷贝和读回、评分、文件导出或 ADB。此表不能当作端到端恢复攻击的加速比，也未做热稳态或随机执行顺序基准测试。

## 怎样避免候选互相影响

1. serial 和 batch 使用相同的已采集前缀 KV。
2. 候选都放在同一待恢复位置 t，使用同一个 RoPE 位置，而不是 t、t+1、t+2、t+3。
3. 每一行的 attention mask 保留 serial 路径相同的前缀可见性（含滑动窗口规则），只额外开放该行自己的候选槽位，屏蔽其他候选槽位。
4. 从 batch 的输出提取每个候选的新 K/V，与单候选 `decode` 对齐比较。

隔离检查逐一固定槽位 0、1、2、3，并改变其余候选，固定槽位的全部 KV 差异均为 0。反向对照故意开放前序候选槽位，最后一个候选的深层 KV 出现变化：四个前缀下最大差异分别为 153、46、98、51 LSB。这支持“自定义 mask 实际生效”的判断。反向对照仍重复同一 RoPE 位置，不是普通连续文本生成。

`analyze.py` 从导出的二进制重新计算差异，并检查四个候选自身的 KV 不完全相同。第 5 层（runtime index 4）的整数 L2 和按模型 scale 反量化后的 L2 均为 0。

这验证了所测样本的数值一致性和隔离性；没有证明所有 token、所有前缀都一致，也没有测试恢复算法的找全能力。

## 文件与结果

- [分析结果](results_verify4_apk/analysis.json)：独立 Python 重算的汇总和逐层指标。
- [原始报告](results_verify4_apk/data/report.json)：设备计时、张量布局和差异。
- [设备原生日志](results_verify4_apk/data/native.log)、[JUnit 日志](results_verify4_apk/device.log)。
- `results_verify4_apk/data/*.bin`：serial 与 verify 的原始 INT8 KV 输出。
- `results_verify4_apk/input.json`、`submission.json`：输入、执行命令和当次测试 APK 校验和。
- `prepare.py`：读取模型签名/量化参数，解包已有 KV 快照，准备候选。
- `probe.cc`：直接调用已有运行库的独立 C++ 探针。
- `run.py`：安装配套测试 APK，启动指定 instrumentation test，取回结果。
- `analyze.py`：独立核对原始输出。

首次成功后，探针又加入了对输入 KV 未被修改的显式检查，并泛化到 128 槽位。首次 4 候选报告中的 `input_kv_unchanged` 为未知；后续 128 候选运行及 [最终 4 候选复核](results_verify4_final/analysis.json) 均通过这一显式检查。保留首次报告原样，不把新检查结果追溯填入旧报告。

最终 4 候选复核在前缀 0、32、128、600 下仍全部为 0 LSB；decoder 调用加速比分别为 3.669×、3.904×、3.912×、4.140×。这一复核与 128 候选实验使用同一个测试 APK，包含输入 KV 完整性检查。复核的全流程耗时约 210.3 秒，表中仅含 decoder 时间，不能把昂贵的探针检查/拷贝和文件操作当作生产恢复路径吞吐。

本实验新增 `CandidateBatchProbeDeviceTest` 和测试 APK 专用 `libcandidate_probe.so`；不替换主 APK 的运行库或模型，不修改原有 100 条采集结果及恢复算法。

## 复现

在仓库根目录运行。输入目录、结果目录必须是新目录，防止覆盖旧结果。

```bash
python3 kvidence/experiments/npu_candidate_batch/prepare.py \
  --candidate-count 4 --output /tmp/ghost-probe-input4

bash kvidence/experiments/npu_candidate_batch/build.sh
```

编译需与现有运行库匹配的 LiteRT headers：revision `26895c9fbcc25c43faa8c1a98cd1fd28951602c3`，以及 nlohmann JSON headers、Android NDK。`build.sh` 支持 `LITERT_HEADERS`、`JSON_HEADERS`、`ANDROID_NDK_HOME` 环境变量；默认路径针对当前工作站。LiteRT headers 中的 `build_common/build_config.h` 使用 `build_common/config/build_config_gpu_npu.h`。

在 `annoymous` 目录构建测试 APK：

```bash
env JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home \
  ./gradlew :app:assembleDebugAndroidTest
```

返回仓库根目录，连接并授权 Pixel 的 USB 调试后：

```bash
python3 kvidence/experiments/npu_candidate_batch/run.py \
  --input-dir /tmp/ghost-probe-input4 \
  --output kvidence/experiments/npu_candidate_batch/results_verify4_repeat
python3 kvidence/experiments/npu_candidate_batch/analyze.py \
  kvidence/experiments/npu_candidate_batch/results_verify4_repeat
```

128 候选准备：`prepare.py --candidate-count 128 --output /tmp/ghost-probe-input128`，再用相同 runner 指向新目录。前四个固定候选之外，按种子 20261019 从原始 tokenizer model vocabulary 抽取 124 个不同 ID，不筛语言；测试前缀长度为 32、600。128 槽位的额外邻居隔离检查只选行 0、1、63、127；serial/batch 一致性仍比较全部候选。

## 128 接口的输入差异

实际签名 `prefill_128` 不接收 `kv_cache_k_14` / `kv_cache_v_14`，但仍输出该层新 KV。探针只填充模型声明的输入；本次包括该层在内的所有输出均与 serial 一致。该签名不输出 logits，不能把全部时间收益归因于候选数量。

失败记录保留在 `results_verify4`（Android 禁止直接执行 shell 探针）、`results_prefill128`（旧输入映射错误）、`results_prefill128_fixed`（ADB 断连，未执行推理）。这些目录不属于有效实验结果。
