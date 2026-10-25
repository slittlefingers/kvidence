# JSON prompts → Pixel NPU → 每层 KV

通过 shell 调用 Ghost（源码目录 `annoymous`）的独立 Android instrumentation 入口。每条样本新建 Session；同一批次只加载一次模型。按本次指定流程，**生成回复完成后**导出当时全部可访问的 KV 缓冲区。

## 一条命令

在工作区根目录执行：

```bash
./kvidence/experiments/npu_batch_capture/capture-short.sh \
  --input kvidence/experiments/npu_batch_capture/short100.json \
  --output kvidence/datasets/shadow_chat/encrypted_chat_100/capture
```

`short100.json` 是现有 `pilot_100_short/inputs/encrypted.jsonl` 的 100 条密文 user prompt，内容逐条保留。也可以直接把原 JSONL 作为 `--input`。默认模型为手机上已有的 Gemma 4 E2B Tensor G5 NPU 文件。生成上限沿用 APK 的 **512 output tokens**，上下文上限 4096；没有修改输入文本或添加额外研究指令。

这里的“回复后”指原生生成调用返回之后；生成可能因 EOS 或输出上限结束。本入口没有单独导出停止原因，不能据此声称每条回复均自然结束。

只运行前两条验证：

```bash
./kvidence/experiments/npu_batch_capture/capture-short.sh \
  --limit 2 --output /private/tmp/ghost-kv-smoke
```

输出目录必须是新目录，程序拒绝覆盖。多个设备时加 `--serial DEVICE_SERIAL`；找不到 adb 时加 `--adb /path/to/adb`。

## 输入格式

JSON 数组，或 `{"samples": [...]}`，或 `.jsonl` 每行一个对象。单个 `{"user_prompt": "..."}` 对象也可以。

```json
[
  {"id": "0001", "user_prompt": "This is an encrypted message.\n\n...\n\nHow do I save it?"},
  {"id": "0002", "user_prompt": "Another user prompt"}
]
```

`id` 可以省略，默认生成四位顺序号；提供时须唯一，允许英文字母、数字、下划线和短横线。prompt 不会 strip、截断或二次转义，拒绝空字符串和 NUL。运行时通过 APK 现有逻辑套该模型的 user/model 模板；输入文件本身应是原始 user prompt。

## 输出

```text
kvidence/datasets/shadow_chat/encrypted_chat_100/capture/
  input.json                 本批规范化输入与设置
  submission.json            主机提交信息
  job.json                   手机、模型/运行库 SHA-256、设置
  status.json                最新状态
  progress.json              已下载并验证的样本
  summary.json               整批成功后生成
  instrumentation.log        Android runner 输出
  0001/
    response.txt             实际生成的回复，UTF-8
    all_layers.gkv.gz        所有导出 KV 缓冲区，原字节无损 gzip
    layout.json              每个 buffer 的层号、名称、dtype、shape、stride、偏移
    sample.json              prompt/回复哈希、推理/导出耗时、数据量
    validation.json          下载哈希、gzip CRC、布局及长度校验
  0002/
    ...
```

`all_layers.gkv.gz` 解压后就是原有 `GHOSTKV1` 格式：8-byte magic、uint64 little-endian manifest 长度、JSON manifest、各缓冲区的原始 allocation bytes。保留 padding 和未使用容量；不在导出阶段反量化、重新计算 KV 或从文本重建 KV。CPU read-lock/mapping 会同步读取 NPU TensorBuffer，这不是任意物理 DRAM 读取。

“全部层”指运行时暴露的全部 KV 缓冲区。KV 共享层不一定各有独立 allocation，不能用 buffer 数量直接当作模型层数。原始 layer ID 以 `layout.json` 为准，不擅自改成论文中的 1-based 层号。

本次 Pixel 10 / Gemma 4 E2B 实测导出 **30 个 INT8 缓冲区，即 runtime ID 0–14 的 K 和 V**。本地对应 Gemma 4 E2B 配置有 35 个 Transformer 层、20 个 KV 共享层；共享使用不意味着额外存在一份独立 KV allocation。`local`、`ringbuffer` 等字段来自现有 inspector 的运行时布局判断，不能代替模型配置中的 attention 类型判定。

导出是**回复结束时的状态**，不是每个 token 的连续时间快照：

- 滑动窗口可能已经覆盖较早 prompt 的 KV；不能据此假设每层都保留整段输入和回复。
- 最后一个 pending token 可能还没有被模型消费，因此可能没有对应 KV。
- 保留物理槽位和布局；逻辑位置、active 标记沿用当前 inspector，后续恢复前需结合具体 ring-buffer 布局核验。
- 原始整数 KV 的 scale/zero-point 若不在 manifest 中，需要从匹配的模型元数据补齐；脚本不虚构这些值。

## 按层提取

默认保留压缩文件节省空间。需要单独处理某层时：

```bash
python3 kvidence/experiments/npu_batch_capture/extract.py \
  kvidence/datasets/shadow_chat/encrypted_chat_100/capture/0001/all_layers.gkv.gz \
  extracted_0001 --layers 4
```

这里的 `4` 是示例 **runtime layer ID**，先查看该样本的 `layout.json`。省略 `--layers` 提取全部缓冲区。`files.json` 把每个 `.bin` 映射到名称、shape、stride 和 dtype。文件包含原 allocation 布局，不能忽略 stride/padding 直接 reshape。

例如本次 runtime ID 4 的 K shape 为 `[1, 1, 4096, 512]`，V shape 为 `[1, 1, 512, 4096]`：两者的序列轴不同。这里的 4096 是容量，不是这条 prompt 的实际 token 数。两条真机预检查样本均通过完整下载校验；指定层提取也已用真实导出文件验证。

## 设备入口与准备

设备需连接并授权 USB 调试，安装现有可调试 Ghost APK 和模型。脚本自动安装配套测试 APK，然后调用：

```text
adb shell am instrument -w -r \
  -e class com.example.annoymous.BatchKvCaptureDeviceTest \
  -e batch_job <job-id> \
  com.example.annoymous.test/androidx.test.runner.AndroidJUnitRunner
```

这使用 APK 的原生 NPU 与导出功能，但入口在**配套测试 APK**，不是普通聊天 Activity。无需修改聊天界面，也不需要开放网络服务或 root。instrumentation 会启动/替换目标应用进程，因此运行前应结束当前交互聊天；当前聊天 Session 不会保留。

编译入口：

```bash
cd annoymous
JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-17.jdk/Contents/Home \
  ./gradlew :app:assembleDebugAndroidTest
```

测试 APK 为 `annoymous/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`。已有匹配测试 APK 时可加 `--no-install-test`。`--prepare-only` 只解析输入并生成任务 JSON，不连接手机。

## 大数据与失败处理

原生导出直接写入 pipe，Java 流式 gzip 到文件，不把整份 KV 转成 Java 字节数组/JSON/CSV。每次手机只保留一条待取回 KV；主机下载、校验并保存结果后发送确认，手机才删除这一条，继续下一条。所有样本完成后，清理手机上的本批输入和元数据。

失败时保留已完成的主机数据和当前任务目录，`summary.json` 不会伪装成成功。取消在 native 调用结束后的检查点生效，不保证立即打断正在执行的 NPU 调用。主机默认整批超时 12 小时；手机等待单条下载确认的上限 10 分钟。第一版没有自动恢复未完成任务的功能，需使用新输出目录重跑所需输入。

普通聊天不新增历史记录；本功能产生的 prompt、回复和 KV 文件属于用户显式启动的研究采集产物。

## 2026-10-19 真机验证

已完成 Pixel 10 / Gemma 4 E2B / NPU 的 **100/100** 条 short prompt 采集，使用独立 Session，保持原有单请求执行方式，未启用并行请求或改变生成参数。

| 项目 | 实测结果 |
|---|---:|
| 整批耗时（含加载、推理、导出、传输、校验） | 2,757.15 秒，约 45.95 分钟 |
| 100 条推理耗时之和 | 2,529.29 秒 |
| 平均每条推理耗时 | 25.293 秒 |
| 平均每条导出压缩耗时（含设备端压缩文件哈希） | 0.432 秒 |
| 压缩 KV 总大小 | 399,327,435 bytes，约 399.3 MB |
| 解压后 GHOSTKV1 总大小（含 manifest） | 3,775,754,700 bytes |
| 每条缓冲区 | 30 个，均为 INT8，runtime layer ID 0–14 的 K/V |
| 主机完整性校验 | 100/100 通过 |
| Android instrumentation | `OK (1 test)`，正常结束 |

结果见 [kvidence/datasets/shadow_chat/encrypted_chat_100/capture/summary.json](../../datasets/shadow_chat/encrypted_chat_100/capture/summary.json)，模型、运行库及设备信息见 [kvidence/datasets/shadow_chat/encrypted_chat_100/capture/job.json](../../datasets/shadow_chat/encrypted_chat_100/capture/job.json)。各样本数据位于 `kvidence/datasets/shadow_chat/encrypted_chat_100/capture/0001` 至 `kvidence/datasets/shadow_chat/encrypted_chat_100/capture/0100`。本批手机端临时任务目录已在主机全部校验通过后清理。

本次验证的是批量推理、真实缓冲区采集与传输完整性，不是 token 恢复准确率测试，也没有测量 NPU 利用率或 batch 加速比。已存在 `kvidence/datasets/shadow_chat/encrypted_chat_100/capture`；再次运行时请换一个新的 `--output` 目录。

## 多模态批量输入

现支持每条一个图片或音频附件，并保持 `media → original user_prompt` 顺序。两套正式数据入口为：

- [image_chat_100](../../datasets/shadow_chat/image_chat_100/README.md)
- [audio_chat_100](../../datasets/shadow_chat/audio_chat_100/README.md)

每条 JSONL 的字段：`id`、`user_prompt`、`content_order: ["media", "text"]`、`media: {"type": "image" 或 "audio", "path": "相对于输入文件的媒体路径", "sha256": "文件的完整 SHA-256"}`。每条只能有一个附件；媒体路径须位于输入文件所在数据目录内，文件非空且不超过 20 MiB。

主机与设备均核验媒体哈希。主机生成设备任务内的媒体路径并上传文件；设备调用已有 `NativeModel.sendMedia`，在回复后导出原始 KV，同时记录媒体身份与顺序。完整批次验证通过后清理设备任务目录。原有纯文本 JSON 格式保持兼容；多模态任务只接受 Gemma 4 E2B。

当前应用配置为语言解码器 NPU、视觉/音频编码器 CPU。此批次没有改变 native 推理库或 token 恢复方法。多模态输入的展开位置不同于纯文本，不应直接复用文本集的目标位置偏移。

先构建并安装更新后的 `:app:assembleDebugAndroidTest` 测试 APK；仅在已安装当前版本后使用 `--no-install-test`。单样本验证保存在 `multimodal_smoke/`，正式 100 条数据保存到各自数据集的 `capture/`。
