# NPU 通信优化

本文记录持久连接这一阶段。后续手机端复制/上下文复用优化及当前 APK 对应校准见 [NATIVE_OPTIMIZATION.md](NATIVE_OPTIMIZATION.md)；下文 socket pilot 是历史 APK 的资产。

本次只改变主机与设备之间的通信、概率排序所在位置和 CPA 开批指令的发送方式。完整词表、种子配额、随机种子、ridge、物理 batch、前缀维护与双方各自的验收规则不变。手机 worker 仍不接收目标 KV 或私有评估标签。

## 数据流

- 初始化通过 ADB 写入一次 `job.json`，启动 instrumentation；读取 `ready.json` 后建立 `adb forward tcp:0 localabstract:<随机任务名>`。
- 后续命令走一个持久 TCP/Android Unix socket 连接。每条命令包含序号和私有随机认证值；不再写 `command.json`、轮询 `response.json` 或读取 `payload.bin`。
- 帧格式为 4 字节网络字节序 JSON 长度 + JSON。响应 JSON 指定随后二进制 payload 的字节数。Python 校验序号、长度及完整收包；native 对短读/短写循环处理。
- 连接中断或回复不完整时立即终止当前恢复，不重试可能已经执行的提交。退出关闭连接并删除 ADB 转发。启动配置、日志读取仍使用 ADB。
- `--transport file` 显式保留旧路径，不在 socket 错误后自动切换，以免重复执行状态变更。

每次 `query` 只回传**所选层、当前待恢复位置、每个候选的 K/V**。例如 Gemma 4 第 5 层是 512 字节 K + 512 字节 V；128 个有效候选共 128 KiB。整个历史 KV 仍在手机内维护。CPA 为保留被选候选状态，会在手机内部读取并保存候选的所有层切片，但不会把这些层全部传回电脑。

`ranking` 在手机对 float32 logits 做稳定降序排序，分数相同按原始 token ID 升序，与原主机 `np.argsort(-logits, kind='stable')` 保持相同定义。仍返回完整词表的 int32 ID 排名，不截断候选。当前 262144 个 ID 仍占 1 MiB，因此这项改变**没有缩小原来 logits 的传输字节数**；减少流量需要进一步设计按需排名接口。

CPA 的 `begin_candidates()` 在 socket 模式延迟到紧接着的首个物理 `query`，通过同一命令的标志先清空旧候选、开启保留，再执行查询。后续物理批不清空缓存。`accept_query` 仍使用已保留的完整候选 KV；没有改变基线的接收或提交语义。

## 使用

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
bash "$APP/native/build.sh"
# 在 annoymous 目录运行 ./gradlew :app:assembleDebugAndroidTest

"$PY" "$APP/run.py" run --method guided --transport socket \
  --observation "$APP/experiments/stored_short0001_layer5/input/observation" \
  --batch-size 256 --output /path/to/new_result

"$PY" "$APP/experiments/transport_validation.py" --output /path/to/new_validation
```

不传 `--transport` 时内置后端默认为 socket；外部插件保持自己的默认行为。`compare --transport socket` 将相同选项传给两个方法。设备记录包含 transport、RPC 版本及排序位置，成对结果检查会核对这些字段。

重新构建改变了测试 APK 哈希。旧 CPA 校准资产继续保留，但恢复时会被运行时身份检查拒绝；必须在新 APK 下重新采集独立校准，不能直接改写旧资产哈希或放松身份检查。

## 已完成的设备一致性检查

结果：[transport_validation_20261019/validation.json](results/transport_validation_20261019/validation.json)。同一新 APK，先 file 后 socket，使用公开前缀及固定随机候选；不读取目标标签。

- 完整 262144 排名与主机稳定排序相同，并在提交前后跨 transport 比较相同。
- 多次查询返回相同 KV；跨 transport 所有测试候选 KV 逐字节相同。
- 145 个候选拆成 128 + 17；选择第二批候选提交，返回 KV 与其原查询切片逐字节相同。
- 提交后的下一位置候选 KV 也跨 transport 逐字节相同。
- 上述操作的 host RPC 总计：file 14.804 秒、socket 7.388 秒；worker 内部总计 7.695 秒、7.189 秒。host 减 worker 的差额由 7.109 秒降到 0.199 秒。此差额包括传输、进程调度、协议处理等，不是纯网络传输计时。

这是固定顺序的一次短检查，不能直接推广为长任务加速倍数。完整既有 109-token 样本重跑为 109/109 正确，查询序列和候选 KV 与优化前一致，在线时间由 610.764 秒降到 222.327 秒（本次约 2.75 倍）。详见 [完整重跑报告](results/stored_short0001_layer5_guided_socket/REPORT.md)。

新 APK 下已重新采集独立已知文本的 8 位置 × 1024 候选校准，保存为 [cpa_gemma4_pilot_socket](assets/cpa_gemma4_pilot_socket/calibration.json)，双阈值与旧 pilot 相同。当前使用此资产仍必须显式 `--allow-pilot-calibration`；它不是全文、全词表校准。

`decoder_seconds` 仍仅计时主 decoder/prefill 的运行调用，包含驱动/运行时开销；不包含 CPU embedder、PLE、mask/RoPE、张量复制和主机搜索。socket 优化不消除这些手机内部开销，也不代表 NPU 算子本身加速。
