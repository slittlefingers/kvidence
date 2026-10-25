# Ghost Attack 实时实验页面

本页监控 KV 重建实验，不改变搜索算法。普通 chat 不新增历史记录。APK 的侧栏新增 **Attack**；一旦存在实验快照，从 Ghost 图标打开应用会先显示 Attack，可通过 **Chat** 返回聊天。重建进行中 Chat 按钮禁用，避免额外加载模型。

## 展示内容

- 当前方法、已接受位置/计划位置、累计候选、耗时、同步状态。
- 每一批真实评估的候选：token ID、KV L2（新整数恢复为 LSB steps，旧记录与 CPA 为 Raw KV L2）；点击方块查看文字、批内次序、累计查询次序和通过状态。CPA 另外显示 K/V 距离。
- 绿色表示通过该方法的验收阈值，粉色表示最终选中的候选。guided 会完成一个候选阶段后再选取最优者，第一批出现通过者不代表马上提交。
- 点击 **Current position / Position** 标题或下方已恢复位置，可以前后翻阅该位置所有已记录批次。每批候选分页展示，详情与批次页均可返回。
- 恢复文字与总结；只有独立 `evaluate` 生成的 `evaluation.json` 才能给出 Verified token IDs / false acceptances。实时 Accepted 不是身份或唯一性证明。

## 启动

终端一，正常恢复命令加 `--live-events`（`compare` 也会传给两个方法）：

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
OUT="$APP/results/my_new_run"
"$PY" "$APP/run.py" run --method guided \
  --observation /path/to/observation --live-events --output "$OUT"
```

终端二，同步到 USB 连接的 Pixel：

```bash
"$PY" "$APP/monitor_attack.py" --run "$OUT" --serial YOUR_DEVICE_SERIAL
```

可以先开同步程序再启动重建；`--run` 也可指向 comparison 输出目录，自动跟随最新产生事件的子实验。默认每秒同步一次。画面可能跳过一秒内的多个快速批次，但完整批次已归档，可在详情中翻阅。

独立评价结束后，同步程序会显示评价结果：

```bash
"$PY" "$APP/run.py" evaluate --run "$OUT" --truth /path/to/private_eval/truth.json
```

同步程序不读取真值文件，只读取已生成的 `evaluation.json`；搜索进程仍使用原有 private-label guard。同步程序按 Ctrl-C 停止，不中断恢复。`--once` 将当前已有记录发布后退出，适用于重新查看已完成结果。

## 文件与进程

- 主机 `events.jsonl`：追加、逐批 flush；只有明确启用 `--live-events` 才创建。
- 原有 `trace.jsonl`、`result.json`、`evaluation.json` 保持原用途。
- 手机私有目录 `files/attack_live.json`：原子替换的最新快照。
- 手机私有目录 `files/attack_runs/<relay-session-id>/events.jsonl`：本次同步的完整事件档案；重新连接创建新档案，不重复追加旧事件。
- UI 运行在 `com.example.annoymous:dashboard` 独立进程，不加载聊天引擎、adapter 或 NPU 模型；恢复仍由原 instrumentation worker 运行。
- 监控进程会持有重建出的候选和文字，取证研究应将该进程与原目标进程分开识别，不能把显示器产生的数据当作目标应用残留。

当前界面展示最近一次运行，不提供多实验列表或在手机上启动/停止恢复。查看完毕后快照保留，离线仍能翻阅已同步的批次。8 秒没有同步会提示更新暂停/上次同步时间，不据此推断实验成功或失败。

## 验证范围

真实 NPU 同一条样本的 guided 与 Collision+ CPA 均已完成：分别为 21/21 与 20/21 正确，实时批次和独立评价已同步到手机。详见 [验证报告](results/attack_dashboard_validation_20261020/REPORT.md)。此次未启动 100+100 数据集的批量恢复；CPA 仍使用独立 pilot 子集校准。
