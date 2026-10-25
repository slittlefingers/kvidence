# 终端与 Ghost 采集控制

在项目根目录操作。电脑必须通过 USB 连接当前 Pixel，控制服务需保持运行。

```bash
./collectctl serve              # 在一个终端保持运行
./collectctl status             # 在另一个终端查看状态
./collectctl watch              # 每 5 秒显示进度，Ctrl+C 仅退出查看
./collectctl pause              # 请求暂停；等 status 显示 paused 再切换
./collectctl resume             # 休息结束后继续，保留已保存样本
./collectctl profiles           # 列出上下文配置
./collectctl phone              # USB 连接 Ghost 的原生采集页面
```

手机打开 Ghost → 数据采集；Attack 页面也有“采集”入口。服务重启后重新运行
`./collectctl phone`，更新手机的连接令牌。USB 断开后手机不能控制电脑；关闭手机页面不停止任务。
采集 worker 由电脑启动，本版本还不支持断开电脑后由手机独立调度。

## 更换图片前的文本上下文

每个配置固定一套上下文。修改内容时创建新配置，避免把不同前缀混进同一训练目录。
`messages.json` 为按顺序排列的完整历史和最后一条 user 文本：

```json
[
  {"role": "user", "content": "上一轮的问题"},
  {"role": "assistant", "content": "上一轮的回答"},
  {"role": "user", "content": "请看接下来的图片。"}
]
```

```bash
./collectctl create my_context --messages messages.json --limit 10000
./collectctl pause
./collectctl status             # 等待 paused
./collectctl use my_context
./collectctl resume
```

仅一条 user 消息也可。原生对话模板、BOS 与图片开始标记自动添加，文本中不填模板标记。
本版本支持最多 2000 个图片前 token；配置名仅字母、数字、下划线和连字符。
可用 `--suffix '图片后的问题'` 指定后缀；训练图片 KV 捕获仍只执行到图像末尾，后缀记录但不参与图像 KV。
手机可输入图片前文本或同样的 messages JSON，创建、选择配置，然后继续。

`fixed991` 是既有 991-token、10000 张任务；`context_short` 是可选的 17-token 示例，未启动批量采集。
新配置写入 `profiles/<name>/`，数据写入 `data/profile_<name>/`，保存文本、精确 token ID、
前缀 SHA256、模型/层信息、量化参数与源图像清单；恢复时检查协议一致性。
切换后模型缓存重新计算，不能复用上一前缀的 KV。重要目标图像保持独立，不加入训练。

## 暂停与休息

新配置在保存当前图片后退出并关闭 worker；初始化中的暂停须等待初始化结束。
原始 fixed991 使用进程中断退出，未完成图片在继续时重试；已保存样本保留。
每新增 500 张关闭 worker 并休息 300 秒；手动暂停后继续也至少休息 300 秒。
严重热状态提前休息；温度回落后才继续。控制页面的温度为最近一次采样。

已验证：终端暂停/继续与配置切换、Ghost 真机显示和连接、17-token 上下文真实 NPU
单图采集及完整数据校验（266 个图像位置，snapshot 和独立 prefix 重算均逐位一致）。
该 smoke 样本在 `data/smoke_context_short/`，不计入 fixed991 的训练进度。
