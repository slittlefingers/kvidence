# 完整 768 方向与 64 方向搜索对照

材料为 image_chat_100/0001，第 5 层，第一个图片位置（position 5）。
搜索变量仍为 768 维 float32 pre-adapter 特征；使用真实手机 vision adapter 和 NPU。
完整版本每轮用 768 个正交方向，测量正负 1536 个探针，另测 22 个更新候选。
候选分批执行，每个物理批次最多 128 条，全部保持相同前缀和位置。

| Seed | 方向数 | 初始 RMSE | 最佳 RMSE | 查询数 | 更新轮数 | 秒 | 通过 |
|---|---:|---:|---:|---:|---:|---:|---|
| 17 | 64 | 65.02700 | 10.61686 | 45001 | 300 | 172.37 | False |
| 17 | 768 | 65.02700 | 7.81387 | 45001 | 29 | 121.01 | False |
| 29 | 64 | 57.96381 | 11.95418 | 45001 | 300 | 172.00 | False |
| 29 | 768 | 57.96381 | 10.12924 | 45001 | 29 | 121.50 | False |

## 本次结果

- Seed 17：同预算 RMSE 降低 26.40%。
  首次达到旧版最终误差需要 23371 次查询。
- Seed 29：同预算 RMSE 降低 15.27%。
  首次达到旧版最终误差需要 21813 次查询。

## 比较口径

两种方法使用相同观测、初始输入、初始 KV、扰动初值、学习率初值与几何求解器。
64 方向使用已完成的 300 轮实验；其日程等价于按候选预算进度推进。
768 方向通过 --max-queries 45001 同样按查询进度推进日程，而非把 300 轮日程直接套在 29 轮上。
在 45,001 次预算内，768 版本完成 28 个完整 768 方向轮次和一个 677 方向的尾轮。
64 版本完成 300 个完整轮次。因此比较的是同查询预算的恢复效果，不是同轮数的效果。
随机种子和初始输入相同，但维度不同使 QR 抽样和搜索轨迹不同。

完整探针增加同一中心的方向覆盖，但更新机会减少，也不会消除非线性或量化误差。
不能把完整 Jacobian 的方向覆盖直接等同于更低的最终恢复误差。

32 次最佳输入重复性测量和 worker 启动不计入搜索耗时；最佳 KV 重放完全一致。
验收仍严格要求 LSB RMSE < 0.125；未通过时不提交、不推进后续媒体位置。
只覆盖一张图片的首个位置，不能推断整张图片恢复或一般速度优势。

验证：111 项测试，109 通过，2 项可选 MLX 测试跳过；新增覆盖完整 768 方向的批处理、
尾轮缩减方向、精确查询预算与失败不提交。

## 复现

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/reconstruction/experiments/image_trust_region/run_spsa_joint.py \
  --reference-run kvidence/reconstruction/experiments/image_trust_region/results/0001_rounds300_seed17_20261020 \
  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \
  --output /tmp/new_full768_seed17 \
  --geometry --directions 768 --steps 300 --max-queries 45001
```

完整协议、代码哈希、每轮输入及实测 KV、最佳输入和轨迹保留在 results/0001_geometry768_seed*_20261021/。
