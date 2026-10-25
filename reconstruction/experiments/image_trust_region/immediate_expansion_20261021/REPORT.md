# 单轮无改善即放大搜索尺度

从多尺度五轮实验的两个最佳点继续；仍为同一样本、第 5 层、第一个图片位置。
所有候选均使用实际手机 vision adapter 和 NPU，前缀不推进；只有真实 KV 误差改善才更新最佳点。

## 规则

- 每轮使用完整 768 方向，三个探针尺度；完整测量正负探针和宽范围步长候选。
- 一轮没有实测最佳值改善，下一轮就将主要探针幅度及岭回归位移半径扩大 2 倍。
- 扩大后只试一轮；有实测改善就保留新尺度。
- 试验没有改善则拒绝该档，保留最佳输入和最近有效半径，再试下一档。
- 主要半径最多为起始值的 4 倍；两档均失败或最大档下一轮没有改善则停止。
- 最小探针幅度始终固定为起始半径的 1/4，避免扩大时丢失原来的精细探索。
- 每轮最大探针幅度为当前主要半径的 4 倍，更新倍率仍为 1/64 到 4。

学习率沿用前次尾值，RNG 与 Adam 状态继续恢复。任何尺度的实测改善都会接受试验档，
因此接受新档本身不证明改善由大尺度造成；逐轮记录保留实际获胜候选及来源。

| Seed | 续跑前 RMSE | 最佳 RMSE | 降低 | 新增轮数 | 新增查询 | 新搜索秒数 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| 17 | 6.415238 | 5.920040 | 7.72% | 7 | 33104 | 85.37 | expansion_exhausted |
| 29 | 5.539718 | 5.523830 | 0.29% | 4 | 18917 | 49.07 | max_scale_plateau |

## 放大与回退事件

### Seed 17

- 第 5 轮，RMSE 5.920040：{'action': 'expand', 'previous': 0.08713840313369006, 'radius': 0.17427680626738012}
- 第 6 轮，RMSE 5.920040：{'action': 'reject_and_expand', 'rejected': 0.17427680626738012, 'restored': 0.08713840313369006, 'radius': 0.34855361253476025}
- 第 7 轮，RMSE 5.920040：{'action': 'reject_and_stop', 'rejected': 0.34855361253476025, 'restored': 0.08713840313369006}

最后改善：{'round': 4, 'rmse': 5.920040118107309, 'selected': {'kind': 'probe', 'scale': 0.25, 'direction': 379, 'sign': -1, 'file': 'round_0004_scale_0.25_probes', 'index': 1147}}

最终状态：{'initial_radius': 0.08713840313369006, 'patience': 1, 'factor': 2.0, 'trial_rounds': 1, 'max_multiplier': 4.0, 'radius': 0.08713840313369006, 'good_radius': 0.08713840313369006, 'stale': 3, 'trial_age': 1, 'trial': False, 'stopped': True, 'reason': 'expansion_exhausted'}

### Seed 29

- 第 1 轮，RMSE 5.539718：{'action': 'expand', 'previous': 0.08713840313369006, 'radius': 0.17427680626738012}
- 第 2 轮，RMSE 5.539718：{'action': 'reject_and_expand', 'rejected': 0.17427680626738012, 'restored': 0.08713840313369006, 'radius': 0.34855361253476025}
- 第 3 轮，RMSE 5.523830：{'action': 'accept_expansion', 'radius': 0.34855361253476025}
- 第 4 轮，RMSE 5.523830：{'action': 'stop_at_cap', 'radius': 0.34855361253476025}

最后改善：{'round': 3, 'rmse': 5.523829768602577, 'selected': {'kind': 'ridge', 'scale': 0.0625, 'damping': 0.001, 'factor': 0.015625, 'rms': 0.0003403843864395858, 'file': 'round_0003_updates', 'index': 0}}

最终状态：{'initial_radius': 0.08713840313369006, 'patience': 1, 'factor': 2.0, 'trial_rounds': 1, 'max_multiplier': 4.0, 'radius': 0.34855361253476025, 'good_radius': 0.34855361253476025, 'stale': 1, 'trial_age': 0, 'trial': False, 'stopped': True, 'reason': 'max_scale_plateau'}

## 验证与限制

113 项已有测试加 3 项策略测试：116 项，114 通过、2 项可选 MLX 测试跳过。
策略测试覆盖有改善重置平台、放大接受、失败回退、半径上限、状态恢复及拒绝无改善档。
集成测试核对实际批次限制、精确查询计数、小尺度锚点和失败不提交。
每次初始输入重放均与上一最佳 KV 一致；停止后的最佳输入各重复 32 次 KV 完全一致。
新增耗时不含 worker 启动和重复性诊断；累计耗时和查询包含此前全部搜索。
seed 29 的旧 20 轮等待策略按用户新指令中断；其完整 9 轮与第 10 轮已执行的 4608 个探针查询均计入成本。
该中断阶段耗时采用包含启动与恢复间隔的保守上界，因此 seed 29 累计耗时也是上界；本阶段耗时为实际计时。
从最后完整检查点恢复。中断后完整保存的两个探针块均未优于已有最佳值，未使用不完整的第三个块。
每轮成本为 4729 次候选测量，加阶段初始复核；这是带额外预算的续跑，不是独立同预算胜率比较。
真实验收仍严格要求 LSB RMSE < 0.125；未通过不提交、不推进后续位置。
没有读取原图或真实特征，没有恢复整张图片。单轮放大失败不是平台期证明，也不证明全局最优。
round_budget 只表示单次保护预算耗尽，不代表尺度探索结束。

代码：kvrecon/media/image_expansion.py；入口 run_full_multiscale.py --adaptive。
每轮保存 expansion_state.json 和 optimizer_state.npz，可从检查点继续。
实际运行参数：--rounds 100 --adaptive --plateau-rounds 1 --expansion-trial-rounds 1 --max-radius-multiplier 4。
