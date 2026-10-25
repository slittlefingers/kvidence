# 平台点的多尺度探针与扩展步长实验

材料为 image_chat_100/0001、第 5 层、首个图片位置；从两个已满足连续 20 轮不改善的平台点开始。
保持 768 个完整正交方向。每轮在同一中心、同一方向基上测量 h/4、h、4h 三种探针幅度。
h=0.0871384031 RMS，三种幅度约为 0.02178、0.08714、0.34855。
每个尺度求 4 种阻尼的岭回归方向，分别测量 1/64、1/32、1/16、1/8、1/4、1/2、1、2、4 倍步长。
同时保留每尺度 4 个 Adam 候选及中心重放，共 121 个更新候选。
每轮 3×1536+121=4729 个实际查询候选，5 轮加初始查询共 23,646 个。

| Seed | 原平台 RMSE | 新最佳 RMSE | 降低 | 新查询数 | 新搜索秒数 |
|---|---:|---:|---:|---:|---:|
| 17 | 7.714257 | 6.415238 | 16.84% | 23646 | 61.24 |
| 29 | 9.663731 | 5.944897 | 38.48% | 23646 | 61.16 |

## 实测改善来自哪里

### Seed 17

- 第 2 轮：RMSE 7.709192，来源 {'kind': 'adam', 'scale': 4.0, 'factor': 0.5, 'rms': 0.00027877595533395413, 'file': 'round_0002_updates', 'index': 117}。
- 第 3 轮：RMSE 7.477701，来源 {'kind': 'ridge', 'scale': 4.0, 'damping': 1.0, 'factor': 1.0, 'rms': 0.12104275471062091, 'file': 'round_0003_updates', 'index': 113}。
- 第 4 轮：RMSE 7.178867，来源 {'kind': 'ridge', 'scale': 4.0, 'damping': 0.1, 'factor': 0.5, 'rms': 0.17427680627269487, 'file': 'round_0004_updates', 'index': 103}。
- 第 5 轮：RMSE 6.415238，来源 {'kind': 'ridge', 'scale': 4.0, 'damping': 0.1, 'factor': 0.5, 'rms': 0.1706549758574792, 'file': 'round_0005_updates', 'index': 103}。

第一轮的更新候选均使用同一个原平台中心，因而可以直接查看尺度和步长对候选误差的影响：

| 探针倍率 | 原有步长子集最佳 RMSE | 扩展步长最佳 RMSE |
|---|---:|---:|
| 0.25 | 7.928083 | 7.863956 |
| 1.0 | 7.834594 | 7.770952 |
| 4.0 | 7.857247 | 7.857247 |

### Seed 29

- 第 1 轮：RMSE 8.639820，来源 {'kind': 'ridge', 'scale': 4.0, 'damping': 0.1, 'factor': 1.0, 'rms': 0.3485536131167105, 'file': 'round_0001_updates', 'index': 104}。
- 第 2 轮：RMSE 7.627241，来源 {'kind': 'ridge', 'scale': 4.0, 'damping': 1.0, 'factor': 1.0, 'rms': 0.14244160131217706, 'file': 'round_0002_updates', 'index': 113}。
- 第 3 轮：RMSE 7.242588，来源 {'kind': 'ridge', 'scale': 4.0, 'damping': 0.1, 'factor': 1.0, 'rms': 0.3485536129596764, 'file': 'round_0003_updates', 'index': 104}。
- 第 4 轮：RMSE 6.344751，来源 {'kind': 'ridge', 'scale': 4.0, 'damping': 1.0, 'factor': 1.0, 'rms': 0.11102551168357215, 'file': 'round_0004_updates', 'index': 113}。
- 第 5 轮：RMSE 5.944897，来源 {'kind': 'probe', 'scale': 0.25, 'direction': 537, 'sign': -1, 'file': 'round_0005_scale_0.25_probes', 'index': 1305}。

第一轮的更新候选均使用同一个原平台中心，因而可以直接查看尺度和步长对候选误差的影响：

| 探针倍率 | 原有步长子集最佳 RMSE | 扩展步长最佳 RMSE |
|---|---:|---:|
| 0.25 | 10.160336 | 10.148651 |
| 1.0 | 10.146438 | 9.930373 |
| 4.0 | 8.639820 | 8.639820 |

## 口径与限制

原有步长子集为岭回归倍率 0.25、0.5、1、2 和该尺度的四个 Adam 候选。
所有探针和更新候选均由真实 NPU 测量，最佳值包括探针；没有按预测值接受候选。
初始 KV 与此前最佳 KV 完全一致，停止后最佳输入重复测量 32 次 KV 完全一致。
恢复了此前 RNG 和 Adam 状态，未随机重启；各尺度使用相同的原 Adam 状态计算候选。
本轮实测胜出候选所属尺度的梯度用来更新 Adam 状态；未改善时使用原尺度梯度。
本实验同时扩大探针和更新尺度，后续轨迹不同，不能仅据最终误差把作用归因到单一参数。
三尺度开销高于单尺度；这不是与单尺度同预算的独立速度优越性试验。
本次仅各 5 轮，状态 round_budget 表示预算用完，不代表发现新平台或全局最优。
真实验收仍为 LSB RMSE < 0.125；未通过不提交，也没有恢复整张图片或后续文本。
未读取原图、真实特征或隐藏文字；累计成本包含此前搜索，详见 summary.json。
新增搜索耗时不含 worker 启动和额外 32 次诊断；状态读取成本计入累计耗时。

测试：113 项，111 通过，2 项可选 MLX 测试跳过；新增测试覆盖三尺度的真实查询预算、
128 条物理批次限制、步长范围、保留最佳值和失败不提交。

## 复现

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/reconstruction/experiments/image_trust_region/run_full_multiscale.py \
  --warm-run kvidence/reconstruction/experiments/image_trust_region/results/0001_geometry768_plateau_seed17_20261021 \
  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \
  --output /tmp/new_full_multiscale17 --rounds 5
```

各尺度探针、更新候选及其实际位移 RMS、实测 KV、来源、逐轮轨迹和协议/代码哈希均保留在对应 results/ 目录。
