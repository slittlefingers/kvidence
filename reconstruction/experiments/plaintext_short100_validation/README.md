# 新明文 KV 数据集的单样本恢复验证

完整数据集在 [plaintext_chat_100](../../../datasets/shadow_chat/plaintext_chat_100/README.md)，含 100 条原文及真实回复后 KV。本实验只验证预先抽定的 `0076`，不代表 100 条都已做恢复。

选择规则在采集前写入 `selection.json`：`random.Random(20261019).choice(sorted(ids))`。目标为原文 15 个 token，第 5 层 producer（索引 4），4 个聊天模板 token 作为公开前缀。

`input/observation` 是数据集相应观测文件的逐字节副本。原文和目标标签没有复制过来，仍在数据集的 `private_eval/0076` 中。恢复阶段 audit guard 禁止读取源数据和标签；对比结束后才单独评价。

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
EXP="$APP/experiments/plaintext_short100_validation"
OUT="$APP/results/plaintext_short100_0076_layer5_cpa_vs_guided_native_v2"

"$PY" kvidence/datasets/shadow_chat/scripts/build_plaintext_short.py
"$PY" "$EXP/check_prefix.py"
"$PY" "$APP/run.py" compare --model-name gemma4-e2b \
  --baseline collision_plus_cpa --observation "$EXP/input/observation" \
  --calibration "$APP/assets/cpa_gemma4_pilot_native_v2/calibration.json" \
  --allow-pilot-calibration --batch-size 256 --transport socket \
  --native-mode optimized_v2 --repeats 2 --output "$OUT"
"$PY" "$APP/run.py" evaluate-comparison --comparison "$OUT" \
  --truth kvidence/datasets/shadow_chat/plaintext_chat_100/private_eval/0076/truth.json
```

已有输出不能覆盖，复跑需要换输出目录。两方法各两次交替执行，同样本、模型、词表和实现版本。CPA 仍为独立 8×1024 pilot 校准，尚非完整全文全词表校准；guided 保留原始 K+V L2<4。

结果保存在 `results/plaintext_short100_0076_layer5_cpa_vs_guided_native_v2`，完成后生成 `REPORT.md`。`prefix_validation/validation.json` 只检查公开模板，3 个非 BOS 位置距离均为 0。
