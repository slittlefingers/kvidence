# 样本 0001：未加密原文，第 5 层恢复

直接使用 `pilot_100_short/prompts/plaintext/0001.txt`，原文 87 UTF-8 bytes、21 个目标 token；不补长、不截断、不添加密文存储问题。`pair_validation.json` 记录它与此前密文的 AES-GCM 认证配对检查，不保存密钥。

已完成实际采集与各两次 CPA/guided 恢复，见 [结果报告](../../results/plaintext_short0001_layer5_cpa_vs_guided_native_v2/REPORT.md)。CPA 仍使用独立 8×1024 pilot 校准，未完成全文全词表校准。

## 目录

- `capture_input.json`：输入 Ghost 的原样 user prompt。
- `capture/`：回复结束后采集的真实 KV、布局、回复、校验和设备身份。
- `prepare.py`：检查来源哈希，将第 5 层 KV 位置 4–24 导入最小恢复观测。
- `input/observation/`：恢复进程可见的目标层 KV 和 4-token 公开聊天模板前缀。
- `input/private_eval/`：事后评价标签；恢复进程不得读取。
- `check_prefix.py`、`prefix_validation/`：只用公开模板做位置/数值一致性检查，3 个非 BOS 位置均为 L2=0。

## 实际执行命令

以下输出已存在，工具会拒绝覆盖。复跑时应使用新的实验/结果目录，并调整准备脚本的输入输出位置；保留本次快照和结果。

```bash
PY=kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python
APP=kvidence/reconstruction
EXP="$APP/experiments/plaintext_short0001_layer5"
OUT="$APP/results/plaintext_short0001_layer5_cpa_vs_guided_native_v2"

python3 kvidence/experiments/npu_batch_capture/capture.py \
  --input "$EXP/capture_input.json" --output "$EXP/capture" \
  --model gemma4-e2b --serial YOUR_DEVICE_SERIAL --no-install-test
"$PY" "$EXP/prepare.py"
"$PY" "$EXP/check_prefix.py"
"$PY" "$APP/run.py" compare --model-name gemma4-e2b \
  --baseline collision_plus_cpa --observation "$EXP/input/observation" \
  --calibration "$APP/assets/cpa_gemma4_pilot_native_v2/calibration.json" \
  --allow-pilot-calibration --batch-size 256 --transport socket \
  --native-mode optimized_v2 --repeats 2 --output "$OUT"
"$PY" "$APP/run.py" evaluate-comparison --comparison "$OUT" \
  --truth "$EXP/input/private_eval/truth.json"
```

两种方法使用自己的恢复前缀；没有将真实目标前缀的缓存装入执行器。模型、原词表、搜索配置、核心实现、APK、运行库及校准资产与此前密文对比一致。明文与密文的长度及外层模板不同，报告单独说明了不能据此隔离加密变量的限制。
