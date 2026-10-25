#!/usr/bin/env bash
# Re-run the existing 21-token plaintext observation, then evaluate separately.
# Optional first argument: new output directory (relative to workspace or absolute).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

KVRECON_PY="kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python"
KVRECON_OUT="${1:-kvidence/reconstruction/results/plaintext21_rmse_$(date +%Y%m%d_%H%M%S)}"
KVRECON_DEVICE="${KVRECON_SERIAL:-YOUR_DEVICE_SERIAL}"

"$KVRECON_PY" kvidence/reconstruction/run.py run \
  --model-name gemma4-e2b \
  --observation kvidence/reconstruction/experiments/plaintext_short0001_layer5/input/observation \
  --method guided --distance-space lsb --distance-metric rmse --epsilon 0.125 \
  --initial-candidates 200 --prediction-candidates 50 --proposals 20 \
  --batch-size 128 --transport socket --native-mode optimized_v2 \
  --serial "$KVRECON_DEVICE" --live-events --output "$KVRECON_OUT"

# Ground truth is read only after reconstruction has exited.
"$KVRECON_PY" kvidence/reconstruction/run.py evaluate \
  --run "$KVRECON_OUT" \
  --truth kvidence/reconstruction/experiments/plaintext_short0001_layer5/input/private_eval/truth.json
"$KVRECON_PY" kvidence/reconstruction/run.py decode --run "$KVRECON_OUT"
printf 'Results: %s\n' "$KVRECON_OUT"
