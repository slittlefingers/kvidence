#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
KVRECON_PY="kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python"
KVRECON_BASE="kvidence/reconstruction/experiments/image_trust_region"
for KVRECON_SEED in 29 17; do
  for KVRECON_MODE in fast hybrid; do
    KVRECON_EXTRA=""
    if [[ "$KVRECON_MODE" == hybrid ]]; then KVRECON_EXTRA="yes"; fi
    "$KVRECON_PY" "$KVRECON_BASE/run_spsa_fast.py" \
      --reference-run "$KVRECON_BASE/results/0001_rounds300_seed${KVRECON_SEED}_20261020" \
      --observation "$KVRECON_BASE/input/0001" \
      --output "$KVRECON_BASE/results/0001_${KVRECON_MODE}300_seed${KVRECON_SEED}_20261020" \
      --directions 64 --center-every 5 --steps 300 ${KVRECON_EXTRA:+--warm-start}
  done
 done
