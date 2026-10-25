#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
KVRECON_PY="kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python"
KVRECON_BASE="kvidence/reconstruction/experiments/image_trust_region"
for KVRECON_SEED in 29 17; do
  "$KVRECON_PY" "$KVRECON_BASE/continue_spsa.py" \
    --source "$KVRECON_BASE/results/0001_spsa300_seed${KVRECON_SEED}_20261020" \
    --observation "$KVRECON_BASE/input/0001" \
    --output "$KVRECON_BASE/results/0001_spsa1000_seed${KVRECON_SEED}_20261020" \
    --total-rounds 1000 --plateau-rounds 100
 done
