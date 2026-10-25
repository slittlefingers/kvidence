# CPA baseline: 21-token plaintext sample

Pixel 10 / Gemma 4 E2B Tensor G5, layer 5, plaintext sample 0001. Uses the existing independent pilot calibration `cpa_gemma4_pilot_dashboard_20261020`; this is not full-vocabulary calibration. Logical batch 256, physical capacity 128, optimized_v2, live telemetry enabled.

- Accepted: 21/21.
- Independently verified: 20/21; false acceptances: 1.
- Candidates queried: 22784.
- Online recovery: 54.363826 s; model decoder execution: 13.655803 s.
- Device setup: 11.248804 s; public-prefix prefill: 0.740387 s (both outside online time).
- Error: target token 2 (absolute position 5), expected ` an` (614), accepted ` a` (496). Later positions use the accepted prefix without ground-truth repair.

CPA retains separate calibrated K/V original-value L2 thresholds with the official 1% margin, and first-passing probability-order selection. It does not use guided LSB L2 < 4; the epsilon field is unused by CPA. Displayed concatenated L2 is diagnostic only.

The previous LSB guided run recovered 21/21 with 1,800 queries and 25.821 s online time (`../guided_lsb_plaintext0001_20261020`). These are separate single runs with different gates and logical batches, not a controlled benchmark establishing a general speedup.

See `protocol.json`, `trace.jsonl`, `events.jsonl`, `result.json`, `evaluation.json`, and `recovered.txt` for provenance, traces, and evaluation.
