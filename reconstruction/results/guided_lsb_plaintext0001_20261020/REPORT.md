# 21-token plaintext: LSB guided reconstruction

Pixel 10 / Gemma 4 E2B Tensor G5, producer layer 5 (index 4), sample plaintext 0001. The observation is the same after-response capture used previously; recovery targets are 21 user-prompt tokens at positions 4–24, not the response.

## Metric and procedure

Native INT8 K/V codes are widened before subtraction. Candidate fitting, ranking, search acceptance and commit verification use concatenated K+V L2 in LSB-step units, strictly < 4. This is not a per-coordinate ±4 gate. The original full vocabulary, 50 probability seeds + 150 random seeds, 20 proposals, physical batch 128 and optimized_v2 execution are retained. No query cap. Only the recovered prefix is replayed; evaluator labels are read in a separate process after reconstruction.

## Results

| Run | Correct | False acceptances | Candidates | Online seconds | Decoder seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| New LSB L2 | 21/21 | 0 | 1800 | 25.820889 | 4.313017 |
| Previous raw-value L2 (historical) | 21/21 | 0 | 1820 | 23.493222 | 4.210506 |

All 21 selected candidates and their committed KV have LSB L2 = 0. Independent evaluation confirms exact token IDs. Historical timing is not a controlled paired timing experiment; the new run does not establish a general speed improvement. CPA was not rerun.

Online time excludes device initialization (14.199 s), surrogate setup (2.505 s), and public-prefix prefill (0.676 s). Real-time telemetry was enabled.

## Artifacts and rerun

- `protocol.json`: model, code hashes, distance units and parameters.
- `trace.jsonl` / `events.jsonl`: all tested candidate IDs, distances and stages.
- `result.json`, `evaluation.json`, `recovered.txt`: final result and independent evaluation.
- From workspace root: `bash kvidence/reconstruction/run_plaintext21_lsb.sh`. Uses a new timestamped output directory by default; the optional first argument selects an unused output directory. The script runs reconstruction, independent evaluation, then decoding. `--live-events` writes events; a separate `monitor_attack.py` process relays them to Ghost.
