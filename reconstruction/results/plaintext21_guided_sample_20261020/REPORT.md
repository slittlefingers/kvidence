# 21-token sample: guided and CPA

Same Pixel 10, Gemma 4 E2B Tensor G5, plaintext sample 0001, layer 5. CPA ran first, followed by guided, with live dashboard telemetry enabled. Evaluation reads the true token IDs only after recovery finishes.

| Method | Correct tokens | False acceptances | Candidate queries | Online s | Decoder s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Guided (LSB L2 < 4) | 21/21 | 0 | 1800 | 20.719456 | 4.128958 |
| CPA (pilot-calibrated original-value K/V thresholds) | 20/21 | 1 | 22784 | 54.363826 | 13.655803 |

All guided candidate acceptance and commit distances are zero LSB. CPA accepts ` a` (496) instead of ` an` (614) at absolute position 5. CPA's remaining recovered IDs match the target, using its own accepted prefix.

These are single runs, with different acceptance rules and logical batches (guided 128, CPA 256; physical NPU capacity 128). They are not a repeated benchmark and do not establish a general speedup. Online timing excludes device initialization and public-prefix prefill; CPA offline calibration is also excluded. CPA calibration is a pilot subset, not full-vocabulary calibration.

Guided initialization: device 11.349 s, surrogate 2.543 s, public prefix 0.817 s.

Command from workspace root:

```bash
bash kvidence/reconstruction/run_plaintext21_lsb.sh kvidence/reconstruction/results/plaintext21_guided_sample_20261020
```

The recorded output directory already exists; choose a new directory for a rerun, or omit the argument to generate a timestamped directory.
