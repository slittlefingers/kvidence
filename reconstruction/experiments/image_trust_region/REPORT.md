# Local ridge + trust-region: first NPU smoke test

- Material: image_chat_100 / 0001; acquired layer index 4 (layer 5), first image slot at position 5.
- Actual phone CPU vision adapter and NPU decoder; no desktop forward model.
- 768-dimensional float32 pre-adapter inputs, 16 search directions, 2 rounds.
- Queries: 67 candidates including initialization; the 67-query cap was selected for interface validation.
- Initial measured LSB RMSE: 57.963813806.
- Final measured LSB RMSE: 54.681419305.
- Improvement: 5.66%.
- Optimizer elapsed time including query RPC: 1.254 seconds (excludes worker setup).
- Acceptance gate: strict RMSE < 0.125; **not passed**. No feature committed.
- No whole-image recovery, image-content reconstruction, or subsequent-text recovery was demonstrated.

The prepared 260-slot code array and positions exactly match the previous image_0001
pilot observation. Full test suite: 96 passed, 2 opt-in model tests skipped.
This is a successful interface/numerical smoke test, not evidence of a speedup over
SPSA or a successful image recovery. No matched-budget baseline was run.

The initial `0001_smoke_20261020` run emitted host BLAS floating-status warnings.
`0001_smoke_checked_20261020` stopped on a vector-matrix contraction shape bug.
Both are retained. The corrected `0001_smoke_final_20261020` completed without those
warnings and is the run reported above. It produced the same final RMSE as the first run.

Run commands, parameter meanings and output files are documented in README.md.

## Official DFBGN comparison (2026-10-21)

Published dfbgn 0.1, unchanged solver source; only process-local NumPy int/float
alias compatibility. Same observation, physical adapter and NPU, first image
position, and initial input replay verification. Residuals are integer-code
errors / sqrt(1024), so the solver minimizes MSE and acceptance remains strict
LSB RMSE < 0.125. Initial radius is recorded per run (package default or explicit 2.7712812921); block 64; single-candidate
callbacks. Each run has a 173-second optimization cap, checked at callbacks.
Startup and repeatability diagnostics are excluded, as in the prior comparison.
No clean image or hidden text is read. Solver exit success does not imply KV
acceptance. The locked-phone launch produced zero queries and is excluded.

| Run | Initial RMSE | Best RMSE | Stage seconds | Stage queries | Acceptance |
|---|---:|---:|---:|---:|---|
| 0001_joint300_seed29_20261020 | 57.9638 | 15.4876 | 172.64 | 45001 | False |
| 0001_dfbgn173_seed29_20261021_retry | 57.9638 | 55.2148 | 18.16 | 114 | False |
| 0001_joint300_seed17_20261020 | 65.0270 | 13.1809 | 173.04 | 45001 | False |
| 0001_dfbgn173_seed17_20261021 | 65.0270 | 62.2825 | 22.56 | 116 | False |
| 0001_dfbgn_radius173_seed17_20261021_retry | 65.0270 | 59.0338 | 25.77 | 130 | False |
| 0001_dfbgn_warm173_seed17_20261021 | 13.1809 | 13.1809 | 23.11 | 118 | False |

The two default-radius runs terminated early because delta reached rhoend,
not because the KV gate passed. They used 114/116 calls and 18.16/22.56 seconds.
Their measured best states repeated exactly 32 times. The default initial
radius is roughly 0.3 in Euclidean norm (about 0.01 per-coordinate RMS in
768 dimensions). The larger-radius follow-up uses 0.1*sqrt(768)=2.7712812921, matching the
Euclidean length of a prior SPSA perturbation. Its disconnected first launch
had zero queries and is excluded; the retry is listed separately above.
These results reject the current default configuration as an improvement on
this sample, not every possible DFBGN configuration.

The larger-radius seed-17 run reduced RMSE from 65.0270 to 59.0338 in
25.77 seconds / 130 queries, then hit the radius limit. The seed-17 warm start
remained exactly 13.1808706678 after 118 queries / 23.11 additional seconds.
Both best states repeated exactly 32 times. Thus these tested DFBGN settings
neither improve cold-start recovery nor break the existing plateau; they do
not establish that all parameterizations fail or that a global minimum was found.

Warm-start runs begin from the corresponding joint method's saved best point;
these are plateau-refinement experiments, not equal-total-cost comparisons.
Total costs including that preceding search are in summary.json. Single sample
and sequential timings are exploratory, not a general speedup claim. Physical
query efficiency can differ from wall-time efficiency because each native call
uses the compiled execution shape even for a single valid candidate. No complete
image recovery or subsequent text recovery has been demonstrated by these runs.

Artifacts: `dfbgn_comparison_20261021/summary.json`, `curves.json`, and each run's
`optimization.jsonl`, `query_*.npz`, `best_input.npy`, `best_kv.npy`, `result.json`.
Entry point: `run_dfbgn.py`; rerun analysis using `analyze_dfbgn.py`.

## Direct first-block input search with vocabulary seeds (2026-10-21)

This pilot bypasses the vision adapter and optimizes its downstream, 1536-dimensional
input to the first Transformer block. The observation remains layer-5 KV at the
first image position (position 5) of image_chat_100/0001; this is not first-layer
KV lookup. The fixed prefix and media PLE row 0 are preserved.

We extract 1,024 uniformly sampled, distinct vocabulary embeddings from the deployed
runtime, measure their KV using the media execution path, and initialize four
independent continuous branches from the best four. This is a sampled seed search,
not a nearest-neighbor search over the complete vocabulary. Refinement uses the
multiscale ridge/SPSA proposals directly in the 1536-dimensional input space,
without invoking the adapter. Neither the original image nor private labels are read.

| Measurement | Result |
|---|---:|
| Best vocabulary seed RMSE | 54.39537 |
| Final measured code-space RMSE | 27.13940 |
| Continuous refinement rounds | 183 |
| Candidate evaluations including seeds | 39,271 |
| Seed extraction time | 8.75 s |
| Seed extraction plus seed scoring time | 11.80 s |
| Seed and optimization time | 132.10 s |
| Refinement budget | 120 s |
| Additional repeatability evaluations | 32 |
| Acceptance target | RMSE ≤ 0.1 |
| Acceptance | Failed |

All 32 repeated best-state measurements are identical. The stored final refinement
clock includes the short repeatability diagnostic; the 132.10-second optimization
clock excludes it. Startup is excluded. The run stopped at its time budget, so it
does not establish convergence or an irreducible error floor.

The seed-to-final decrease demonstrates that this direct-input interface works,
but does not demonstrate automatic alignment of visual features to vocabulary
embeddings. The final distance is worse than the preceding joint seed-17 result
(13.18087); different initializations, dimensions and time budgets prevent a
controlled optimizer comparison. No whole image or subsequent text was recovered.
An unconstrained recovered input can also lie outside the adapter's output range,
so KV matching alone would not establish recovery of the original visual features
or image pixels.

Entry point: `run_token_seeded.py`.
Artifacts: `results/0001_token_seeded_direct_20261021/`, including `protocol.json`,
`seed_results.json`, measured candidate arrays, `best_input.npy`, `best_kv.npy`,
and `result.json`. Validation: 105 reconstruction tests ran, 103 passed and 2 skipped.

## Three-round text-style vocabulary search (2026-10-21)

Same image position and layer-5 target, with no adapter or continuous refinement.
Reuses the text ExistingRidge proposal implementation and full original vocabulary.
Initial seeds are 50 text-probability-ranked IDs plus 150 random IDs (seed 217).
The probability ranking is a text prior, not a visual-token distribution.
Proposal features use the configured desktop embedding table; actual candidate
inputs are deployed embeddings with media PLE 0.

| Stage | Cumulative candidates | Best measured LSB RMSE |
|---|---:|---:|
| Initial seeds | 200 | 54.96160 |
| Guided round 1 | 220 | 50.36246 |
| Guided round 2 | 240 | 50.15023 |
| Guided round 3 | 260 | 50.15023 |

Search took 6.79 seconds, excluding setup. This is a limited candidate-selection
pilot, not a convergence test. All three rounds completed and measurements were
saved; result-summary serialization initially failed on a NumPy bool and was
fixed, with the summary rebuilt from the saved trace without additional queries.
No candidate passed RMSE ≤ 0.1, and no original image was recovered.
Artifacts: `results/0001_vocab_guided3_20261021/`; entry point `run_vocab_guided.py`.
