# Image recovery: local ridge + trust region

This is an opt-in image optimizer. Existing Adam/SPSA experiments and the text
search algorithm are unchanged. Entry point: `kvrecon/media/image.py`, function
`recover_trust_region`; numerical solver: `kvrecon/media/image_trust.py`.

It optimizes one image position at a time, using the phone's actual
`vision_adapter_280` (CPU) followed by the phone NPU decoder. The interface takes
768-dimensional **float32 pre-adapter vectors**, not INT8 input codes. The adapter
produces 1536-dimensional decoder inputs. Candidate batch rows are alternatives at
one fixed position, not simultaneous optimization of multiple image positions.

For each position:

1. Start from a seeded random vector, independent of the original image.
2. Probe positive/negative directions in a rotating orthonormal subspace.
3. Fit a ridge map from subspace displacements to measured KV changes.
4. Solve a damped least-squares trust-region subproblem.
5. Measure the proposed step on NPU; use actual/predicted improvement to adjust
   radius. Retain the best actually measured candidate, including probes.
6. Commit only after the query passes; independently require the committed KV to
   pass before moving to the next position. A failure stops sequential recovery.

A previous successful displacement is retained as one direction in the next
subspace. Default subspace dimension is 16; each full round costs 32 probe
candidates and one proposed candidate, plus one initial query per position.
Ridge input coordinates are divided by radius; outputs remain in the declared KV
units. No original image, hidden text, original features, or target KV prefix is
supplied to the optimizer. This is an experimental local method, without a global
optimality or historical-input uniqueness guarantee.

## Material and preparation

Source: `kvidence/datasets/shadow_chat/image_chat_100`, containing 100 captured
samples. Preparation reads the manifest and acquired KV snapshot only. The known
Gemma capture template and image dimensions provide image-slot boundaries. This
is a declared boundary side-information assumption, not automatic boundary
recovery from an arbitrary memory image. The prepared attack directory contains
only image KV codes, positions, public prefix, and provenance metadata.

From the workspace root:

```bash
export PYTHONPATH="$PWD/kvidence/reconstruction"
KVRECON_PY="kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python"

# Already prepared for sample 0001. For another sample, change both 0002 values.
"$KVRECON_PY" -m kvrecon.media.image trust-region prepare \
  --sample 0002 \
  --output kvidence/reconstruction/experiments/image_trust_region/input/0002

# New output directory required; never overwrites an existing experiment.
"$KVRECON_PY" -m kvrecon.media.image trust-region run \
  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \
  --output "kvidence/reconstruction/experiments/image_trust_region/results/0001_$(date +%Y%m%d_%H%M%S)" \
  --max-positions 1 --max-queries 2048 --directions 16 \
  --distance-space lsb --distance-metric rmse --epsilon 0.125 \
  --serial YOUR_DEVICE_SERIAL --no-install
```

`--max-positions 0` requests every image slot, stopping at the first failure.
`--max-queries` is a per-position candidate budget, not a number of batches.
Preparation and recovery are separate: the recovery process enables the existing
private-data access guard, which blocks direct dataset and private-label reads.
The current reader supports the retained INT8 full-attention Gemma capture;
it rejects local layers/ring-mapped snapshots instead of assuming slot mappings.

The default gate follows the new LSB RMSE convention. To reproduce the historical
image experiment's raw L2 < 5 criterion, explicitly pass:

```bash
--distance-space original --distance-metric l2 --epsilon 5
```

These are different criteria and must not be compared as equivalent thresholds.
All radius/regularization settings are provisional, not tuned claims of speedup.

## Outputs

- `protocol.json`: source hashes, sample/observation identity, parameters and units.
- `device/`: NPU worker/transport logs.
- `recovery/position_XXXX/initial.npz`: first input and measured KV.
- `round_*_probes.npz`, `round_*_trial.npz`: queried inputs and actual KV.
- `optimization.json`: radius, predicted/actual improvement, queries and distance.
- `best_input.npy`, `best_kv.npy`: best measured candidate and its KV.
- `features.npy`, `commit_kv.npy`: adapter output and independent commit verification,
  written only when a query passed.
- `accepted_features.npy`: only positions that passed both checks.
- top-level `result.json`: full-image completeness distinguished from a position limit.

The output is replacement adapter features, **not a recovered image file**.
This runner does not yet evaluate visual content or recover subsequent text.

## Matched-budget SPSA comparison

`run_spsa_comparison.py` runs averaged two-sided SPSA gradient estimates with Adam
updates, without ridge fitting. This is an SPSA+Adam variant, not a claim to
reproduce every detail of Spall's classical step-size schedule. It reuses the
reference run's exact initial vector and checks that its first KV replay is
identical. Both methods use 16 directions and 33 candidates per round; 300 rounds
cost 9,901 search candidates. Initialization search is disabled. SPSA's
perturbation is coordinate-wise 0.1 with slow decay; Adam starts at 0.05 with cosine
decay. These provisional hyperparameters differ from the trust-region radius
policy and have not been tuned on multiple images. All KV coordinates contribute
to a half-MSE loss; acceptance remains strict LSB RMSE < 0.125.

```bash
"$KVRECON_PY" kvidence/reconstruction/experiments/image_trust_region/run_spsa_comparison.py \
  --reference-run kvidence/reconstruction/experiments/image_trust_region/results/0001_rounds300_seed29_20261020 \
  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \
  --output kvidence/reconstruction/experiments/image_trust_region/results/new_spsa_seed29
```

The best actually measured perturbation or update is retained; the optimizer's
current iterate can temporarily become worse. The report distinguishes current
and best RMSE. A 32-candidate repeatability check follows the search, counted
separately from its budget; failed candidates are not committed. This experiment
still concerns one image position, not whole-image or downstream text recovery.

## Continue SPSA to 1,000 total rounds

`continue_spsa.py` reconstructs the original 300-round Adam moments and RNG state
from saved perturbations and KV measurements. Every replayed perturbation input
and updated iterate must match the original saved input exactly. Before continuing,
the phone independently replays the current and best vector and must reproduce
the recorded KV. The old experiments remain immutable.

The original cosine learning-rate schedule ends after 300 rounds; continuation
holds its declared floor (0.005) instead of restarting or lengthening the original
schedule. Perturbation decay continues with the absolute round index. No ridge
model, restart, input change, or best-point reset is introduced.

Stop at 1,000 total rounds or at 100 consecutive rounds without ANY strictly smaller
best measured RMSE (or successful threshold acceptance). A plateau is empirical,
not a mathematical global convergence claim. The continuation saves 25-round
Adam/RNG checkpoints, all measured candidates, and a combined 1..N trace; original
1..300 candidate payloads remain in the source run.

```bash
bash kvidence/reconstruction/experiments/image_trust_region/run_spsa1000.sh
```

The script serializes seeds 29 and 17 on the same phone. Its dated output paths
must not already exist. Search budgets include the original queries, while worker
setup and 34 continuation diagnostic queries are reported separately.

## Full-batch and ridge-warm-start variants

`run_spsa_fast.py` adds `--center-every` (default 5), `--directions 64` and
`--warm-start`. Every round measures 128 perturbed candidates in one physical
batch; only every fifth round and the last round measure the new center. An
unmeasured center's distance is stored as null, never inferred or accepted.
The best actual probe or measured center is always retained. After 300 rounds,
there are 38,461 search candidate measurements, not the baseline's 9,901.
The gradient uses 64 independent Rademacher directions and otherwise the same
SPSA/Adam and 300-round decay schedules. This is a combined modification, not an
ablation isolating the effect of each change.

With `--warm-start`, the starting input/KV are the preceding ridge run's measured
best. The entire preceding ridge run's queries and optimization time are charged
in totals, including its non-improving tail. Initialization replay must match that
recorded KV exactly. Warm starts do not use original media or hidden text.

```bash
bash kvidence/reconstruction/experiments/image_trust_region/run_fast_variants.sh
```

This runs both variants for seeds 29 and 17 sequentially. Wall-clock efficiency
and candidate efficiency must be compared separately because budgets differ.
`analyze_fast.py` generates complete wall-time curves and time-to-common-distance
checkpoints; no threshold success should be inferred from a lower error alone.


## Official DFBGN comparison

`run_dfbgn.py` connects the published `dfbgn==0.1` solver to the same measured
Pixel residual `(codes - target) / sqrt(d)`. The sum of squared residuals equals
code-space MSE. Defaults: fixed subspace size 64, 173-second optimization budget,
45,001 maximum callbacks, package-default initial radius, final radius 1e-8.
There is no modification to the installed optimizer source. Two process-local
NumPy compatibility aliases (`np.int=int`, `np.float=float`) restore removed
names used by the published package. Solver/package hashes are recorded.

```bash
KVRECON_PY="kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python"
KVRECON_BASE="kvidence/reconstruction/experiments/image_trust_region"
"$KVRECON_PY" -m pip install dfbgn==0.1
"$KVRECON_PY" "$KVRECON_BASE/run_dfbgn.py" \
  --reference-run "$KVRECON_BASE/results/0001_rounds300_seed29_20261020" \
  --observation "$KVRECON_BASE/input/0001" \
  --output "$KVRECON_BASE/results/dfbgn_$(date +%Y%m%d_%H%M%S)" \
  --seconds 173 --block 64
```

Change the reference seed folder for another existing initialization. Optional
`--warm-run` accepts a measured joint/SPSA result folder and charges its previous
cost in totals. `--rhobeg` controls the Euclidean trust-region radius; it is not
per-coordinate RMS. The official callback is sequential, with one candidate per
phone query; no unimplemented batch speedup is claimed.

Every physical query saves its input and measured KV, with `optimization.jsonl`
for time/distance tracking and `result.json` for the final status. A solver success
flag can mean its radius became small: **only the separate strict KV RMSE gate
and subsequent commit verification count as acceptance**. No original image,
text or desktop-model KV is used. Timings exclude worker startup and the final
32-repeat diagnostic. The time budget is checked between callbacks and can
overshoot by one call or solver work between calls.

## Vocabulary-seeded direct embedding recovery

`run_token_seeded.py` bypasses the vision adapter entirely. It extracts 1536-D
embeddings from the deployed phone embedding signature, including runtime
scaling, and injects them through `query_embeddings` with the image-position
PLE configuration (row zero). This is not a normal text-token-ID replay.
The observed target remains layer 5, position 5, with the same public prefix.

The initial pilot samples 1,024 IDs uniformly without replacement from the
full original vocabulary, keeps four measured best seeds, and refines them as
separate branches. It does not claim to find the nearest entry in the whole
vocabulary. Coordinates are scaled by the RMS of the seed embedding matrix,
independently of the target; actual vectors and measured KV are saved. Four
perturbation scales and regularized secant proposals use measured center KV.
Only physical measurements and subsequent commit verification can accept a
position. The explicit pilot gate is LSB RMSE <= 0.1.

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/reconstruction/experiments/image_trust_region/run_token_seeded.py \
  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \
  --output "kvidence/reconstruction/experiments/image_trust_region/results/direct_$(date +%Y%m%d_%H%M%S)" \
  --seed-candidates 1024 --seconds 120 --epsilon 0.1
```

`seed_results.json` records IDs and their measured seed distances. The run saves
all trial input/KV pairs, `optimization.json`, `best_input.npy` (post-adapter,
1536 dimensions), `best_kv.npy`, and `result.json`. Refinement time budget excludes
seed extraction/scoring; total optimization time includes both. Worker startup
and final repeatability checks are excluded from optimizer_seconds. This search
can leave the adapter output range, and recovering a matching embedding alone
does not reconstruct image pixels or prove later-text recovery.

### Three-round text-style vocabulary pilot

`run_vocab_guided.py` reuses `ExistingRidge`, the text search surrogate, with
50 probability seeds, 150 random seeds, and three rounds of 20 proposals over
the complete vocabulary. The configured desktop embedding asset is used only
for proposal features, exactly as in text search; actual candidate inputs are
extracted from the deployed runtime and measured with media PLE 0 at the fixed
first image position. The adapter and continuous refinement are not called.
The probability ranking is a text prior at the media prefix, not a visual-token
probability distribution. Distances are measured code-space RMSE.

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/reconstruction/experiments/image_trust_region/run_vocab_guided.py \
  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \
  --output /tmp/image_vocab_guided3 --rounds 3
```

## Measured-center orthogonal joint proposals (2026-10-21)

`run_spsa_joint.py --geometry` enables an experimental alternative; omitting the
flag preserves the existing joint method. It uses the same 64 probe pairs and
22 update candidates per round (45,001 search candidates for 300 rounds).

For input width d, draw Q with orthonormal columns and set D = sqrt(d) Q^T.
The rows of D have RMS one and D D^T = d I. Measure F(u ± c D_i) and the actual
incumbent F(u). Define B_i = (F(u+c D_i)-F(u-c D_i))/(2c). Solve

    min_w ||B^T w - (target - F(u))||² + lambda ||w||².

The proposal is delta_u = D^T w; its RMS equals ||w||. Each damped solution
is capped to RMS c before trying multipliers 0.25, 0.5, 1 and 2. Four damping
values and four Adam candidates remain, as in the baseline. The center uses
its saved actual KV, requiring no extra center query. All probes and updates
are actually measured; only the best measured input becomes the next center.
Query and independent commit gates remain unchanged. The selected input is a
768-dimensional pre-adapter vector, not an accepted or original image feature
unless those gates pass.

This jointly changes probe geometry, residual anchoring and ridge step cap;
an improvement cannot be attributed to one component without an ablation.
`changed_code_fraction` measures paired KV code differences (unlike equal scalar
losses, which can hide distinct KV). `midpoint_bias_rmse` measures the difference
between average paired midpoint KV and the measured center.

Example (choose a new output directory):

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/reconstruction/experiments/image_trust_region/run_spsa_joint.py \
  --reference-run kvidence/reconstruction/experiments/image_trust_region/results/0001_rounds300_seed17_20261020 \
  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \
  --output /tmp/new_geometry_seed17 \
  --geometry --steps 300
```

`analyze_geometry.py --runs RUN... --output NEW_JSON` fits paired secant models
from saved baseline probes and scores separately measured update candidates.
These are correlated within-run holdouts, not proof that new proposals work.
`analyze_geometry_comparison.py --output NEW_JSON` checks matching observations,
initial vectors, hyperparameters and candidate budgets before summarizing the
seed-17/29 comparisons. Results are under `geometry_comparison_20261021/`.

## Tiny nonlinear surrogate (2026-10-21)

`run_spsa_joint.py --geometry --surrogate` fits 64 fixed tanh features with a
full linear skip using local dual ridge regression. It uses the current and
preceding measured probe batches (at most 256 rows); no original image/features
or model gradients are read. The output predicts all 1,024 KV codes. Input
coordinates are centered at the actual incumbent and scaled by probe RMS.
Nonlinear features are also centered, anchoring the predicted center to its
measured KV. Kernel scale is normalized by feature width and regularization by
mean kernel diagonal. Default tanh weight is 0.1; `--nonlinear 0` supplies the
matched linear-only control.

Each round replaces the 16 ridge proposals with 16 surrogate-optimized inputs.
Their starts are the original measured-center ridge proposals. Twelve cheap
surrogate Adam steps are bounded to RMS displacement 2*c. Four original Adam
proposals and two incumbent candidates remain. The real NPU still measures
all candidates, and acceptance/commit checks are unchanged. The analytic
surrogate gradient never substitutes for actual NPU measurement.

```bash
kvidence/experiments/kv_guided_search_gemma4_int8/.venv/bin/python \
  kvidence/reconstruction/experiments/image_trust_region/run_spsa_joint.py \
  --reference-run kvidence/reconstruction/experiments/image_trust_region/results/0001_rounds300_seed17_20261020 \
  --observation kvidence/reconstruction/experiments/image_trust_region/input/0001 \
  --resume-run kvidence/reconstruction/experiments/image_trust_region/results/0001_geometry300_seed17_20261021 \
  --output /tmp/new_tanh_search \
  --geometry --surrogate --nonlinear 0.1 --steps 100
```

Resume validates observation and seed, verifies initial KV by real replay and
reports total costs including the preceding run. Startup and 32 repeatability
measurements remain separate. `benchmark_surrogate.py` evaluates saved,
separately measured candidate rows excluded from fitting; duplicate training
inputs are removed from that holdout. `analyze_surrogate.py` summarizes those
predictions and the matched-budget warm-start NPU controls. See
`surrogate_comparison_20261021/REPORT.md` for measured speed and limitations.

## Full 768-direction probe (2026-10-21)

`--geometry --directions 768 --max-queries 45001 --steps 300` probes a complete
orthogonal basis of the 768-dimensional pre-adapter input at each center.
Each full round measures 1,536 signed probes in independent physical batches
of at most 128, followed by the usual 22 update candidates. Queries do not
advance the prefix. The solver still accepts only actual measured KV.

With `--max-queries`, learning rate and perturbation follow used candidate-budget
fraction. At 45,001 total candidates this matches the historical 64-direction,
300-round schedules in budget space. The full version can take 28 full rounds
and one final 677-direction round, spending exactly the same query budget.
The flag preserves the original round-based schedules when omitted. Tiny
surrogates remain restricted to <=64 directions so their training set never
silently exceeds the intended 256 rows.

`analyze_full_directions.py` verifies initial inputs, observations and candidate
budgets and summarizes the saved 64/768-direction runs. Results and reproduction
commands are in `full_directions_20261021/REPORT.md`. Full direction coverage
alone is not a guarantee of local prediction accuracy or global recovery.

## Continue until a measured plateau

`--resume-run PREVIOUS --continue-tail --plateau-rounds 20 --geometry --directions 768`
continues from the previous best measured input, holding its final learning rate
and perturbation RMS. It restores Adam moments, bias-correction step count and
RNG state. For older cold-start geometry runs without checkpoints, it replays
saved measured probes and verifies that regenerated directions match inputs.
New runs save `optimizer_state.npz` and best input/KV after every round.

The stopping rule requires 20 consecutive complete rounds without ANY measured
best-RMSE improvement; any improvement resets the counter. The continuation
counter starts at zero and does not count stagnation in the preceding phase.
This is an empirical fixed-parameter plateau, not a global optimum certificate.
`--steps 300` is a per-process safeguard: reaching it without stagnation is
reported as `iteration_limit`, not plateau. Resuming does not count old query
cost as free, and state reconstruction time is included in cumulative costs.

`analyze_plateau.py` checks the saved stagnation window and summarizes both
initializations in `plateau_20261021/REPORT.md`.

## Full-basis multiscale and broad line-search trial

`run_full_multiscale.py --warm-run PLATEAU_RUN --observation OBS --output NEW --rounds 5`
reuses the same complete 768-direction basis across probe RMS values h/4, h and
4h at each center. For each scale it measures four damped ridge directions at
multipliers 1/64 through 4 (nine values), plus the four previous Adam step
multipliers. The original nominal-scale proposals are included as a subset.

Each round costs 4,608 probes and 121 updates, all measured using batches <=128;
five rounds plus initial replay cost 23,646 candidates. Prior RNG/Adam state is
restored. Each scale computes candidate moments from the same incoming state;
the measured winner's scale supplies the next moment update, or the nominal
scale when no candidate improves. Global measured best includes probe wins.

This is a five-round trial, not an assertion that a new plateau has been found.
`analyze_full_multiscale.py` reports error reductions and the exact measured
winning scale, damping, step factor and input RMS displacement. Saved per-scale
metadata also permit within-center comparisons of original vs expanded step
sets. Results are in `full_multiscale_20261021/REPORT.md`.

## Immediate scale adjustment (current default)

Following the user's correction, `--adaptive` now defaults to
`--plateau-rounds 1 --expansion-trial-rounds 1`: one non-improving round changes
the main probe radius and associated ridge step cap for the next round. No
20-round confirmation is required. A failed enlarged trial immediately tries
the next bounded level, preserving the best actually measured input and the
original small-probe anchor. At the configured cap, another non-improving round
ends this bounded trial; it is not evidence of convergence or a proven plateau.

`--reset-expansion-policy` applies these new policy settings to an existing run
while retaining its best input, RNG and Adam state. This explicit flag is
required to change settings on a saved policy. Source snapshots for the two
immediate-adjustment hardware runs are preserved in each run's `source/`.
Results are summarized by `analyze_adaptive_expansion.py` in
`immediate_expansion_20261021/REPORT.md`.

## Trainable tiny MLP (MindEye-inspired mapping experiment)

MindEye's official `BrainNetwork` uses a learned residual MLP with default width
4096 and four blocks. This experiment borrows the learned nonlinear mapping
idea, with a much smaller 768→128→64→1024 SiLU correction network (172,224 learned
parameters) over a frozen local linear ridge fit. Both branches are centered
at the actual measured input/KV. This is not a reproduction of MindEye's
fMRI→CLIP mapping, projector, losses or diffusion prior.

`benchmark_mlp.py --output NEW_JSON` compares linear, fixed-tanh and trained MLP
on matching saved probe data and separate measured updates. It times CPU
single-thread PyTorch: first 100 steps, later 20-step warm updates, batch 64.
No held-out test errors select weights or stop training.

`run_full_multiscale.py --mlp-test --probe-radius 0.08713840313369006 --rounds 5`
retains the complete 768-direction, three-scale measured search. It samples 42
paired directions per scale (252 training rows) for both auxiliary models.
Sixteen identical starts are optimized by the learned MLP and the matched
linear-only surrogate, then all 32 extra candidates are measured on NPU.
Each round costs 4761 candidates; five rounds plus initial replay cost 23,806.
The small model never replaces real acceptance or the full-direction probes.

`analyze_mlp.py` distinguishes ordinary-search improvements from extra linear
or MLP wins, and reports training/inference latency and limitations under
`mlp_comparison_20261021/REPORT.md`. Per-round MLP weights and source/configuration
hashes are retained with the real queried inputs and KV. Warm training currently
persists within one process; restarting a process builds a new auxiliary MLP
while restoring the main search RNG/Adam checkpoint.
