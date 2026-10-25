# Two-stage image reconstruction: next-round instructions

Date: 2026-10-25 (Australia/Melbourne)

## Objective

Diagnose the weak run01 reconstructions before changing the collection protocol or scaling the model. Prioritize faithful image and text recovery over visual plausibility. This document is an experiment specification, not an executable training script.

## What the existing dataset contains

Each sample contains image-position tensors:

- `kv_codes`: INT8 [N, 1024], K followed by V, 512 dimensions each.
- `pre_adapter`: FP32 [N, 768], image features before the adapter.
- `embedding`: FP32 [N, 1536], image embeddings after the adapter.
- `pixels`: UINT8 [H, W, 3], the image actually processed by the native pipeline.
- `grid`: [grid_height, grid_width]; H = 48 * grid_height, W = 48 * grid_width.
- `positions`: image-token positions following the fixed 991-token text prefix.

The samples do NOT contain the full preceding text-token embedding or KV sequences. Prefix text and metadata are stored separately in `prefix.txt` and `prefix.json`; these are not prefix embedding tensors. Inspect those files to determine the exact public information available; do not assume undocumented tensor fields exist.

The current recovery input is image-position KV, already conditioned on the fixed prefix. It does not explicitly concatenate the 991 text embeddings. Removing a prefix from the model input therefore is not an available ablation of run01. A no-prefix capture is a different forward experiment requiring new KV acquisition.

Since the prefix is fixed across samples, the inverse model can potentially learn a mapping conditioned on that constant. This is not evidence that the prefix has no effect, or that the inverse is uniquely identifiable. Explicit prefix conditioning is a later ablation, particularly for variable-prefix datasets.

## Preserve run01 and evaluation boundaries

1. Do not overwrite run01, its checkpoints, statistics, figures or split lists.
2. Use a new directory, e.g. `experiments/two_stage/run02_diagnostics/`.
3. Reuse the existing train/validation/test grouping and train-only normalization statistics. Record hashes of these artifacts.
4. Select changes and checkpoints using training/validation data only. The run01 test set has already been examined; label later comparisons on it as iterative exploratory evaluation. Arrange a new final holdout before making final generalization claims.
5. The private target photo, its variants and labels must not enter training or checkpoint selection. The report says it is not present on this host: record that limitation, do not silently substitute another image.
6. Do not modify, restart or access the phone collector as part of these offline diagnostics.

## Step 1: audit representation and spatial layout

Check the actual implementation, not just the report:

- Features and pixels are paired by sample ID and verified checksums.
- Standardization/de-standardization use the intended axes and training statistics; check the numerical round trip.
- Array loading, batch collation and reshaping preserve token order, height/width orientation and RGB channel order.
- Pixel normalization and output activation agree with the chosen loss; check clipping and image-export conversion separately.
- Padding is excluded from losses and metrics if introduced.
- Training loss uses correctly scaled gradient accumulation, including the last incomplete accumulation group.
- Inspect per-channel feature scale, variance and outliers. Report both raw and standardized feature error.

Important: N = grid_height * grid_width proves compatible dimensions, not correct spatial token order. Independently trace the native image pipeline's patch/merge order where source or instrumentation is available. If unavailable remotely, record the ordering as unverified. Do not choose a permutation using the test set.

Save `audit.json` with checks, evidence, unresolved assumptions and failures.

## Step 2: small-subset memorization test of the decoder

Purpose: test whether the current implementation and capacity can fit known training examples. This does not establish test-set generalization or information-theoretic recoverability.

- Start with one training image, then a deterministic set of 16 images, including natural images, art and text images.
- Save the selected training IDs before fitting. Use no validation or test images.
- Input TRUE `pre_adapter`; do not involve the KV recovery model.
- Disable random augmentation and stochastic regularization for this diagnostic.
- Run the existing decoder first, with fixed seed and a finite predeclared budget (e.g. at most 5,000 optimizer updates).
- Report loss components, gradient norms, training PSNR/SSIM and unchanged native-resolution crops at fixed intervals.
- Save best training-loss checkpoint for this memorization diagnostic only. Do not use this selection rule for the production model.

Interpretation:

- Failure even on one image: inspect optimization, implementation, loss scaling, output range and capacity before a larger training run.
- Good one-image fitting but poor 16-image fitting: investigate shared capacity, feature informativeness and optimization. This result alone does not identify the cause.
- Good subset fitting but poor held-out reconstruction: investigate generalization, data diversity, capacity and training duration.

There is no universal PSNR threshold proving a correct implementation. Report actual curves and compare the original and diagnostic outputs.

## Step 3: improve true-feature image decoding

Only after the audit and memorization test:

A. Continue the run01 decoder with its saved optimizer/scheduler state if present. Both run01 stages selected epoch 30, so a plateau was not established. If optimizer state is absent, label continuation as a warm start and record the new schedule.

B. Test one controlled architecture change, such as preserving a 48x48 patch output path with neighboring-grid residual refinement, or increasing convolutional decoder capacity. Keep data, losses and selection protocol fixed for the architecture comparison. Parameter count alone is not a measure of high-resolution decoding capacity.

Use true features initially. Keep RGB L1 plus the existing gradient loss as the baseline. Any text-region weighting requires independent training annotations mapped to the actual processed pixel coordinates; do not infer ground truth from generated images or the private target.

Evaluate on validation images by source. Keep a fixed validation crop panel. Save native-resolution images without sharpening or generative postprocessing.

Optional representation comparison: train matched-budget decoders from `pre_adapter` (768D) and post-adapter `embedding` (1536D), separately. This tests the practical decoding target. Do not describe the 768D tensor as lossless or inherently optimal without evidence.

## Step 4: connect KV recovery to the frozen decoder

Run01 trained KV recovery only against raw feature MSE. Smaller feature error did not always produce better images. Test decoder-aware training explicitly.

For image-position KV X, feature predictor F and frozen decoder D:

    U_hat = F(X)
    I_hat = D(U_hat)
    L = lambda_feature * L_feature(U_hat, U)
        + lambda_pixel * L1(I_hat, I)
        + lambda_gradient * L1(gradient(I_hat), gradient(I))

Requirements:

- Feed PREDICTED features into the decoder during this training.
- Freeze decoder parameters, but keep its forward pass differentiable with respect to predicted features. Do not wrap it in `no_grad()`.
- Use training-only statistics. Compare raw feature MSE and standardized feature MSE as separately named ablations; large-scale dimensions may otherwise dominate.
- Log each unweighted and weighted loss component; choose weights on validation data and record them. Do not assume equal numeric weights imply balanced gradients.
- Retain feature-only run01 as a baseline.
- Use the SAME decoder checkpoint for ridge-versus-Transformer comparisons; report changed-decoder comparisons separately.
- Start with a small finite pilot and report throughput, peak allocated GPU memory and validation change before a full run.
- Consider low-learning-rate joint fine-tuning only after this frozen-decoder experiment.

No genuine KV-consistency training loss is available without the matching forward model or a separately validated differentiable approximation. Do not relabel a feature loss or learned surrogate error as real phone KV consistency.

## Resource and runtime measurements

Confirm actual GPU model and available memory (`nvidia-smi`), physical batch size, gradient accumulation, precision and CPU worker count. A full A40 has 48 GB; a reported 24 GB allocation needs its actual virtualization/quota context recorded.

Run01 decoder used physical batch 1 and accumulation 16. That does not have the same throughput as physical batch 16. Test safe physical batches 1, 2 and 4 with identical images and precision; stop increasing on memory pressure. Report actual images/s rather than assuming a speedup.

Measure data loading/decompression, host-to-device transfer, forward/backward, optimizer and validation time separately, with appropriate CUDA synchronization or events. Avoid interpreting asynchronous CPU launch time as GPU compute time. Read only required NPZ members per stage; measure before repacking or caching the dataset. Keep diagnostic profiling separate from normal throughput measurements.

## Evaluation and delivery

Save:

- `config.json`: methods, losses, weights, optimizer, schedule, seeds, precision, input tensors and budgets.
- `audit.json`, `subset_ids.json`, split/statistics hashes and code version/hash.
- `metrics.jsonl`: training and validation metrics, per-source metrics, timing and memory.
- Checkpoints with model and optimizer/scheduler state where applicable.
- Fixed validation original/reconstruction panels and native-resolution crops.
- `summary.json` and a concise report distinguishing measured results, hypotheses and unexecuted work.

Required reported metrics:

- Feature raw RMSE, standardized RMSE and cosine, plus the mean-feature baseline.
- Image PSNR/SSIM/MAE, per-image and per-source.
- CER and exact-string match ONLY where independent transcription is available. Otherwise mark unmeasured; OCR confidence is not correctness.
- Real recaptured K/V error ONLY after the same phone model, prefix, layer, positions and image-processing pipeline have been executed and checked. Mark unmeasured remotely.

For phone evaluation later, first replay the original processed image to quantify reprocessing drift. Re-encode each reconstructed image through the complete image pipeline; never inject true image embeddings. Report K and V separately in dequantized units. Grid mismatches are pipeline mismatches, not valid row-wise distance comparisons. Low KV error alone does not prove unique source-image identity.

## Immediate priority

Perform Steps 1 and 2 first and report findings before committing to another full training cycle. Do not change or recollect the prefix merely because run01 images are blurry. The true-feature decoder already exhibits a bottleneck independent of KV recovery; isolate it first.
