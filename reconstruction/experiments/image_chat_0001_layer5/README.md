# Real Pixel image-chat pilot: sample 0001

**Status (2026-10-20): not recovered.** The initial MLX proposal path below is
retained as a failed cross-backend diagnostic, not a valid phone-native recovery
result. After the user's correction, `device_adapter_test.py` calls the actual
phone `vision_adapter_280` and NPU decoder. Its controls pass, but the separate
finite-difference search does not recover even the first image position.
`synthetic_device_control.py` also fails to converge on a known reachable native
target, so this is not evidence of intrinsic unrecoverability. See [REPORT.md](REPORT.md).

Follow-up: phone-only SPSA + Adam improves the one-position synthetic control
to raw KV L2 5.3121 / 4.9037 (two initializations of one target). The user then
revised **image** acceptance to raw concatenated K+V L2 < 5, not 5 LSB.
`run_real_spsa.py` tests the real image causally with this revised criterion.
See [the real-image results](real_spsa_l2_5/REPORT.md) and
[the protocol](SPSA_PROTOCOL.md). Partial position matches do not establish
whole-image content recovery. The text-search threshold is unchanged.

For the user's content check, the previous full-image latent proposals were
replayed through the **phone** adapter and decoder. One description correctly
identifies soccer players, a ball, grass and trees, but neither full proposal
passes the KV threshold. See [semantic comparison](full_proposal_phone_content/REPORT.md).
This result is separate from the phone-only SPSA run, interrupted after five
accepted positions to prioritize the content check.

## Question

From the stored INT8 layer-5 KV of `image_chat_100/0001`, can an image
representation be reconstructed, and can that representation support recovery
of the following unknown text? Selection is the first sample ID, before seeing
recovery results. This is one sample with two optimization initializations.

## Data separation

`prepare.py` reads the immutable source dataset and writes two separate areas:

- `input/`: layer-5 observations, public template, media layout, and the public
  model's constant end-of-image embedding.
- `private_eval/`: original text IDs, original image path, and annotation path.

Recovery scripts enable the framework's read guard, which denies reads from
`datasets` and `private_eval`. The target KV is a loss target only; it is never
inserted into the decoder cache. Text recovery uses the complete original
262,144-entry vocabulary.

Layout is 5 public prefix positions, 260 image positions, the constant image-end
position, one separator, and 21 unknown text positions starting at 267. The
image size and public preprocessing rule supply the layout. Preparation also
checks **only public template** positions against first-layer V. No secret
token or image position is recovered from that layer. Consequently this pilot
uses supplied layout; it does not establish automatic boundary discovery when
only layer 5 is available.

## Initial cross-backend diagnostic (not the corrected recovery path)

1. `verify_worker.py`: compare ordinary-token commit with continuous embedding
   commit using the same known token and PLE row. Check exact KV/logit equality
   and public prefix agreement with the capture.
2. `fit_image.py`: optimize all 260 pre-adapter vectors jointly using the
   existing MLX differentiable image map. Model and adapter weights stay fixed.
   Seeds 17/29, 3,000 Adam steps each, initial learning rate 0.08. This is a
   cross-backend proposal: MLX arithmetic is not the Pixel INT8 graph.
3. `pixel_recover.py`: replay each recovered representation through the actual
   NPU decoder with media PLE row zero. Select by total measured media KV squared
   error, before opening private labels. Replay produces every layer's cache
   through ordinary forward computation.
4. Recover subsequent text using the unchanged guided method: 50 probability
   seeds and 150 random seeds, then batches of 20 proposals. Accept only real
   NPU K+V L2 < 4 and verify the committed token. No 600-query cap.
5. Probe image content with a fixed description question on the recovered
   representation. This tests retained semantic content, **not original-pixel
   reconstruction**.
6. Only after recovery finishes, evaluate exact text identity and compare the
   description to the original image/annotations.

Media acceptance requires **every** media position's raw concatenated K+V L2
to be below 4. A later-text diagnostic is still run if that media gate fails,
but it must not be reported as a fully accepted end-to-end pipeline. A distance
threshold alone does not prove token identity or uniqueness.

## Implementation scope

The added `commit_embeddings`, `vision_project` and `vision_query` operations
are in the instrumentation test worker. `vision_project` runs the actual phone
model's CPU adapter; `vision_query` feeds its outputs into isolated NPU candidate
slots, sharing one already reconstructed prefix. These operations take proposed
latent/input vectors only, never observed KV.

The `commit_embeddings` operation is in the instrumentation test worker;
it accepts continuous input vectors, not observed KV. It does not replace the
app's installed inference library or change the existing text search algorithm.
The Python experiment wrapper handles this known image layout explicitly. It
does not claim generic automatic NPU image/audio routing.

Model config has no bidirectional image attention. Replay uses the causal
decoder mask, original positions and RoPE. Known-token equivalence validates
the new input path, but is not by itself a proof that original vision prefill
and single-position decoder arithmetic are identical.

Build the worker with `bash kvidence/reconstruction/native/build.sh`, then build
the Android test APK with `:app:assembleDebugAndroidTest`. Use the existing
Gemma multimodal MLX environment for fitting, and the existing Gemma guided
search environment for Pixel recovery. `prepare.py` and recovery outputs refuse
to overwrite completed runs. See `REPORT.md` for measured results when complete.
