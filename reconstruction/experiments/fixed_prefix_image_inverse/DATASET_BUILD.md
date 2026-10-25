# Fixed-prefix dataset build (2026-10-23)

**Latest scope:** user resumed collection to exactly 10000 training-source images. Reuse all 1000 verified captures from the published baseline; acquire 9000 more under the identical frozen prefix and native protocol. No auxiliary validation/test capture is scheduled. Stage is recorded in build_status.json; per-image progress is data/full_fixed991/progress.json. A fresh-worker control checks that the reused baseline remains numerically identical. The earlier stop-and-upload stage below was completed before this resumption.

**Updated scope:** user requested exactly 1000 training-source captures, then stop and upload. The earlier 10000-image expansion, additional downloads, and 1200-image pilot controller were stopped. Active collector: collection_1000.log. Active publication workflow: publish_1000.py / publish_1000.log. The user also explicitly authorized public upload of the target photo as the important independent test object. It remains excluded from training and model selection. The original broader plan below is historical, not an active job.

Purpose: reconstruct the privately retained phone whiteboard photo from real Gemma 4 E2B NPU layer-index-4 INT8 image-position K/V under a known fixed conversation. The target is excluded from all training and checkpoint selection. Target source pixels, true features and observations remain under private_target/ and are not authorized for public upload.

## Frozen conversation

- The fixed constructed conversation prefix has 991 tokens. Its preceding complete user/assistant exchange is 511 tokens; the second user message supplies the remaining pre-image text. This is a constructed experimental conversation, not an extracted victim chat.
- Exact text and IDs: protocol/prefix.txt and protocol/prefix.json. Tokenizer SHA256 is recorded.
- No system prompt. Native image-start token is 255999; exported HF placeholder 258880 is not directly used as the runtime image-start marker.
- The image is followed by “这个图片的 password 是什么？”. Training collection stops after image K/V; the suffix is recorded in metadata, and cannot affect earlier image positions under the causal mask. The separate target capture actually executes EOI and the suffix.

## Sources and counts

- 3000 HierText official training images: 900 contain qualifying visible handwriting. Auxiliary validation/test: 300 images each from their official splits. Geometric legibility filter and JPEG sources are documented in kvidence/datasets/hiertext/.
- 5000 COCO train2017 photographs, deterministic seed 20261023. Original per-image license metadata is retained.
- 2000 Cleveland Museum of Art paintings/drawings designated CC0 with official web JPEGs, deterministic seed 20261024. Official source: https://github.com/ClevelandMuseumArt/openaccess . The selected web JPEGs and object IDs are recorded, rather than assuming all museum objects permit reuse.
- Combined source manifest (when ready): kvidence/datasets/fixed_prefix_mixed_10000/manifest.json. Final training count 10000, auxiliary validation/test 300 each. All auxiliary validation/test images are from HierText; this is not a domain-balanced benchmark.
- Source image IDs and byte hashes must be unique. This does not certify absence of all visually similar or differently encoded duplicates. The private target's exact file hash is excluded and it is not used for source selection.

## Acquisition

Real phone CPU vision encoder and adapter; real phone NPU decoder. Sequential causal prefill_128 for image rows. The public text prefix is initially computed with ordinary token decode and retained as an in-process cache snapshot. Each new image restores that prefix, including all layer caches. Fixed prefix re-use is validated against a separately recomputed full prefix.

Per-image NPZ: kv_codes INT8 [N,1024], embedding FP32 [N,1536], pre_adapter FP32 [N,768], native resized pixels UINT8 [48H,48W,3], grid [H,W], absolute positions [N]. RGB labels are recovered losslessly from native STB /255 input patches. K/V quantization and model identity are stored separately. Candidate-query paths or alternate prefill boundaries are not certified by this control.

Smoke acquisition: data/smoke_fixed991 (4 images), validation.json passed. First image snapshot replay and independent fresh-prefix replay are bitwise exact (0 LSB). Target observations: private_target/capture_fixed991/observed.npz; true features and pixels are separate evaluation-only material.

## Running build

run_dataset_build.py first acquires data/pilot_fixed991 (1000 train / 100 validation / 100 test), runs validate_fixed.py, then waits for combined sources and builds data/full_fixed991 (10000 / 300 / 300). Completed pilot files are hard-linked into the full dataset so they are not recaptured or duplicated on disk. Failures are recorded in build_status.json and are not silently treated as completed data.

Progress: build_status.json, data/*/progress.json.
Logs: build.log, pilot_collection.log, full_collection.log, prepare_mixed_sources.log.
Completion requires build_status.json stage=complete and a passing data/full_fixed991/validation.json. Model training is a subsequent step, not part of this acquisition run. No public upload is part of this build.
