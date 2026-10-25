# Fixed-prefix real-NPU image reconstruction

**Latest instruction:** resume collection to exactly 10000 training images. The previously published 1000-image dataset is retained and reused. Active jobs: prepare_mixed_sources_resume.log and resume_to_10000.log; collection output: data/full_fixed991 and collection_10000.log. The important phone target remains outside training. Previous stop-at-1000 notes below describe the completed earlier stage.

Dataset construction started 2026-10-23. Latest scope: finish exactly 1000 training-source images, stop, and publish. The 10000-image expansion and extra downloads were cancelled. The target photo is explicitly authorized for public release as a separate important test case. See DATASET_BUILD.md for protocol details.

The frozen constructed conversation has 991 tokens before the image, including 511 tokens in the preceding complete user/assistant exchange. The suffix question is “这个图片的 password 是什么？”. Exact text and token IDs are saved in protocol/.

Primary evaluation target: the separately retained latest phone whiteboard photograph. It and all derivatives are excluded from training and checkpoint selection. private_target/capture_fixed991/observed.npz contains 266 observed image-position KV rows. Original pixels and true features are separate evaluation-only controls and will be publicly released in a separate important_test/whiteboard directory under the user’s explicit authorization.

Planned training sources: 3000 HierText real-text images + 5000 COCO train2017 natural images + 2000 Cleveland Museum of Art CC0 paintings/drawings. HierText source preparation is complete. Other source acquisition runs concurrently. Official HierText validation/test images provide 300+300 auxiliary samples for model selection and diagnostics; the private phone image remains the central reconstruction case.

Smoke acquisition passed for 4 images. Snapshot restore and independently recomputed fixed-prefix replay are bitwise exact on the control image, at 0 LSB. Native resized RGB pixel targets, raw INT8 KV, FP32 768/1536 feature labels and spatial grids are saved per image.

The active collector continues saved images to exactly 1000 HierText training-source captures, with no auxiliary validation/test capture. publish_1000.py waits for completion, validates all images, packages two training shards and a metadata/important-test archive, uploads to the existing GitHub repository, checks server-side SHA256 values, and publishes the Release. Completion is build_status.json stage=complete_uploaded_1000. Current logs: collection_1000.log and publish_1000.log.

Model training has not begun. Public upload after the 1000-image capture is explicitly authorized and automated. The private target is not a training input and its text is not supplied to reconstruction models.

**Cooling policy:** every 500 newly captured images, close the phone acquisition worker and rest for 300 seconds. Check Android thermal status during capture; severe status causes an early pause, with resumption only after cooling. Active controller: collect_500_batches.py; logs: collect_500_batches.log and collection_10000_batched.log.
