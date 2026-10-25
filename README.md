# KVidence

Research code for inspecting and reconstructing text and image representations from model KV caches, with a terminal workspace, local case assistant, real-phone acquisition, registered search methods, and experiment reports.

## Repository layout

- `reconstruction/`: terminal, case metadata, recovery engine, Android native worker, tests, configurations, image acquisition and diagnostics.
- `experiments/`: supporting experiment code and available reports.
- `references/`: third-party reference implementations and their original license files.
- `reconstruct`, `casectl`, `collectctl`: terminal entry points.

This is a source-and-report snapshot of the local research workspace. Reports describe their own historical revisions and conditions; their results are not guarantees for a new installation. No datasets, captured KV arrays, model weights, local case records, APKs or compiled native libraries are included. Some report links refer to artifacts intentionally excluded from this source repository.

## Workspace setup

The existing code expects the Android app and KVidence repositories as siblings:

```text
workspace/
  annoymous/    # https://github.com/slittlefingers/annoymous
  kvidence/    # this repository
```

Clone both into the same workspace. Set up the Android app/runtime using its README. Models, tokenizer, embedding tables, runtime section metadata and experiment-specific assets must be supplied separately. Configurations are under `reconstruction/configs/`; edit device serial placeholders, local paths and cloud host placeholders for your environment.

The terminal wrappers currently expect a UI environment at `reconstruction/experiments/fixed_prefix_image_inverse/.venv/` and recovery code expects another environment at `experiments/kv_guided_search_gemma4_int8/.venv/`. Alternatively invoke the Python entry points directly with your configured interpreter. The NumPy/SciPy base requirements are in `reconstruction/requirements.txt`; optional features additionally require tokenizers, PyTorch, MLX/MLX-LM (local Mac agent), or sentencepiece as documented by each component. The offline local agent also requires separately downloaded model weights.

See [terminal commands](reconstruction/TERMINAL.md), [case workflow](reconstruction/CASES.md), [native transport](reconstruction/TRANSPORT.md), and [reconstruction research notes](reconstruction/README.md).

## Data and licensing

Dataset publication is separate: https://github.com/slittlefingers/gemma4-image-prefix-dataset

Follow the original dataset/model licenses. See [third-party notices](reconstruction/THIRD_PARTY_NOTICES.md) and retained upstream notices/licenses. This snapshot does not grant a new license for third-party components or redistribute their model binaries.
