# Third-party attribution

`native/image_collection.h` follows the Gemma4 preprocessing algorithms in
LiteRT-LM `support/preprocessor/stb_image_preprocessor.cc` and
`image_preprocessor_utils.cc` (Copyright 2025 The ODML Authors, Apache License
2.0; source and license retained in the local LiteRT-LM checkout). The native
build uses the same external stb headers for JPEG decoding and resizing. stb
is dual licensed under the MIT license or public domain; this use selects the
public-domain option provided in the headers.

The Collision+ threshold procedure in `kvrecon/cpa.py` adapts the branching and optimization objective from `attack/collision.py` in [SiO-2/kvcloak](https://github.com/SiO-2/kvcloak), commit `6b40f36edb2f337557543e7e60b10022308883d4` (Apache License 2.0). The original [license](../references/kvcloak/LICENSE) and source checkout are retained. The adapter separates calibration collection from fitting, validates inputs, exports provenance, and rejects an undefined zero-other-variance optimization case. This is modified implementation code, not an unmodified copy of the upstream attack runner.

`tests/test_cpa.py` extracts only the upstream pure threshold definitions at test time to check parity. The calibration text under `inputs/cpa` is the public Bitter Lesson excerpt included in upstream `src/config.py` and identified in the paper's Appendix B; it is used as known calibration input, not private evaluation data.
