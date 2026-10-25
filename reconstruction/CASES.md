# Local case assistant

Start the workspace with `./reconstruct` and describe the task. No slash commands are needed for
case creation, metadata answers, method selection, or recovery requests. Chinese and English input
are supported; the workspace and follow-up questions are in English.

```text
You: Create case1
Agent: Where is the canonical observation directory containing metadata.json and observed.npz?
You: The observation is at /path/to/observation
Agent: [inspects the capture, identifies a registered model when its reference SHA256 matches]
You: Use ridge, Collision+ and exhaustive search
You: Start recovery
```

If the producing model cannot be identified, the assistant asks for it and records `unknown` when
the user does not know. It does not default to Gemma. Recovery is blocked until required information
is supplied. A canonical observation is the existing framework's prepared recovery input; this first
version does not automatically import arbitrary raw memory dumps. Use the existing `prepare` workflow
to produce an observation first. Unsupported models need a registered configuration and adapter.

## Fixed commands

The same metadata is also available through deterministic commands. These commands do not load or
call the language model. Use either interface within a case; both record field provenance and apply
the same schema and evidence checks.

Inside the workspace:

```text
/case new case1
/case set model_name gemma4-e2b
/case set observation /path/to/observation
/case set methods guided,collision_plus,exhaustive
/case set notes "Recovered text prefix; image boundary not yet resolved"
/case check
/case show
/case plan
/case run
```

`/case open NAME` selects an existing case; `/case list` lists saved cases. `/case import FILE`
imports a JSON object of editable fields. Optional case names can be supplied explicitly, for example
`/case set case1 max_queries 20000`. Use `unknown` for an unidentified model and `null` to clear
optional path or budget fields. Evidence-derived layer, prefix and quantization facts come from the
validated observation and are not changed by metadata editing commands.

For shell scripts, always specify the case name:

```bash
./casectl new case1
./casectl set case1 model_name gemma4-e2b
./casectl set case1 observation /path/to/observation
./casectl set case1 methods guided,collision_plus,exhaustive
./casectl check case1
./casectl plan case1
./casectl run case1
```

Creating, setting, checking, or previewing a case does not start recovery. `run` is the explicit
execution request and still checks collection state and phone cooling.

## Local model and execution

- Model: `mlx-community/Qwen3-0.6B-4bit`, stored in `assets/local-agent/qwen3-0.6b-4bit`.
- Runtime: the existing Apple Silicon MLX environment. The model loads on the first natural-language
  message and remains resident until the workspace closes. Runtime inference uses local files with
  Hugging Face offline mode enabled. No case messages are sent to a remote model.
- The model extracts intent and metadata. The application validates these against a fixed schema and
  registered methods. It never executes shell text returned by the model.
- Common model names, method names and pasted paths are normalized by deterministic templates.
  A method selection or a negative request does not authorize a recovery run.
- Collection must be paused before recovery. Say `pause collection`, wait for paused status, then
  say `start recovery`. Starting a case does not interrupt the ongoing 10,000-image collection.

## Methods

Default case methods, executed sequentially against the same observation:

1. `guided`: existing online ridge-guided candidate search, with real KV verification.
2. `collision_plus`: existing model-probability ordering with the shared RMSE acceptance rule.
   This is an adaptation, not the original independently calibrated CPA rule.
3. `exhaustive`: ascending original token IDs, without probability ranking or a ridge surrogate.
   Enumeration stops at the first accepting batch; it selects that batch's smallest passing distance.
   If no candidate passes, it visits the entire allowed vocabulary unless an explicit budget stops it.
   Threshold acceptance does not prove uniqueness or token identity.

`collision_plus_cpa` is also available when explicitly selected. It requires an independently produced
calibration JSON and retains the framework's calibrated K/V thresholds. Its threshold units differ
from shared-RMSE methods. The existing CLI validates calibration and runtime compatibility.

Explicit query or position limits are recorded in the case and passed to every selected method.
The existing framework performs independent identity evaluation; the assistant does not read private
evaluation labels or convert acceptance counts into correctness claims.

## Files

```text
cases/case1/
  metadata.json             # facts, unknown fields, provenance, readiness and run history
  conversation.jsonl        # user answers, agent replies and extracted model plans
  logs/                    # case supervisor logs
  runs/<run-id>/
    plan.json              # frozen metadata, evidence fingerprints, argv and method outcomes
    guided/                # existing protocol, events, trace and result artifacts
    collision_plus/
    exhaustive/
```

Cases are excluded from Git by default. Metadata and observations are never overwritten when a new
case uses an existing name. To open an existing case, say `open case1`; do not recreate it.
`/stop` interrupts the current case supervisor and its active recovery worker. Closing the workspace
keeps a launched recovery supervisor running. The local language model itself is closed on exit.

## Validation

A real local MLX session was tested in Chinese: create a case, record an unknown model, select the
three methods, supply a canonical observation, resolve readiness and request recovery. No phone
recovery was launched during that test. Details are in `results/local_case_agent_validation_20261024.json`.
CPU tests verify case isolation, evidence checksums, registered templates, missing calibration,
conversation persistence, unsupported actions, explicit run intent, and exhaustive ordering/budgets.
