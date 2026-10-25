# Reconstruction terminal workspace

From the project root, run:

```bash
./reconstruct
```

The workspace uses a pixel-art KVidence wordmark and pink ghost mascot, a compact status header,
a progress bar, an activity feed, and a command prompt. The artwork is drawn with terminal half-block
characters, with a compact version for narrow windows; no image protocol or extra font is required.
The interface, help, status labels, and control messages are in English. Recovered text and experiment
prompts retain their original content.

You can also describe a case in plain language. The local MLX agent asks for missing facts and
records them in case metadata, then invokes registered recovery templates. See [Local cases](CASES.md).

Use `./reconstruct --demo` for a read-only preview. No extra terminal UI packages are required.
Supports macOS Terminal, iTerm, and other curses-compatible terminals. Minimum size: 54 columns
by 16 rows; recommended: 100 columns by 30 rows. The palette separates warm orange branding, a pink ghost, bold cyan commands, soft-gray explanations,
green results, and red errors. Metadata and separators are subdued. The terminal background is preserved.

## Commands

| Command | Action |
|---|---|
| `/help` | Show available commands |
| `/status` | Refresh collection status; the header also refreshes automatically |
| `/pause`, `/resume` | Pause or resume image collection, preserving cooling rules |
| `/profiles` | List image-prefix context profiles |
| `/create NAME FILE` | Create a profile from a messages JSON file |
| `/use NAME` | Switch profiles after collection has paused |
| `/watch PATH` | Follow a recovery output directory or events.jsonl |
| `/run ARGS` | Start `run.py run ARGS`, with live events enabled |
| `/stop` | Interrupt the recovery worker started in this session |
| `/clear`, `/quit` | Clear the activity feed or leave the workspace |

The `/help` and `/case help` references open from the first line in a dedicated help view.
Use Up/Down or PageUp/PageDown to browse, Home/End to jump, and Esc or q to return.

Outside help, use Up/Down for command history, Left/Right and Home/End to edit, Ctrl+U or Ctrl+C to clear input,
and PageUp/PageDown to scroll. Paths containing spaces support shell-style quotes, but commands
are never executed through a shell. Bracketed paste prevents pasted newlines from executing commands.

The collection service remains independent: run `./collectctl serve` in another terminal.
Ghost and the terminal share that service. Existing recovery records can be viewed while the
service is offline. Before launching recovery, `/run` checks that collection has stopped.
Do not resume collection from Ghost while a recovery worker is using the NPU.

## Follow or start recovery

```bash
./reconstruct --watch kvidence/reconstruction/results/my_run
```

Inside the workspace:

```text
/pause
/status
/run --observation /path/to/observation --output kvidence/reconstruction/results/new_run --method guided
```

Wait for `Paused` before `/run`. Recovery uses the existing scientific Python environment and
requires a new output directory. Worker logs are saved beside it in `.reconstruct-logs/`.
The feed displays accepted token positions, candidate counts, recovery status, and recovered text.
Live text decoding supports Gemma JSON tokenizers; other formats still show token IDs and progress.
Accepted candidates are not independently verified identities; use the existing evaluation workflow.

`/pause` controls collection; `/stop` interrupts recovery. Recovery does not guarantee resumability.
Leaving the workspace does not stop independent collection or recovery workers. Reopen with `/watch`
to follow an existing run; a new session does not take ownership of the previous session's `/stop`.

Validation: interactive PTY rendering and input, historical recovery event playback, and seven unit
tests covering paste protection, read-only preview, collection preflight, complete event records,
worker detachment, Unicode text width, and English layout at multiple window sizes.

Text experiment results default to `results/text_experiments/` (relative to reconstruction).
`/results` or `/experiment_demo` exports and displays existing plaintext_chat results.
Add `encrypted_chat` to select ciphertext input recovery, and an optional sample ID
(e.g. `/results plaintext_chat 0076`) to select one sample. Without an ID the scope
is the whole selected dataset. These commands are read-only archive exports; they
never launch NPU recovery. Results are matched by model and capture fingerprints.

Each timestamped export includes `experiment.json`, `summary.json`, `summary.csv`,
and `details/<sample>/<run>/candidates.csv`. Summaries record online time, setup
costs, query counts, accepted tokens/s, and independent accuracy when available.
Candidate exports preserve recorded token IDs, distances, stages and acceptance.
Per-position timing repeated on candidate rows must not be summed. Full original
trace/event/protocol paths remain in the summary. Distance metrics and spaces must
be checked before comparing runs. Acceptance alone does not establish correctness.
New case runs also automatically save a summary and candidate CSVs in their run
folder, including partial results on failure/interruption.

Launch selected methods with:

```
/experiment_demo plaintext_chat 0076 --methods guided
/experiment_demo plaintext_chat 0076 --methods guided,collision_plus,exhaustive --max-queries 5000
/experiment_demo encrypted_chat --methods guided
```

Providing `--methods` starts a new experiment; `--run` selects all three default
methods. Omitting a sample runs the entire selected dataset. Collection must be
paused and cooling complete; the supervisor checks again between samples and
methods. `/stop` interrupts the process group. Each launch uses a unique default
results folder with `samples/<ID>/<method>/`, summary JSON/CSV and detailed traces.
The query budget is passed to each method's existing recovery engine.
Official `collision_plus_cpa` requires independent calibration and remains available
through `/case`, rather than this dataset shortcut. Encrypted experiments target
the entire supplied user prompt (including ciphertext), not AES decryption.
Benchmark tokenization supplies target boundaries; target token labels are written
to a separate private evaluation directory and passed only to post-run evaluation.
Without `--methods` or `--run`, both commands still export archived measurements.

For the 2026-10-25 HierText overnight acquisition, `/watch hiertext` (alias
`/watch overnight`) switches the collection header to local overnight status and
live per-batch counts, refreshed every three seconds. `/watch live` returns to
the original collection service. This changes the displayed progress only and
does not start, stop or switch the underlying collection job.
