# Native-selected internal context validation

This revision includes `--triangle-internal-context`, which adds compiled
navigation to real Rust Spine chat. It remains opt-in. The first historical
chat battery ran with the flag off and an empty model navigation forest; its
native 14/14, triangle-only 11/14 and assisted 3/3 scores remain historical
results of that path. The second chat battery enabled the flag on November 3,
2026. Its [aggregate evidence](../evidence/internal-context-20261103.json) records
completed runs, retained failures, execution checks and artifact identities.
Default chat still installs an empty navigator forest.

## Mechanism and acceptance boundary

Native Rust code retains its existing query ranking, goal selection, depth-eight,
fanout-three, 16-coordinate and 1,024-estimated-token rehydration budgets. It keeps
the existing maximum of 12 source excerpts, each truncated after 1,000 characters,
and at most two recent events per selected node.

For each selected root, Rust exports its complete original topology, including
the real apex and both branches of each triangle. Each bank must fit 64 combined
node/triangle records and depth eight. Oversized closures fail explicitly. The
provider does not scan the source store, invent leaf substitutes, export the
whole durable forest or silently fall back to native prompt context.

The topology contains no source bodies. Original selected source rows travel in
a separate system-message sidecar, along with retained native context. Before
each completion the provider reinstalls the banks from that request and runs
commands through the checkpoint's five compiled layers. Rust independently
checks exact integer traces, installed address order, forest identity and
execution evidence. Only actual emitted apex IDs authorize the original source
rows. All selected rows must pass before the provider inserts their excerpts
into the language prompt.

The main turn's sidecar persists through tool rounds, repair/finish requests and
harness checkpoints. Auxiliary child requests refresh native context from their
task and recent tool output; they do not inherit the main turn's retained goal.
Onboarding retains its no-recall policy and sends an explicit empty packet.
An empty selection is labeled `no_memory`, not successful memory navigation.

This is an internal navigation gate over host-selected evidence. The language
model sees original excerpts after the gate; the excerpts are not encoded into
the checkpoint's weights. Native host code still performs triangle construction,
semantic ranking, compaction and source storage. Exact navigation does not prove
answer correctness, source entailment or recovery of omitted evidence.

## Actual-checkpoint causal probe

On November 3, 2026, the unchanged merged checkpoint on the Intel Arc Pro B70
ran a native bank with 20 records, including one triangle. Both left and right
branch traces exactly matched their expected traces. Clearing only the resident
record bank changed a valid open into `INVALID`; restoring the original bank
restored the exact original trace. The record bank's restored digest matched its
pre-ablation digest, and no checkpoint weights were written.

The runtime identified the executor as `merged_checkpoint_prefix` with five
prefix layers, recorded advancing command tick counts and measured navigation
state digests. This is evidence that real checkpoint navigation depends on the
resident bank. It is a standalone causal probe and does not establish a passing
full-agent language or tool run.

## Full-agent validation

All five corrected conditions completed with `run_completed=true` using the
unchanged merged checkpoint. Every completed model round passed the source gate
and checkpoint-execution checks; language generation preserved navigation state.
All synthetic source hearts stayed unchanged.

| Condition | Behavioral score | Model rounds | Bank executions | Triangle-record installations | Branch commands |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native mixed memory | 14/14 | 31 | 93 | 183 | 79 |
| Triangle-only automatic memory | 12/14 | 27 | 81 | 76 | 59 |
| Triangle-only boundaries | 1/2 | 5 | 15 | 4 | 4 |
| Explicit recall of boundaries | 2/2 | 4 | 12 | 4 | 4 |
| Explicit recall of literal | 1/1 | 3 | 9 | 0 | 0 |
| Total execution evidence | — | 70 | 210 | 267 | 146 |

The table counts banks associated with completed model rounds. Installations
include repeated loads of selected native topology, not unique triangles. Two
interrupted requests executed six additional banks, with 49 triangle-record
installations and 15 branch commands, before interruption; those are excluded
from the table. All raw execution records therefore total 216 banks, 316
triangle-record installations and 161 branch commands.

An audit corrected the initial parser's omission of 24 bank JSON records
prefixed by the terminal's `you> ` prompt, and separated interrupted execution
from completed rounds. The correction changes execution accounting, not
inference, runtime behavior or behavioral scores. Original reports and logs
remain retained. The literal recall probe used leaf-only banks: its navigation
gate passed, but it is not a triangle-branch test. The other four conditions
executed real triangle branches. Mixed mode also supplies automatic hybrid
recall; only the native triangle excerpt rows use the compiled source gate.
Its answer score does not isolate the gated excerpts as the reason for success.

The two triangle-only sequence failures were a fabricated literal/citation and
a file write to an incorrect absolute path. The host denied the write, and the
required file was absent. The other 12 assertions passed, including real file
read, fact acknowledgement, recall after a new process, operator interruption
and subsequent inference. The boundary run recovered the older merged event but
failed the long-source-tail value. Successful navigation does not recover text
omitted by the native source budget.

Explicitly requested recall recovered both boundary values and the exact literal
with supporting original citations, for a separate 3/3. These instructions
change the task; the assisted successes do not replace the automatic failures.
All conditions are single runs at the host's minimum temperature of 0.1.
Differences from historical scores are not statistical estimates of improvement.

### Full-agent negative control

A separate real Spine request deliberately cleared its installed resident record
bank. The actual checkpoint prefix returned an invalid trace, and Rust rejected
it before calling language completion. There were zero accepted completions.
The original resident-bank bytes were restored exactly, and the source heart
was unchanged. This passed the expected negative control despite the request's
nonzero exit status. It complements the standalone zero/restore probe above by
showing that a navigation failure blocks the full-agent source gate.

### Checks and resource evidence

Targeted CPU validation passed 24 backend tests, 26 CLI triangle tests, one
message-serialization check, 28 lifecycle checks and four auxiliary
child/onboarding/regression checks. These are implementation checks, not model
accuracy scores. Sidecar persistence through actual harness checkpoint
serialization is CPU-tested; the GPU interruption test asks a new question.

The corrected matrix ran on one Intel Arc Pro B70. Its guard finished with exit
status zero, peak charged host memory of 3.857 GiB and no high/max/OOM memory
events. Maximum recorded reserved GPU memory was 11.088 GiB. The guard enforced
a 24 GiB host cap, 20 GiB throttle, no swap and a 16 GiB host reserve. These
counters do not cover every driver or shared-cache allocation. The earlier suite
was deliberately interrupted after the compatibility failure; its guard retained
exit status -15, a 4.254 GiB charged host peak and no OOM events.

### November 4 release verification

This revision also fixes the test runner's nonzero-exit enforcement for navigation
failures in shortened runs. The November 3 completed traces independently passed
the navigation checks; their recorded scores are unchanged. This runner correction
does not represent a model rerun or new model measurements.

Ten additional CPU-only driver regressions passed, covering each shortened mode's
success/failure exit, a missing required branch, behavioral failure, preservation
of execution errors, and rejection of `--require-triangle-branch` without
`--triangle-internal-context`. They use synthetic application responses with the
real navigation validator; they are separate from the 83 earlier focused checks.

### Retained first attempt

The first triangle-only application sequence recorded 10 of its 14 assertions
before stopping: seven passed and three failed. The preference answer gave the
correct value with the wrong source citation, the literal answer fabricated a
label/citation, and the file write reached the host action budget without
creating the required file. The remaining remember/restart/interruption checks
did not complete and must not be counted as passes or ordinary answer failures.

The audit verified gating for its 24 completed model rounds, with 72 associated
banks, 105 repeated triangle-record installations and 60 branch commands. One
additional bank executed during the next incomplete request before installation
failed. Each completed round preserved navigation state during language
generation. The original report's lower parser counts remain in the aggregate
alongside these audited counts. The incomplete sequence does not establish full
operational acceptance and fails the corrected full-run-completion gate.

The next remember turn failed with `invalid_forest: Apex is not coherent with
both branches`. Native triangle creation enforces coherence, but native
structural verification permits an existing triangle to survive subsequent
centroid changes. The Python installer reapplied the creation-time cosine
threshold to the current centroids and rejected that native-valid topology.
The suite was stopped for a compatibility repair; no fallback was used. The
native backend now validates retained topology and finite vector shape/norm
without reapplying the creation-time cosine threshold. The standalone
`src/triangle_memory/` loader is unchanged. A native centroid-drift regression
and the repaired backend CPU suite pass; they establish compatibility checks,
not full-model behavioral acceptance.

The first triangle-only boundary run completed 1/2: it recovered the older event
but guessed the required long-source tail. The audit verified its five completed
rounds, 15 banks, four triangle-record installations and four branch commands.
The original report's undercount is retained separately. The source
fixtures were unchanged in both reports. The mixed-memory attempt was stopped
without a completed result. The corrected runs above use fresh test copies and
are kept separately from these first attempts. The first scorer checked only
completed model rounds; the corrected scorer also requires full run completion.

For a populated test fixture, the navigation evidence must show actual checkpoint
prefix execution, successful source gating for every completed model round and
unchanged navigation state during language generation. The suite must
include a real triangle branch; leaf-only opens cannot establish branch execution.
Tool, restart, interruption and assisted-recall behavior have separate assertions.

The [reproduction commands](reproduction.md#internal-context-acceptance) write
`results.json` and `internal-navigation-events.json`. The original
[scoring limitations](results.md#historical-native-harness-empty-navigator)
still apply: the 14 assertions combine answer and operational checks, the restart
assertion does not independently match its citation to the original event, and
interruption recovery asks a new question rather than resuming the interrupted
essay. Explicit recall changes the task and must remain separately scored.
