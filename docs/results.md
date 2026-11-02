# Recorded results and failure record

Measurements span October 31–November 2, 2026. This report separates current
normal-chat behavior, earlier compiled-navigation experiments and superseded
provider trials. Aggregate source data are included in
[current-harness.json](../evidence/current-harness.json),
[retention-and-raw-controls.json](../evidence/retention-and-raw-controls.json),
[nli-diagnostic.json](../evidence/nli-diagnostic.json) and
[provenance.json](../evidence/provenance.json). The tables report measured runs;
they do not promise identical regenerated model outputs or a tested clean install.

## Current native harness

All runs use the actual unchanged merged checkpoint, real `spine chat`, NLI and
native tools with newly created test copies of synthetic stores. The provider
adds no source scan and installs an empty internal forest. Thus these evaluate
the model/harness connection and native memory behavior, not active compiled
navigation during chat.

| Run | Strict score | Observed scope |
| --- | --- | --- |
| Native mixed-memory sequence | 14/14 | Eight primary questions, citations, real file tools, persisted fact after restart and interruption recovery. |
| Native mixed-memory boundaries | 0/2 | Both values correct; both required original event citations omitted. |
| Triangle-only automatic-memory sequence | 11/14 | Literal fabrication, wrong-question repair and file-write failure. |
| Triangle-only boundaries | 1/2 | Older event recovered with citation; source-tail label guessed incorrectly. |
| Explicit recall of boundaries | 2/2 | Separately instructed native recall returns exact values and original citations. |
| Explicit recall of literal | 1/1 | Separately instructed recall returns exact Unicode label and original citation. |

The 14-assertion sequence combines answer and operational checks; it is not a
14-question statistical benchmark. `--triangle-context-only` removes automatic
hybrid recall text but retains task/history, native triangle context, plans,
working evidence, tools and policy. The three assisted recall successes do not
replace the automatic failures.

The [operational scorer](../scripts/spine_end_to_end.py) checks a new recall
request after interruption; it does not test resuming the interrupted essay.
Its restart assertion requires the persisted beacon value and a 64-character
hexadecimal string, but does not independently match that
identifier to the original event. The 14/14 score retains these original
acceptance criteria and should not be read as verification of task resumption
or of citation provenance after restart.

The literal failure invented `ZIRCON_RECOVERY_Ω` instead of `Δ-sable_47/µ` and
fabricated a citation without a recall call. The extractor emitted zero claims,
so NLI/repair did not run. For the unknown-IP question, the gate *did* extract a
claim and request repair (coverage 0.016); repair then answered the previous
label question. The latest question remained in the supplied request. The
file-write failure used an invented absolute `/workspace` path, repeated it
after denial, and was stopped by the action budget. The source-tail answer cited
the real event but guessed a value absent from the visible prefix.

The boundary source is 2,320 characters: native recall's 3,000-character prefix
can expose its tail, whereas the 1,000-character triangle excerpt cannot. This
does not test arbitrary source pagination. The NLI problem is described in
[issue #40](https://github.com/Jaredcastorena/Rust-spine/issues/40) and the
[diagnostic](unicode-nli-diagnostic.md); no fix is included, and that bug is not
an explanation for all three sequence failures.

At this stage the Rust workspace passed 277 tests and the Python backend passed
19; strict Clippy/build checks also passed. Recorded request checks confirmed at
most 12 triangle excerpts, zero automatic hybrid-recall records in triangle-only
mode, and no provider-owned source scan. Guarded runs kept the 24 GiB host cap,
20 GiB throttle, no swap and 16 GiB reserve, with no recorded memory events.
These are historical validation totals, distinct from the packaging checks below.

## Standalone compiled navigation and source retention

The harness uses real MiniLM embeddings (256-token encoder input), encrypted
storage, signed events, native DCMDb/compaction/ranking/rehydration and the CLI
renderer. Across 24 question/history pairs, real BF16 navigation traces match
native coordinates and alternate routes, and adapter evidence matches the CLI.

| Stored content tokens | Events | Logical session labels | Triangle + native sources | Recent 2K source bodies | Oracle sources |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,274 | 26 | 2 | 8/8 | 1/8 | 8/8 |
| 32,812 | 140 | 6 | 8/8 | 1/8 | 8/8 |
| 131,215 | 534 | 23 | 8/8 | 1/8 | 8/8 |

Seven questions require correct values with supporting original citations; one
requires abstention. They cover a later correction, exception, pending approval,
old failure code, formatting preference, exact Unicode, cancellation and an
unknown fact. At the largest stage, the active identifier changes from
`FALCON-AX-482` to `RAVEN-ZQ-917`. Triangle-assisted prompts use 572–901 tokens.
The recent baseline succeeds only on abstention, and its 2,048-token body budget
becomes roughly 2,660 prompt tokens with metadata. Oracle labels select only
upper-bound evidence, never native routes.

There are three actual store-reopen checkpoints; the session labels are logical
threads. Canonical event content survives, cognition is current, transcript
persistence is idempotent and maintenance retains the forest. Natural-language
compaction yields 18 nodes plus one triangle at 4K/32K and 19 nodes plus one
triangle at 128K. A designed-vector fixture separately checks 32 nodes,
31 triangles, depth eight and 32 native walks on CPU/B70.

| Additional boundary | Triangle + native renderer | Oracle/full source | Cause |
| --- | ---: | ---: | --- |
| Older still-binding event within a merged node | 0/1 | 1/1 each | Only the two most recent node events are rendered. |
| Required value near the end of a long source | 0/1 | 1/1 each | Source rendering ends at 1,000 characters. |

Original evidence remains intact. The model reports missing evidence when given
the incomplete excerpts. Both cases remain strict expected-failure tests once
their native fixtures are generated. Correct address traversal cannot recover
text omitted after selection.

A separate-process check saves an alternate route, exits, restores it in another
model process and correctly answers `CANCELLED` with the recovered event ID.
An earlier prompt asked for task details absent from the selected cancellation
record and received an appropriate abstention; another ad hoc prompt failed the
JSON-format assertion. The final established benchmark prompt passed. These
prompt failures are separate from state-restore correctness.

## Full-history controls and numerical failure

| Stored content | Actual prompt tokens | Recorded outcome |
| ---: | ---: | --- |
| 4,274 | 6,435 | 8/8 independently source-verified; bounded stop at 4,096 and fresh-process continuation reproduce the prior raw answer bytes. |
| 32,812 | 42,742 | 8/8 independently verified; bounded run matches the prior raw answer bytes. |
| 131,215 | 167,780 | Initial bounded completion has NaN state/logits and 1,400 punctuation tokens; mechanical 0/8 is numerically invalid. |
| 131,215 | 167,780 | Distinct checked replay from the finite 65,536-token checkpoint: 8/8, 746 answer tokens, no truncation. |
| 131,215 | 167,780 | Fresh process restores full-input cache: 8/8 again; answer bytes, generated tokens and prompt identity exactly match. |

The successful large replay retains every prompt position in all eight
full-attention caches and leaves navigation state unchanged. Its finite checks
also synchronize execution more frequently. No CPU fallback was triggered, and
the failing operation did not recur. The original NaN trigger is unresolved.

The invalid run's 65,536-token checkpoint is finite. At 131,072 and 167,780 tokens,
NaNs occur from merged layer index 33 in language recurrence and later layers,
with non-finite final logits; dormant prefix layers 0–4 remain finite zeros.
Argmax over those invalid logits explains the punctuation output. It is not an
interpretable forgetting measurement.

Two earlier attempts produced no complete answer: eager 32K prefill stopped after
32,768 of 42,731 logged tokens, and fused 128K prefill last logged 65,536 of
167,780 tokens. Host memory pressure/device reset and a later host reboot were
observed, but host peak RAM was not measured and exact causes remain unresolved.
They remain unscored. The legacy runner stays capped at 45,000 prompt tokens;
larger controls use the separate bounded runner with finite checks.

Bounded caches retain full input and save 38 state files, not summaries or
independent conversations. A 4K checkpoint is 197,439,624 bytes; the large
65,536-token checkpoint is 2,210,705,608 bytes. Across seven bounded processes,
minimum host availability was 52.001 GiB, maximum charged cgroup memory
13.095 GiB and maximum reserved XPU memory 15.303 GiB. No cgroup memory-limit,
throttling or OOM events were recorded. These counters are not total driver or
shared-cache accounting. One B70 was used; no second GPU was needed.

Earlier completed raw 4K/32K runs took 43.740/81.667 seconds with peak PyTorch
allocations of 10.16/11.31 GiB. All eight questions were answered together, so
those times cannot fairly be compared with per-question retrieval latency.

## Historical provider experiment

An earlier full-application provider owned a source-window scan and installed
whole exported forests. It passed 14 operational assertions and both boundary
questions, including 8/8 primary answers, real tools, persisted-fact recall and
interruption recovery. This source-scan path was subsequently removed. Its
successes do not describe the current provider or solve the current bounded
internal-handoff problem.

That main run had 21 model rounds: 15 used compiled navigation and six used
explicit native fallback after exceeding 64 resident records. Prompts were
11,115–17,561 tokens, including the full tool interface and dialogue. Earlier
source-window bounds were 12 excerpts/8,192 characters across all canonical
inline events. The scan's cost and richer evidence prevent an exclusively
triangle-directed or sublinear-retrieval claim.

Its restart citation came from a prior assistant acknowledgement, so that check
established persistence rather than independent-source attribution. Interruption
was followed by a new recall request; resuming the interrupted essay itself was
not tested. Terminal input was also exercised. At that stage Rust had 274 passing
tests and the backend 19. Later native-harness totals above supersede the Rust
count. Minimum host availability was 50.115 GiB and peak reserved XPU memory
12.719 GiB, with no recorded guard events.

Earlier integration attempts failed because of a launcher positional argument,
an impractically slow unoptimized build, JSON versus native XML tool framing,
and rejection of capitalized boolean scalars as JSON. These are implementation
failures, not recall scores. The corresponding parser/build corrections are
included; the earlier long raw-context NaN cause remains unresolved.

## Initial controller and formal checks

The first saved-route demo recovered an offline recovery-port exception (`7300`)
with zero prior conversation tokens and a 126-token prompt. It emitted 32 answer
tokens; the dense navigation snapshot was 10,815,512 bytes. Load/verification
took 19.374 seconds, prompt/generation 8.142 seconds and the whole demo
27.784 seconds. Subsequent commands took about 24–31 ms. A fresh process answered
again with 171 prompt tokens/30 answer tokens in 29.631 seconds. This is a small
demonstration, not an overall compression or throughput comparison.

Initial development failures included inference-mode tensor mutation, an emitter
naming requirement and a stalled single-copy transfer of a large memory-mapped
embedding. The final loader uses 4 MiB staging copies. The first controller CPU
suite reached 12 unique checks, with four B70 model checks plus a separate
full-capacity boundary check. Five Bend selector/apex laws passed both the normal
checker and independent verifier using Lean 4.34.0. The laws do not prove runtime
navigation, compiler translation or language behavior.

A later internal-state audit matched all 70 prefix tensors, reserved command
embedding rows and output dimensions against the measured checkpoint. Small
BF16 tests then executed with host forest lookup blocked, showed that zeroing
resident records invalidated open, and restored records/cursor from a snapshot.
Its initial import failure preceded the successful audit. This was CPU controller
verification, not another full-checkpoint semantic benchmark.

The bounded-control-stage CPU suite reported 23 passes, one optional 4B skip and
two expected renderer failures; seven later affected checks included non-finite
checkpoint rejection. Native regressions then added 25 passes. During source
snapshot preparation, CPU checks on the portable package reported **20 passes
and seven skips**, and its Rust harness **26 passes**. The controller-only demo,
snapshot/reopen and state-residency audit also passed. The seven skips reflect
absent generated fixtures/optional full-model execution; no new 4B or GPU
benchmark was run during packaging. Counts from different suites/stages are not
summed into one score.
