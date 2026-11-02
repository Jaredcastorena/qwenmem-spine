# qwenmem-spine: compiled navigation with durable source evidence

Research measurements from October 31–November 2, 2026. This is a fixed,
unmaintained research snapshot. New code is MIT licensed; experimental intent
does not restrict commercial reuse under the applicable licenses.

## Research question

Can a small deterministic navigation program execute inside language-model
layers, retain its traversal state across sessions, and reconnect a fresh
language context to original evidence held by a durable memory system?

The experiment demonstrates that bounded mechanism. Five compiled BF16 layers
store a resident triangle forest and execute navigation commands. Rust Spine
retains original events and chooses semantic routes. A host adapter resolves
returned apex addresses into evidence for Qwen3.5-4B. A separate full-application
integration shows that the merged checkpoint works through Spine's normal model
provider and native tools. These two results have different execution scopes:
the current normal-chat path uses an empty compiled navigation forest, and its
bounded handoff to the internal controller is unfinished.

The contribution is an implemented and tested connection between a compiled
navigation machine and a provenance-preserving source store. It does not show
that arbitrary conversations have been compressed into model weights, that the
language model autonomously chooses routes, or that the complete native triangle
lifecycle has moved inside the model.

## Representation and execution

A triangle names a real apex node and two children. Children may themselves be
triangles. The apex refers to existing evidence rather than generated summary
text. Native compaction retains descendants in the durable registry and replaces
roots with recursive handles. An outer handle can remain small while backing
storage grows. A controlled native audit measured the same 33-byte serialized
root handle from one through 96 source coordinates; all 96 were reachable after
reload. That is reference retention, not constant-space storage or semantic
recall.

The compiled controller installs a flattened resident table of at most 64 total
nodes and triangles. It supports eight descents per opening, real apex emission,
explicit leaves and errors, and an eight-entry alternate-route stack. The
record table is read-only during navigation; cursor, pending child, route stack
and descent count mutate in Gated DeltaNet recurrent state. The host validates
topology, provenance, address resolution, finite semantic coordinates and the
apex/branch coherence constraint before installation.

Qwenputer compiles the program to five layers with five logical heads, 185 logical
residual dimensions and 217 logical feed-forward units. The merged checkpoint
prepends those layers to the language stack and expands the residual width from
2,560 to 2,816. Base weights are padded and normalization gains adjusted.
BF16 rounding can alter language behavior; the merge does not promise bitwise
equivalence to base Qwen. It extracts the text component and includes no vision
model. No evaluation questions are used to fine-tune the language weights.

Command execution and language generation are explicitly dispatched by the
runtime. Commands update the compiled prefix. Source-based answers use a fresh
language cache and preserve the installed navigation tensors. Some language
prefill/control paths traverse the full stack with dormant circuit features;
ordinary text decoding does not autonomously follow the triangle table. The
runtime is part of the method, so a generic text-generation server is not an
established substitute. See [architecture](architecture.md) and the
[merged runner](../src/triangle_memory/merged_runner.py).

## State and persistence

There are four separate objects:

| Object | Contains | Persistence |
| --- | --- | --- |
| Model weights | Navigation program and pretrained language parameters | Generated checkpoint. |
| Navigation state | Installed topology, cursor, alternates and counters | Identity-bound safetensors snapshot. |
| Spine heart | Source bodies, original event IDs, durable hierarchy and journal | Encrypted host store. |
| Language cache | Token-history keys/values and hybrid recurrent state | Normally fresh; separately checkpointed only by raw-context controls. |

Navigation restore checks forest and weight identities, schema, shapes and
checksums before changing live state. Forest replacement invalidates old slot
addresses. These checks detect accidental corruption; they do not authenticate
a file against someone able to rewrite its data and metadata.

An initial separate-session demonstration restored an alternate route and
recovered the original offline recovery-port exception using zero prior
conversation tokens and a fresh 126-token prompt. Dense navigation state was
10,815,512 bytes, and the merged weights were roughly 9.8 GiB. Original evidence
still occupied the heart. Small prompt size therefore is one measured cost, not
an overall compression ratio. [Results](results.md) include the timing and
separate-process follow-up.

## Three evaluation scopes

### Standalone native-memory and compiled-navigation evaluation

The synthetic harness uses real Spine MiniLM embeddings, signed events, the
encrypted heart, DCMDb updates, compaction, ranking, rehydration and CLI evidence
rendering. It closes and reopens the store at three checkpoints. For 24
question/history pairs, real compiled BF16 traces match native routes and the
adapter's rendered evidence matches the native CLI exactly.

Eight primary questions pass at each of 4,274, 32,812 and 131,215 stored content
tokens. Retrieval prompts are 572–901 tokens. Seven questions require source
citations; the eighth asks for an unknown fact and correctly abstains. A recent
2K-source-body baseline passes only that abstention, while oracle evidence passes
all eight. Oracle labels do not guide native route selection. Two extra boundary
questions fail because the historical renderer selects only two recent events
per node and truncates source bodies at 1,000 characters.

The corpus has 534 events at its largest stage, 16 repeating distraction
templates and 23 logical session labels. Those labels are not 23 process restarts
or a study of elapsed-time aging. Natural-language compaction creates only one
triangle at each stage. A separate designed-vector fixture covers 32 nodes,
31 triangles and depth eight; it tests geometry, not learned semantic quality.

### Full-history controls

Valid full-history runs also answer all eight questions. The largest consumes
167,780 prompt tokens, including metadata. Its bounded cache preallocates full
attention storage, appends in place and retains the hybrid recurrent state.
Checkpoints partition state across 38 files and bind it to prompt identity,
position, capacity, weights and checksums. They are persistence partitions of
one continuous computation; no history tokens are intentionally summarized,
evicted or split into independent conversations.

Two early large-context attempts were interrupted and produced no scored answer.
A later bounded run completed input but developed NaN recurrent state and logits,
then emitted 1,400 punctuation tokens. Its mechanical 0/8 audit is not a valid
forgetting score. A distinct replay with per-operation numerical checks and
synchronization resumed a finite 65,536-token checkpoint and produced 8/8.
Another process restored the final full-input cache and reproduced the same
answer bytes and generated tokens. The successful replay does not isolate the
earlier NaN trigger or establish a permanent numerical fix.

The raw controls answer all questions together, while retrieval prompts answer
one at a time. Their timing is not a matched throughput comparison. The study
establishes a difference in language-context input, with external storage and
runtime costs still present.

### Current complete agent integration

The current provider consumes context and tools supplied by Spine. It does not
own a heart/encoder or run an extra source scan. The Rust harness owns tool
permissions, action budgets, planning, transcript recovery, native recall and
grounding. The model/provider connection is operational: native mixed-memory
tests pass 14/14 operational assertions, including real file tools, restart and
interruption recovery.

With automatic hybrid recall withheld, triangle-only automatic memory passes
11/14. Failures include a fabricated exact literal, a repair that answers an
older question, and a file write using an invented absolute path. Separate
explicit-recall instructions recover both boundary sources and the literal,
with original citations, for 3/3. These assisted results do not replace the
unassisted failures.

The current controls install an empty compiled forest. The completed connection
and native recall results therefore do not establish active compiled navigation
inside ordinary chat. Earlier integration trials did use a provider-owned source
scan and compiled traces; those historical results are recorded separately and
are not acceptance of the current path.

## Host grounding defect

The failing Unicode answer contains a false label and fabricated citation. The
native extractor yields zero claims, so NLI and repair are skipped. A four-cell
comparison of original versus forced extraction and short versus full evidence
confirms the boundary. The source event is only 108 characters, making the two
evidence windows identical; this example cannot measure truncation. Forcing the
wrong claim triggers repair, but forcing the correct exact literal also fails
coverage. Pair-direction probes further show that reversing NLI inputs would
not safely repair exact-string verification.

This host validation defect is tracked in
[Rust-spine #40](https://github.com/Jaredcastorena/Rust-spine/issues/40), and its
fix is not included. It explains a verification bypass, not all failed answers,
tool arguments or recovery decisions. See the
[diagnostic](unicode-nli-diagnostic.md) and [known limitations](known-limitations.md).

## Validation and interpretation

Independent integer traces test the compiler's bounded behavior; real CPU/B70
checks test generated weights; state checks test rejection and restart. Five
Bend selector/apex laws concern the abstract specification, not the compiler,
PyTorch runtime or language model. Small-cache parity tests at tolerance `1e-6`
do not establish long-run numerical stability. Unit tests, synthetic answer
scores and application assertions must remain separate kinds of evidence.

All reported large-model runs used one Intel Arc Pro B70. Guarded bounded runs
used a 24 GiB host limit, 20 GiB throttle, no swap, 16 GiB system-headroom reserve
and 70% XPU allocator cap. The seven bounded raw processes recorded at least
52.001 GiB available host RAM, at most 13.095 GiB charged cgroup memory and
15.303 GiB reserved device memory, with no recorded memory-limit events. Those
counters do not account for every shared-cache or driver allocation.

The tested controller is bounded, externally installed and host-routed. The
source store remains necessary. Source selection, exact-value grounding,
autonomous tool choice, numerical stability and the unfinished native-to-internal
handoff limit what can be inferred. This snapshot documents those limits without
promising later development. Pinned inputs and commands are in
[reproduction](reproduction.md); aggregate measurements are in
[evidence](../evidence/provenance.json).
