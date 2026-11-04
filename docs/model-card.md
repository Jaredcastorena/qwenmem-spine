---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: merge
tags:
  - research
  - custom-runtime
  - triangle-memory
  - text-generation
---

# qwenmem-spine

A text-only Qwen3.5-4B derivative with five compiled triangle-navigation layers,
built for an experimental Rust Spine memory integration. It was compiled,
not fine-tuned on the evaluation questions. It requires the matching custom
runtime; ordinary hosted inference widgets and generic generation servers are
not validated entry points.

Source and instructions: this revision of
[Jaredcastorena/qwenmem-spine](https://github.com/Jaredcastorena/qwenmem-spine),
including the opt-in internal-context integration and its November 3 evidence.
Use the runtime from the same repository revision as this card; historical
source baseline pins do not include all integration changes.
Model distribution destination:
[Jared1728/qwenmem-spine](https://huggingface.co/Jared1728/qwenmem-spine).
Access to either repository depends on its visibility and account permissions.
This card describes the measured artifact; it does not assert that a download or
fresh-machine installation has been tested.

## Maintenance and licensing

This is a **fixed, unmaintained research snapshot**. No updates, bug fixes,
new features, compatibility updates or ongoing support are planned. It is
intended for experimental reproduction; this is not a field-of-use restriction.
Commercial reuse is permitted under the applicable license terms.

The base and derivative model weights retain **Apache-2.0**. Newly authored source
code is **MIT**. Rust Spine retains **MIT OR Apache-2.0**, and Qwenputer plus
adapted compiler/runtime code retain **MIT**. Preserve the corresponding notices.
The source repository's `THIRD_PARTY.md` describes component scope. This
maintenance policy does not describe upstream Qwen, Qwenputer or Rust Spine.

## Artifact identity

The measured checkpoint has ten weight shards containing 496 tensors. Its
composite weight identity is:

```text
ebbcad96c6073c9a2cab13cba4d7414f3ed45117626ae8320fd6092a3210ac35
```

This is SHA-256 of the JSON-serialized sorted shard-digest map in `triangle.json`,
not a hash of one concatenated model file. The runtime checks shard identity
before inference. The source repository's `checkpoint-manifest.json` records
the measured 17-file checkpoint set, including configuration, tokenizer,
chat template, index and notices. Checksums detect accidental changes; a
replaceable manifest is not a cryptographic authenticity guarantee.

The merge prepends five BF16 compiled layers and widens the language residual
stream from 2,560 to 2,816. Base-weight padding and normalization-gain adjustment
can introduce BF16 rounding differences, so bitwise equivalence to base Qwen is
not claimed. The model includes no vision component. The separate Qwenputer
Filesystem checkpoint was a comparison input, not the source of these triangle
weights and not a required runtime artifact.

## Memory and runtime behavior

The controller supports 64 combined resident node/triangle records and eight
descents. Its program is in the weights; installed topology and traversal state
live in recurrent tensors. Original source text, provenance, semantic vectors,
stable IDs and the complete native hierarchy remain in Spine's encrypted heart.
Rust chooses semantic routes and evidence. In-model triangle creation, folding,
ranking, paging and autonomous routing are not implemented.

Default chat uses native harness messages and tools and installs an
**empty internal navigation forest**. Its provider performs no extra heart scan.
Native compaction, ranking, automatic hybrid recall and explicit memory tools
remain active. Optional triangle-only automatic memory withholds hybrid recall
text while preserving the task, history, retained state and explicit tools.

The opt-in `--triangle-internal-context` mode installs complete topology for each
native-selected root, bounded to 64 records and depth eight per bank, with no
source bodies in that topology. The actual checkpoint's five-layer prefix
executes navigation commands. Rust independently checks exact traces, installed
addresses and execution evidence before returned apex IDs authorize the original
selected excerpts for the language prompt. An oversized closure or failed check
stops the request. Every completion replays a durable system-message sidecar;
there is no provider-owned source scan or cross-request context store. Ranking,
compaction, source storage and source budgets remain native host operations.
See the source repository's `docs/internal-context-validation.md` for the measured
scope; standalone navigation is a separate execution path.

Use this repository revision's
[Python backend](../rust-spine/backends/triangle/server.py), included Rust source,
pinned compiler and matching checkpoint. Follow the
[reproduction guide](reproduction.md) and
[direct runtime instructions](../rust-spine/docs/triangle-model.md). Runtime
loading uses local files; dependencies must be provisioned first. A generic
Transformers loader is not an established substitute for the command/text
dispatch and state-validation contract.

Normal chat caps prompts at 32,768 exact tokens and generated output at 2,048
tokens per round. Each completion has a fresh fixed-capacity language cache,
256-token prefill chunks and finite-value checks. Navigation tensors must remain
unchanged during language generation. Process restart recovers continuity from
the heart and journal, not an automatically restored full-history KV cache.
Weights do not contain a user's heart, conversation or installed forest.

## Recorded evaluations

The first historical chat battery ran with `--triangle-internal-context` off
and an empty model navigation forest. The second battery, measured November 3,
2026, enabled that flag and used the same unchanged model weights. It is included
in this revision; the flag remains opt-in.

| Evaluation | Result and scope |
| --- | --- |
| Internal native mixed-memory application | 14/14 operational assertions; all 31 completed rounds used verified compiled navigation, with native hybrid evidence also available. |
| Internal triangle-only automatic memory | 12/14 operational assertions; all 27 completed model rounds used verified compiled navigation. Literal/citation and file-write failures remain. |
| Internal triangle-only boundaries | 1/2; older event recovered, long-source-tail value missed despite verified navigation. |
| Internal explicitly instructed recall | 3/3: both boundaries and the literal, with original citations. The literal probe used leaf-only banks. |
| Historical native mixed-memory application | 14/14 operational assertions with an empty navigator forest, including real native/file tools, restart and interruption. |
| Historical triangle-only automatic memory | 11/14 with an empty navigator forest; literal fabrication, wrong-question repair and tool-path failure retained. |
| Historical unassisted source boundaries | Native mixed-memory 0/2 because required original citations were omitted; triangle-only 1/2. |
| Historical explicitly instructed native recall | 3/3 selected probes: both boundaries and exact Unicode literal, with original citations. Assisted results do not replace failures. |
| Earlier compiled-navigation retention | 8/8 at each of 4,274, 32,812 and 131,215 stored content tokens, with 572–901-token evidence prompts. Two extra renderer boundaries failed. |
| Valid raw-context control | 8/8 at 167,780 input tokens; a fresh process restored the full cache and reproduced answer bytes and token IDs. |

These are small fixed synthetic tests and operational assertions, not general
accuracy estimates. The internal triangle-only run passed the navigation gate
while failing two behavioral assertions; navigation is not answer verification.
The internal mixed mode also supplies native hybrid recall; its answer score
does not isolate the gated triangle excerpts as the source of success. These
are single runs with the host's minimum temperature of 0.1.
All 70 completed internal-context rounds passed the source gate across 210 bank
executions and preserved navigation state during language generation. A separate
full-agent resident-bank fault produced an invalid trace and was rejected before
language completion,
with zero accepted completions. The [aggregate evidence](../evidence/internal-context-20261103.json)
records these runs and the independent standalone zero/restore probe.

The historical application controls above do not exercise active compiled
navigation. Earlier provider trials used an extra source scan that has since
been removed; their results do not describe current retrieval behavior. Complete
tables, failure categories and sanitized aggregate evidence are in the source
repository's `docs/results.md` and `evidence/`.

## Known failures and limitations

The Rust host's claim extractor can miss factual string identifiers, producing
zero claims and bypassing NLI and repair. This is tracked in
[Rust-spine issue #40](https://github.com/Jaredcastorena/Rust-spine/issues/40);
no fix is included. Forced extraction triggers repair for the wrong Unicode
literal but also rejects the correct literal under the measured NLI gate.
Pair-order reversal is not a reliable exact-string fix. A real citation does not
by itself establish source support. The defect does not invalidate the working
provider/tool connection and does not explain every automatic-answer failure.

Native source rendering can omit older events or truncate required tails.
Explicit recall recovers the measured boundary fixtures but has no arbitrary
event-pagination guarantee. Repair can answer an older question. These limits
preclude reliable autonomous recovery or lossless/unlimited-memory claims.

Two early raw-context attempts were interrupted without complete answers. A
separate bounded run developed NaN recurrent state and logits and emitted
punctuation. That invalid output is not scored as forgetting. A later checked
replay and restart passed, but the original numerical trigger remains unresolved.
The runtime's finite-value checks do not establish a permanent backend fix.

Full-model tests used Linux, Python 3.12.3, PyTorch 2.14.1+xpu, Transformers
5.19.0, safetensors 0.8.0, NumPy 2.5.4 and one Intel Arc Pro B70 with 32 GB device
memory. Controller checks also run on CPU. Full-model CUDA, macOS and Windows
execution and clean-machine installation are unverified. Guarded runs used a
24 GiB host cgroup limit, 20 GiB throttle, disabled swap, 16 GiB host reserve and
70% XPU allocator cap. These are measured controls, not portable memory-fit
guarantees. Source text and dense runtime state still consume storage.

## Pinned provenance

| Component | Revision |
| --- | --- |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `divagr18/qwenputer` | `01cf657d353bdc0c795c2d57a28a3e3f483f20a1` |
| Integration source and runtime | This repository revision, including the November 3 internal-context implementation and evidence. |
| Historical Rust Spine baseline | `07857378505d2573b623da4f81a892bfb9b5f2cd`; the included source adds integration changes. |
| Published merged model weights | `f69dbcfe5a7cebea1c287f6f1d76c8e90b2c9a98` at `Jared1728/qwenmem-spine` (unchanged). |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `cross-encoder/nli-MiniLM2-L6-H768` | `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` |

The source repository's `models.lock.json` and `requirements.lock.txt` identify
download/build inputs. Source, weights, navigation state, encrypted heart and
raw-context cache checkpoints are distinct artifacts with different roles.
