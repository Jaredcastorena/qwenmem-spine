# Local triangle model integration

The local triangle provider runs through normal `spine chat`, including terminal and browser input, the tool registry, host policy, encrypted transcript journal, grounding, context compaction and operator controls. Rust owns the heart and selects memory. A persistent Rust-owned Python process runs the merged Qwen3.5 checkpoint. This remains a Python-backed inference implementation. The current normal-chat path is a native-harness control: it installs an empty navigation forest and consumes only the harness's messages. Compiled navigation is still exercised by the separate `spine triangle` diagnostic; it is not currently active in normal chat.

The actual checkpoint is connected to the normal harness: native memory recall,
file tools, heart restart and interruption recovery have been exercised. The
native mixed-memory acceptance sequence passes 14/14 checks, and explicit memory
recall recovers the tested omitted source values with their citations. The known
NLI extraction bug is a host validation limitation; it does not imply that the
provider or memory-tool connection failed. The separate internal-navigation
handoff remains unfinished as described below.

## Required artifacts

Use the pinned Rust toolchain and Cargo dependencies, an existing Python environment matching `backends/triangle/requirements-runtime.txt` with PyTorch, the merged checkpoint including `triangle.json` and all ten weight shards, the pinned Qwenputer compiler checkout, MiniLM embeddings and Spine's pinned NLI snapshot. Runtime loading uses local files only. The merged model's shard identities are checked before loading. The filesystem Qwenputer comparison model is not the runtime model.

Build from this fork:

```sh
cargo build -p spine-cli --locked --offline
```

The development profile is optimized because native CPU embedding and grounding inference are too slow without optimization. Release builds retain the upstream release profile.

## Normal agent launch

Set `PYTHON`, `TRIANGLE_MODEL_DIR`, `COMPILER_DIR`, `MINILM_DIR` and `NLI_DIR` to existing local artifacts, then run:

```sh
./target/debug/spine --no-settings chat ./triangle.heart \
  --workspace ./workspace --model-dir "$MINILM_DIR" --nli-model-dir "$NLI_DIR" \
  --no-download --terminal --max-tokens 1024 \
  --triangle-backend "$PYTHON" \
  --triangle-backend-arg "$PWD/backends/triangle/server.py" \
  --triangle-backend-arg=--model-dir --triangle-backend-arg "$TRIANGLE_MODEL_DIR" \
  --triangle-backend-arg=--compiler-dir --triangle-backend-arg "$COMPILER_DIR" \
  --triangle-backend-arg=--device --triangle-backend-arg cpu
```

Create the workspace directory first. A missing heart is initialized by the normal first-run flow; the operator chooses its passphrase. Replace `--terminal` with `--web` for the existing browser interface. For the configured Intel XPU, select `xpu` and launch the entire application through the host memory guard. Local triangle sessions do not read or save remote-provider connection preferences.

The explicit path-based command above is the portable configuration interface.
Provision the Python environment, compiler checkout and model artifacts separately
before launch. They are not bundled into this source distribution. The automatic
dependency setup used by standard Spine does not install this optional backend's
large checkpoint or compiler.

## Retrieval and model ownership

Native triangle compaction is recursive. A parent stores a real apex node and two
child handles; it retains descendant triangles in the durable registry. Three
coherent triangle roots can therefore become one through two pairwise folds.
The outer handle remains small while the backing heart and registry grow. A
coherence failure can leave more roots than the requested target.

Normal chat preserves native ranking and rehydration: three ranked roots, each
bounded at depth eight, fanout three, 16 coordinates and 1,024 estimated tokens.
The renderer emits at most 12 source excerpts of 1,000 characters. The normal
default also supplies native automatic hybrid recall. `--triangle-context-only`
withholds that automatic recall text while keeping the current task, bounded
history, retained plans and working state, tools and host policy. It means
triangle-only **automatic memory retrieval**, not an instruction-free prompt or
triangles shared directly with the language KV cache.

The provider has no heart or encoder handle, adds no evidence, and makes no
private retrieval calls. The model can request `heart_recall`/`search_memory` or
fact tools through the normal harness. Native recall returns bounded source
prefixes up to 3,000 characters; it does not expose a continuation offset.
Triangle-only mode adds the displayed triangle source texts to grounding evidence.
The baseline claim extractor can miss arbitrary string claims, and a repair can
answer the wrong question: grounding is not a guarantee of successful recovery.

The standalone diagnostic still flattens a forest into 64 resident slots and
checks host-selected commands against independent traces. That limit is not a
native cap on retained history. The next internal-controller handoff must accept
bounded views selected by the harness, preserve dormant handles and revisions,
and be tested separately. Whole-heart export and provider-owned source scanning
have been removed from normal chat; their earlier results are historical.

Each completion uses a fresh, fixed-capacity language cache. Prefill processes 256 tokens at a time; XPU operations synchronize at chunk boundaries, recurrent chunk outputs and logits are checked for finite values, and language generation must leave navigation tensors unchanged. There is no automatic raw full-history cache restoration during chat. Process restart loads the merged weight checkpoint and reopens the heart; the native harness rebuilds its bounded context from durable evidence.

## Tools, bounds and interruption

The provider renders the checkpoint's own chat template, including tool descriptions and prior tool results. It parses Qwen's `<tool_call><function=...><parameter=...>` format, preserves schema-declared strings, supports the template's `True`/`False` scalar spelling, rejects duplicate/unknown parameters and unavailable tools, and returns calls to the existing Rust harness. It does not execute tool text in Python. Tool permissions, action budgets and journaling remain host-owned.

Local chat accepts at most 32,768 prompt tokens and 2,048 generated tokens per model round; the default local output budget is 1,024. Rust reuses the native Llama provider's message budget with space reserved for tools and output. It drops older complete turns first, then marks shortened message content while retaining message/tool structure. This is an estimate; the backend rejects any remaining exact-token overflow. Incomplete output is rejected. Host temperature modulation is passed through; tests requesting base temperature zero still receive the host's minimum 0.1. Output arrives at completion boundaries rather than token-by-token streaming.

Only one completion owns the model process at a time. Timeout, backend failure or dropped inference cancels that child. A later request starts a new child and reinstalls the forest. The normal heart journal retains durable user messages and tool outcomes; rejecting an incomplete assistant response does not mean those earlier records disappear. `/interrupt`, `/stop` and `/resume` retain their upstream meanings.

The verified B70 launch uses a 24 GiB host cap, 20 GiB throttle, disabled cgroup swap, a 16 GiB global reserve and a 70% XPU allocator cap. These are local execution constraints, not hardware-independent fit guarantees. The earlier long raw-context NaN failure remains unresolved; synchronized checks fail on invalid numerical output but do not establish a general backend fix.

## Navigation diagnostic terminal

`spine triangle` remains available for direct `/open`, `/left`, `/right`, `/route`, `/save` and `/restore` operations with verified traces and revision-bound snapshots. Its `--backend-program`/`--backend-arg` options configure the same backend. These snapshots persist compiled navigation state and are separate from model weights, the encrypted heart and language KV checkpoints. Normal chat continuity uses the heart and journal.

## Validation scope

Known upstream NLI limitation: string-valued claims can produce zero extracted
claims and bypass verification/repair. [Rust-spine issue #40](https://github.com/Jaredcastorena/Rust-spine/issues/40)
has been submitted; a fix is deferred. See [known limitations](known-limitations.md).

The full workspace and backend unit suites exercise native projection, cryptography, tool lifecycle, browser API, persistence, protocol parsing, source pagination and state validation. Actual-model application acceptance additionally uses disposable copies of synthetic hearts, the public browser chat API, local memory tools, NLI and explicit filesystem tool tasks. The lab records exact outcomes separately; unit or mock checks are not counted as real-model accuracy evidence. Clean-machine installation, broad model quality and public distribution remain separate release gates.
