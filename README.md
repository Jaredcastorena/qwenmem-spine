# qwenmem-spine

**Model weights:** [Jared1728/qwenmem-spine on Hugging Face](https://huggingface.co/Jared1728/qwenmem-spine).
See the [pinned download and setup instructions](docs/reproduction.md#merged-checkpoint-download-or-build).

An experimental memory system that combines Rust Spine's durable memory and
agent tools with a modified Qwen3.5-4B language model. Five added model layers
execute a small navigation program over a hierarchy of memory records.
Original source text stays outside the model, in Spine's encrypted memory store
(the **heart**) or the standalone tests' source fixtures.

Standalone tests exercise the compiled navigator. The first historical chat
test battery ran without `--triangle-internal-context`, using Spine's native
retrieval and tools with an empty model navigation forest. The second battery,
measured November 3, 2026, enabled `--triangle-internal-context`: native-selected
topology runs through the compiled navigator before the selected source excerpts
reach language generation. This revision includes that opt-in integration and
both batteries' evidence. Default chat retains the empty-forest path.

**This snapshot is unmaintained.** No updates, bug fixes, new features,
compatibility updates or ongoing support are planned. This describes maintenance
status and intended research use, not a field-of-use restriction: new code is
[MIT licensed](LICENSE), including commercial reuse under its terms. Upstream
components retain their own licenses. This policy does not describe the
maintenance status of Rust Spine, Qwenputer or Qwen.

## What is implemented

- A deterministic controller with **64 combined node/triangle addresses**, eight
  descents per traversal, real apex nodes and saved alternate routes.
- BF16 compiled navigation layers, a merged text-only Qwen checkpoint builder,
  verified recurrent-state save/restore and an independent integer reference.
- The normal Rust Spine agent loop, connected to the model through a persistent
  Python process, with native memory tools, file tools, grounding, durable
  transcripts, restart and interruption handling.
- An opt-in native-selected internal context handoff: complete bounded root
  topology runs through the checkpoint's compiled prefix, and Rust verifies its
  exact traces before releasing the original selected source excerpts.
- Synthetic evaluation builders, scoring/audit code, tests, abstract selector
  proofs and sanitized aggregate [evidence](evidence/provenance.json).

**Default chat still installs an empty internal navigation forest.** Internal
context is an explicit option, with a separate
[validation record](docs/internal-context-validation.md). Each installed bank
contains a selected root's complete original topology, bounded to 64 records and
depth eight; source bodies stay on the host. Triangle creation, semantic ranking
and compaction remain host operations. This is not a full internal triangle
implementation or a claim of unlimited, lossless context.

## Measured results

| Evaluation | Recorded result | Scope |
| --- | --- | --- |
| Internal native mixed-memory application | 14/14 operational assertions | Verified compiled navigation in all 31 completed rounds; native hybrid evidence also available. |
| Internal triangle-only automatic memory | 12/14 operational assertions | Verified compiled navigation in all 27 completed model rounds; literal/citation and file-write failures retained. |
| Internal triangle-only boundaries | 1/2 | Older event recovered; long-source-tail value missed despite verified navigation. |
| Internal explicitly requested native recall | 3/3 | Two boundaries and one literal; assisted probes, with leaf-only navigation in the literal probe. |
| Historical native mixed-memory application | 14/14 operational assertions | Actual checkpoint and Rust harness; empty navigator forest, tools, restart and interruption. |
| Historical triangle-only automatic memory | 11/14 | Empty navigator forest; three failures retained; explicit memory tools remain available. |
| Historical explicitly requested native recall | 3/3 | Two boundary cases and one Unicode literal; assisted probes, not replacement scores. |
| Earlier standalone retention | 8/8 at each of 4,274, 32,812 and 131,215 stored content tokens | 572–901-token retrieval prompts; two additional source-selection failures. |
| Valid full-history control | 8/8 at 167,780 input tokens, repeated after cache restore | A separate earlier run developed NaNs; its cause remains unresolved. |

The [November 3 evidence](evidence/internal-context-20261103.json) records verified
gating in all 70 completed internal-context model rounds across 210 bank
executions. A separate resident-bank fault caused the full agent to reject
navigation before language completion.
The fixed synthetic questions and operational assertions are not population
accuracy estimates. The full [results and failure record](docs/results.md)
separate execution modes and earlier provider experiments. The known host
claim-extraction/grounding bypass is tracked in
[Rust-spine issue #40](https://github.com/Jaredcastorena/Rust-spine/issues/40);
this snapshot does not contain a fix.

## Start here

This repository contains source and aggregate evidence, **not model weights**.
The [reproduction guide](docs/reproduction.md) covers pinned downloads, building
the merged checkpoint, CPU checks, guarded Intel XPU runs and fixture generation.
A clean-machine installation has not been validated. The tested full-model
environment was Linux, Python 3.12 and Intel Arc Pro B70; portability to other
accelerators is not established.

For a first run, follow the guide's [small CPU controller](docs/reproduction.md#small-cpu-controller)
section. It exercises navigation without downloading the 4B language model.
For chat, use the [full Rust agent](docs/reproduction.md#full-rust-agent-and-current-acceptance)
section after provisioning its dependencies. Build the included `rust-spine/`
source: standard upstream Spine binaries do not include this integration.

After provisioning the environment and model described in that guide:

```bash
# Compiled controller only, on CPU:
./run.sh

# Merged-model navigation and source-based answer, on the tested Intel setup:
./scripts/guarded-python.sh -m triangle_memory.cli --device xpu \
  --model models/qwen3.5-triangles demo --output artifacts/merged-demo

# Full Rust agent, after building its binary:
./scripts/spine-local.sh --terminal

# Require compiled navigation of native-selected context:
./scripts/spine-local.sh --terminal --triangle-internal-context
```

The guarded XPU launcher requires working user-systemd memory controls and the
matching Intel runtime. It enforces a 24 GiB host RAM cap, no swap and a 16 GiB
available-host-memory reserve. See the guide before running the large controls.

## Read the research

| To understand… | Read |
| --- | --- |
| The research question and interpretation | [Technical write-up](docs/writeup.md) and [technical note PDF](papers/qwenmem-spine-technical-note.pdf) |
| How to rebuild the paper | [Paper source and build instructions](papers/README.md) |
| Which work happens in Rust, Python or model state | [Architecture and terminology](docs/architecture.md) |
| What passed, what failed and what each score means | [Results](docs/results.md) |
| How internal context is gated and validated | [Internal context validation](docs/internal-context-validation.md) |
| How to set up and run each experiment | [Reproduction](docs/reproduction.md) |
| The downloadable checkpoint and its required runtime | [Model card](docs/model-card.md) |
| Where the system remains unreliable | [Known limitations](docs/known-limitations.md) and [Unicode NLI diagnostic](docs/unicode-nli-diagnostic.md) |

## Source layout and licenses

| Path | Purpose |
| --- | --- |
| `src/triangle_memory/` | Controller, reference, merged model, persistence and scoring. |
| `spine-harness/` | Synthetic native-memory evaluation against the included Rust source. |
| `rust-spine/` | Rust Spine application, optional Python backend and upstream licenses. |
| `scripts/`, `tests/`, `proofs/`, `examples/` | Build, evaluation, checks and synthetic fixtures. |
| `models.lock.json`, `requirements.lock.txt` | Pinned model/compiler inputs and tested Python dependencies. |
| `evidence/`, `docs/` | Aggregate measurements, provenance and explanation. |
| `papers/` | Technical note PDF and reproducible paper source. |

New contributions use MIT. Rust Spine retains MIT OR Apache-2.0; Qwenputer and
adapted compiler/runtime code retain MIT; Qwen base weights retain Apache-2.0.
See [THIRD_PARTY.md](THIRD_PARTY.md) and the accompanying notices for component
scope. Installing dependencies or building a checkpoint does not erase their
license obligations.
