# qwenmem-spine

**Model weights:** [Jared1728/qwenmem-spine on Hugging Face](https://huggingface.co/Jared1728/qwenmem-spine).
See the [pinned download and setup instructions](docs/reproduction.md#merged-checkpoint-download-or-build).

An experimental memory system that combines Rust Spine's durable memory and
agent tools with a modified Qwen3.5-4B language model. Five added model layers
execute a small navigation program over a hierarchy of memory records.
Original source text stays outside the model, in Spine's encrypted memory store
(the **heart**) or the standalone tests' source fixtures.

There are two execution paths: standalone tests exercise the compiled navigator;
normal chat uses Spine's native retrieval and tools with the modified language
model. **Normal chat does not yet use the compiled navigator.** This repository
preserves both paths as a fixed research source snapshot.

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
- Synthetic evaluation builders, scoring/audit code, tests, abstract selector
  proofs and sanitized aggregate [evidence](evidence/provenance.json).

**The current normal-chat path installs an empty internal navigation forest.**
It uses Spine's native context and memory tools. The standalone compiled
navigator works, but a bounded native-selected handoff to that navigator is
unfinished. Triangle creation, semantic ranking and compaction remain host
operations. This is not a full internal triangle implementation or a claim of
unlimited, lossless context.

## Measured results

| Evaluation | Recorded result | Scope |
| --- | --- | --- |
| Current native mixed-memory application | 14/14 operational assertions | Actual checkpoint, real Rust harness, tools, restart and interruption. |
| Current triangle-only automatic memory | 11/14 | Three failures retained; explicit memory tools remain available. |
| Separate explicitly requested native recall | 3/3 | Two boundary cases and one Unicode literal; assisted probes, not replacement scores. |
| Earlier standalone retention | 8/8 at each of 4,274, 32,812 and 131,215 stored content tokens | 572–901-token retrieval prompts; two additional source-selection failures. |
| Valid full-history control | 8/8 at 167,780 input tokens, repeated after cache restore | A separate earlier run developed NaNs; its cause remains unresolved. |

The fixed synthetic questions and operational assertions are not population
accuracy estimates. The full [results and failure record](docs/results.md)
separate current behavior from earlier provider experiments. The known host
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
```

The guarded XPU launcher requires working user-systemd memory controls and the
matching Intel runtime. It enforces a 24 GiB host RAM cap, no swap and a 16 GiB
available-host-memory reserve. See the guide before running the large controls.

## Read the research

| To understand… | Read |
| --- | --- |
| The research question and interpretation | [Technical write-up](docs/writeup.md) |
| Which work happens in Rust, Python or model state | [Architecture and terminology](docs/architecture.md) |
| What passed, what failed and what each score means | [Results](docs/results.md) |
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

New contributions use MIT. Rust Spine retains MIT OR Apache-2.0; Qwenputer and
adapted compiler/runtime code retain MIT; Qwen base weights retain Apache-2.0.
See [THIRD_PARTY.md](THIRD_PARTY.md) and the accompanying notices for component
scope. Installing dependencies or building a checkpoint does not erase their
license obligations.
