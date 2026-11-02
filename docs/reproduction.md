# Reproducing the research snapshot

Run these commands from the repository root. The source snapshot includes no
weights or generated stores/caches. The commands below provision those inputs
explicitly. They describe the supported research workflow, not a validated
clean-machine installation or a promise of identical language-model outputs.

## Environment and pins

The measured environment used Linux, Python 3.12.3, Rust 1.99.1, PyTorch
2.14.1+xpu, Transformers 5.19.0, safetensors 0.8.0 and NumPy 2.5.4, with one
Intel Arc Pro B70 (32 GB device memory). Small controller checks also run on CPU.
Full-model CUDA, macOS and Windows execution have not been verified. The included
Rust toolchain and Cargo lock files pin the native build; Python pins are in
[requirements.lock.txt](../requirements.lock.txt).

[models.lock.json](../models.lock.json) pins the base model, comparison model,
compiler, MiniLM encoder and NLI model. The compiler revision is
`01cf657d353bdc0c795c2d57a28a3e3f483f20a1`; its archive SHA-256 is
`90808cfa03ac0f0ec7c5883985b8d4b49a456870d695204414d940ee5960984a`.
The supplied downloader checks that archive and requests explicit model revisions.

## Small CPU controller

Bootstrap installs dependencies only. It does not fetch model weights or compile
the merged model. Choose `cpu` for the small controller or `xpu` for the tested
Intel full-model path; both use the same Python source.

```bash
./scripts/bootstrap.sh cpu
.venv/bin/python scripts/download.py --compiler-only
./run.sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_internal_triangles.py \
  --state-only --output artifacts/controller-state.json
```

`./run.sh` defaults to a small real compiled BF16 circuit on CPU using
[examples/synthetic-forest.json](../examples/synthetic-forest.json). It opens a
branch, saves state, discards the session and restores an alternate route. It
does not load the 4B language model. The state-only audit blocks host forest
lookups after installation, verifies traversal, clears resident records to test
invalid access and restores state. It does not compare the 4B checkpoint prefix.

The default pytest suite skips the optional 4B test and tests whose generated
native fixtures are absent. A passing reduced suite does not imply those paths
ran. Source-preparation checks recorded 20 passes/seven skips, a passing
controller demo and state audit, and 26 passing native harness tests against the
included Rust source. No clean network install or new 4B benchmark was performed.

## Native build and synthetic fixtures

The harness is wired to the included `rust-spine/` tree using relative paths.
No separate checkout or source-rewriting configuration step is needed. Install
the small model inputs and build the application, harness, geometry binary and
Unicode diagnostic:

```bash
.venv/bin/python scripts/download.py --models tokenizer embedding nli
./scripts/build-native.sh
./scripts/make-native-fixtures.sh
```

`tokenizer` downloads the pinned base tokenizer/config without the 4B weights;
the corpus builders use it to measure actual content tokens. `embedding` and
`nli` fetch their small model snapshots. Build uses the pinned Cargo locks and
defaults to two build jobs. On a machine with populated dependency caches,
`./scripts/build-native.sh --offline` forbids Cargo network access.

Fixture generation creates the primary corpus, boundary corpus, native memory
exports and geometry fixture under `artifacts/retention/`. The native harness
refuses to overwrite an existing store. Use a fresh copy of the source or fresh
output paths when rerunning its lower-level binary. These fixtures are
synthetic; no existing user heart is required. Native generation uses CPU
embeddings, not 4B language generation.

Once fixtures exist, the renderer-loss regressions are expected failures rather
than skips, and native trace/geometry checks become available:

```bash
.venv/bin/python -m pytest -q
CARGO_TARGET_DIR=artifacts/spine-target CARGO_BUILD_JOBS=2 \
  cargo test --manifest-path spine-harness/Cargo.toml --locked --offline
```

The two strict expected failures retain the known older-event/source-tail
renderer losses. Do not count them as successful recall or silently weaken them
to make an all-green result.

## Merged checkpoint: download or build

The model destination is
[Jared1728/qwenmem-spine](https://huggingface.co/Jared1728/qwenmem-spine).
An authenticated account with access is needed while the repository is private.
Download a fixed model revision into the runtime's expected directory:

```bash
.venv/bin/hf download Jared1728/qwenmem-spine \
  --revision f69dbcfe5a7cebea1c287f6f1d76c8e90b2c9a98 \
  --local-dir models/qwen3.5-triangles
```

This is the immutable model publication revision recorded for this snapshot;
its uploaded payload sizes and hashes were verified against the staged files.
The runtime verifies weight-shard identities in `triangle.json`.
[checkpoint-manifest.json](../checkpoint-manifest.json) records the measured
17-file checkpoint set. The separate source repository and model repository
retain their respective license notices.

To rebuild instead, fetch the base weights and run the supplied merge:

```bash
.venv/bin/python scripts/download.py --models base
./scripts/guarded-python.sh -m triangle_memory.merge \
  models/qwen3.5-4b models/qwen3.5-triangles
```

The output directory must be suitable for a new generated checkpoint. This
compiles navigation and merges it with the text component; it does not train or
fine-tune on the evaluation questions. Allow substantial disk space for base and
merged weights, Python dependencies, compiler and any later state checkpoints.
`--models all` additionally downloads the large Qwenputer Filesystem comparison
checkpoint, which is unnecessary for this runtime or these reproductions.

## Guarded Intel full-model execution

Install the XPU environment with `./scripts/bootstrap.sh xpu` if the environment
was initially provisioned for CPU. A compatible Intel Level Zero/OpenCL runtime
must be installed. If it resides in a custom directory, `INTEL_RUNTIME_ROOT`
can name the layout expected by [guarded-python.sh](../scripts/guarded-python.sh);
otherwise the wrapper uses system libraries. It does not install drivers.

The wrapper requires a user-systemd manager with cgroup memory controls. It
enforces a 24 GiB host cap, 20 GiB throttle and no swap; startup requires 40 GiB
available host RAM. A watchdog stops the child below 16 GiB available memory,
and one exclusive model job runs at a time. CPU helper threads use selected
physical cores (`TRIANGLE_CPU_THREADS` defaults to four). The runtime/bounded
runner also caps the XPU allocator at 70%. Do not remove these controls to repeat
the historical large-context workload. Missing controls cause refusal rather
than an unbounded fallback.

Run the merged navigation demo and optional real-model checks:

```bash
./scripts/guarded-python.sh -m triangle_memory.cli --device xpu \
  --model models/qwen3.5-triangles demo --output artifacts/merged-demo
TRIANGLE_TEST_DEVICE=xpu ./scripts/guarded-python.sh \
  -m pytest tests/test_model.py tests/test_native_geometry.py -q
TRIANGLE_MERGED_TEST=1 TRIANGLE_TEST_DEVICE=xpu ./scripts/guarded-python.sh \
  -m pytest tests/test_merged.py -q
```

The model tests without `TRIANGLE_MERGED_TEST` use small real BF16 controller
models. The merged test additionally loads the 4B checkpoint, checks fresh-context
language recall and requires unchanged navigation state. Without `--state-only`,
`scripts/audit_internal_triangles.py` also compares the regenerated prefix with
the local merged checkpoint on CPU; it is not a new full-model language test.

## Full Rust agent and current acceptance

After native build and model provisioning:

```bash
./scripts/spine-local.sh --terminal
# Or launch the same agent through its browser interface:
./scripts/spine-local.sh --web
```

The convenience launcher creates a separate local encrypted heart, prompts for
its passphrase, and runs the application through the guard with local model
paths. It does not load saved remote-provider preferences. The direct path-based
configuration interface is documented in
[rust-spine/docs/triangle-model.md](../rust-spine/docs/triangle-model.md).

Generate the synthetic native fixtures first, then use new output directories:

```bash
./scripts/guarded-python.sh scripts/spine_end_to_end.py \
  --output artifacts/acceptance/native-mixed
./scripts/guarded-python.sh scripts/spine_end_to_end.py \
  --triangle-context-only --output artifacts/acceptance/triangle-only
./scripts/guarded-python.sh scripts/spine_end_to_end.py \
  --boundaries --output artifacts/acceptance/native-boundaries
./scripts/guarded-python.sh scripts/spine_end_to_end.py \
  --triangle-context-only --boundaries --output artifacts/acceptance/triangle-boundaries
./scripts/guarded-python.sh scripts/spine_end_to_end.py \
  --triangle-context-only --boundaries --force-recall \
  --output artifacts/acceptance/explicit-boundaries
./scripts/guarded-python.sh scripts/spine_end_to_end.py \
  --triangle-context-only --query literal --force-recall \
  --output artifacts/acceptance/explicit-literal
```

`--force-recall` changes the question and labels the report as an explicit-recall
probe. Keep that score separate. Current normal chat installs an empty compiled
forest and uses native context/tools; these commands do not test the unfinished
bounded internal-controller handoff. The recorded outcomes and exact failure
categories are in [results](results.md).

The CPU NLI reproduction is:

```bash
./scripts/guarded-python.sh scripts/unicode_nli_comparison.py \
  --output artifacts/unicode-nli/reproduction
```

It uses the included minimal synthetic fixture. Its supplemental evidence is
not the original six-excerpt failing request; see the
[diagnostic's scope](unicode-nli-diagnostic.md#reproduction-scope).

## Standalone retention and cross-process state

With the native fixtures, XPU environment and merged model ready:

```bash
./scripts/guarded-python.sh scripts/retention_benchmark.py \
  --native artifacts/retention/native-v2 --output artifacts/retention/reproduction
.venv/bin/python scripts/score_retention.py artifacts/retention/reproduction \
  --native artifacts/retention/native-v2
./scripts/guarded-python.sh scripts/native_session_roundtrip.py write
./scripts/guarded-python.sh scripts/native_session_roundtrip.py read
```

The last commands are two separate processes using the native 128K fixture and
the generated `artifacts/retention/native-session/` snapshot. The reader checks
the recovered alternate and scores the answer against the supporting source.

## Raw full-history control

The legacy runner is limited to 4K/32K stored-history stages and rejects raw
prompts over 45,000 tokens:

```bash
./scripts/guarded-python.sh scripts/full_context_control.py \
  --stages 4k 32k --output artifacts/retention/raw-small
.venv/bin/python scripts/audit_raw_controls.py --directory artifacts/retention/raw-small
```

For the large diagnostic, use the separate bounded runner with its enforced
guard and finite checks. It requires the generated primary native fixtures and
substantial checkpoint disk space. The unresolved earlier NaN/interruption
failures remain relevant even though later runs passed.

```bash
./scripts/guarded-python.sh scripts/bounded_full_context.py --stage 128k \
  --output artifacts/retention/raw-large --stop-after 65536
./scripts/guarded-python.sh scripts/bounded_full_context.py --stage 128k \
  --output artifacts/retention/raw-large \
  --resume artifacts/retention/raw-large/128k-cache-65536
.venv/bin/python scripts/audit_raw_controls.py --directory artifacts/retention/raw-large
```

The first process exits after a valid checkpoint. The second verifies and resumes
the same token history, without eviction, summarization or resetting recurrence.
Completed output/checkpoints are not overwritten. Only finite, complete answers
with independent source/citation checks count as valid raw controls. A later
fresh-process restore can use the saved full-prompt checkpoint with a new output
directory to compare answer bytes and token IDs.

## Abstract laws

With the separately installed Bend2 CLI, the five abstract selector/apex laws
can be checked with:

```bash
bend proofs/PROOF.bend --check-only
ELAN_TOOLCHAIN=leanprover/lean4:v4.34.0 bend proofs/PROOF.bend --verdict
```

The measured tool was Bend 2.0.38; the independent verifier used Lean 4.34.0.
These laws cover preferred/alternate branch selection and apex preservation,
not the compiler, runtime, complete traversal bounds or model answers.
