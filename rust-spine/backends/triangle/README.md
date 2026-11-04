# Local triangle model backend

This is the backend protocol reference. For setup and normal chat, start with
the snapshot's [reproduction guide](../../../docs/reproduction.md). Commands
below run from the included `rust-spine/` directory, with `python` pointing to
the provisioned environment.

Run with an existing Python environment, a merged triangle/Qwen3.5 checkpoint,
and the pinned Qwenputer compiler checkout named in [NOTICE.md](NOTICE.md):

```sh
python backends/triangle/server.py \
  --model-dir /path/to/merged-checkpoint \
  --compiler-dir /path/to/qwenputer \
  --device cpu
```

`--compiler-dir` also accepts the checkout's `experimental/riscv` directory.
The CPU command illustrates the interface; full-model execution was measured on
Intel XPU. Use `--device xpu` only with a configured supported XPU environment and a host memory guard. The backend caps its XPU allocator at 70% of device capacity. Model loading
is lazy; `hello` and `install` require only the Python standard library. A model or
device failure is returned as an error and terminates the backend. There is no
runtime reference or fake-model fallback. The process uses local files only.

Requests and responses are UTF-8 JSON lines on stdin/stdout; diagnostics go to
stderr. A request is `{"schema":1,"id":1,"op":"hello","args":{}}`.
Successful responses contain `ok:true,result:{...}`; errors contain
`ok:false,error:{code,message}`. Request IDs are unsigned 64-bit integers.

Operations are `hello`, `install`, `execute`, `save`, `restore`, `generate`, and
`shutdown`, plus `complete` for the normal agent. Install accepts `forest:{schema:1,nodes,triangles,roots}` and returns
the forest SHA-256 plus addresses sorted as nodes then triangles. Execute accepts
`forest_sha256` and `command` (`open:SLOT`, `peek`, `left`, `right`, `route:INDEX`).
Save and restore accept `forest_sha256` and `path`. Generation accepts `question`,
`evidence:[{node_id,event_id,text}]`, and `max_tokens`.

Execute returns the integer `trace`, `forest_sha256` and an `execution` object:
`engine`, `prefix_layers`, `ticks_before`, `ticks_after`, `state_before`,
`state_after`, `resident_records` and `resident_triangles`. The measured engine
is `merged_checkpoint_prefix` with five prefix layers. Rust independently checks
the expected trace, forest identity, address ordering and execution evidence
before authorizing source excerpts. A reported execution alone is not a claim
that a generated answer is correct.

The limits reported by `hello` are 2 MiB per wire line, 64 resident records, eight
descents, a 16 KiB question, 64 evidence records totaling 128 KiB of JSON, an
8,192-token prompt, and 1–512 output tokens. Empty forests allow generation but
reject navigation. Changed forest content resets navigation; identical installs
retain it. Snapshots bind forest, weights, protocol, tensor layout and checksums,
are atomically replaced, and have mode `0600`.

Native triangle coherence is enforced when triangles are created. Retained
centroids can subsequently evolve; installation validates structure, provenance,
dimensions and finite nonzero vectors without reapplying that creation-time
cosine threshold. The standalone research loader keeps its own coherence check.

Generation uses a fresh language cache and leaves navigation unchanged. It rejects
non-finite logits. `truncated:true` indicates exhaustion of the output budget.
Returned citations are exact supplied event/node IDs occurring in the answer;
this checks identifier provenance, not whether evidence supports each claim.

Standard-library protocol checks:

```sh
python -m unittest discover -s backends/triangle/tests -p 'test_backend.py' -v
```

CPU circuit/session and injected language checks, using an already provisioned
environment with the packages in `requirements-runtime.txt` and PyTorch:

```sh
TRIANGLE_COMPILER_DIR=/path/to/qwenputer python -m unittest discover \
  -s backends/triangle/tests -p 'test_runtime.py' -v
```

The latter compiles a small real circuit; it does not load a language checkpoint
or use a GPU. A real merged-checkpoint run remains a separate acceptance check.

## Full-agent completion

`complete` accepts the installed `forest_sha256`, OpenAI-shaped `messages` and
`tools`, `max_tokens` (1–2048), and finite `temperature` (0–2). It uses the pinned
Qwen chat template; Rust interprets its native XML-style function calls and applies
the existing tool controls. Prompts are limited to 32,768 tokens. Input is not
silently truncated. A fresh fixed-capacity KV cache uses 256-token prefill, fused
SDPA on XPU, numerical checks, and independent navigation-state comparison.

For local acceptance debugging only, `SPINE_TRIANGLE_TRACE_DIR` records model
responses in exclusive mode-0600 files. Adding `SPINE_TRIANGLE_TRACE_REQUESTS=1`
also records complete input requests, including private harness context and tool
results. These belong with private session evidence, not in source or model-weight
uploads. Normal runs omit both variables. Rust budgets supplied messages before
calling the backend; the backend itself rejects exact-token overflow.

## Native-selected internal context

Default Rust chat installs an empty navigation forest. With
`--triangle-internal-context`, the Rust provider uses the existing operations to
install and navigate each native-selected root's complete closure before
`complete`. Every bank must fit 64 combined node/triangle records and depth
eight; exceeding those limits fails explicitly, with no automatic paging.
Topology carries no source bodies. Original source excerpts remain in a host
sidecar, and only returned apex IDs can authorize their release into language
messages. Empty selections are labeled `no_memory` and are not successful
navigation evidence. Host semantic ranking, triangle construction, compaction
and source storage remain native operations.

Main-turn sidecars survive tools, repair/finish and harness checkpoints. Child
requests refresh their own native packet; onboarding remains explicitly without
recall. `--triangle-context-only` separately withholds automatic hybrid recall
while retaining explicit tools. See the [integration guide](../../docs/triangle-model.md)
for source budgets and lifecycle details.

The November 3, 2026 [validation record](../../../docs/internal-context-validation.md)
and [aggregate](../../../evidence/internal-context-20261103.json) report 83
focused CPU checks and actual Intel Arc Pro B70 acceptance: mixed 14/14,
triangle-only 12/14, boundaries 1/2 and separately assisted recall 3/3. All 70
completed model rounds passed the source gate across 210 banks, 267 repeated
triangle-record installations and 146 branches. Interrupted work is counted
separately. The full-chat zero-bank control blocked completion, while the
standalone zero/restore probe recovered exact trace and bank bytes. These checks
do not establish correct language output, unrestricted capacity, clean-machine
installation or support on other accelerators. Historical empty-forest scores
remain separately labeled, and NLI issue #40 remains deferred.
