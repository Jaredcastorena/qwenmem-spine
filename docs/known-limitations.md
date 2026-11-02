# Known limitations

This fixed research snapshot will not receive updates, bug fixes, compatibility
work or ongoing support. Its licenses permit reuse under their terms; research
intent is not a commercial-use prohibition.

## Internal controller scope

The controller supports 64 combined resident nodes/triangles and eight descents.
The resident table is installed externally and remains read-only during
navigation. Semantic ranking, coherent-apex selection, triangle creation/folding,
chronology, source storage and evidence selection stay on the host. There is no
resident paging or autonomous language-model route policy. Current normal chat
installs an empty navigation forest. Its bounded native-selected handoff is
unfinished, and standalone navigation results cannot establish that handoff.

Native recursive storage can exceed the compiled table. A compact outer handle
does not make durable storage free, and a small language prompt does not account
for model weights, dense navigation state or source bodies.

## Source selection and automatic recovery

The historical native renderer can omit older still-binding events by selecting
only two recent events per node and can omit source tails after 1,000 characters.
Both are strict expected-failure regressions when their generated fixtures exist.
Current `heart_recall` can recover the measured 2,320-character source-tail case
because its prefix budget is 3,000 characters. It has no event continuation
offset, so this is not arbitrary-length paging.

Automatic triangle-only application behavior passed 11/14, with fabricated
literal/citation, wrong-question repair and invented tool path failures. Both
native mixed-memory boundary answers contained the right values but omitted
required original citations, scoring 0/2. Triangle-only boundaries scored 1/2.
Explicitly requested recall passed the three selected probes; assistance changes
the task and does not erase the original failures.

## Grounding: Rust-spine issue #40

The baseline claim extractor misses some factual string identifiers. Zero
extracted claims suppress NLI and repair even when an answer is false and the
original source is available. The defect is tracked in
[Rust-spine #40](https://github.com/Jaredcastorena/Rust-spine/issues/40); its fix
is not included here. Forcing extraction triggers repair for the wrong Unicode
literal, but the correct literal also fails coverage. Reversing the NLI pair
order raises both scores and would incorrectly accept the wrong value in the
diagnostic. No production fix or successful generated repair is claimed.

The defect belongs to host validation. It does not show a disconnected model
provider, and it does not explain every automatic-answer/tool failure. Citation
presence alone is not entailment, and NLI is not a byte-exact string comparator.
See the [four-cell diagnostic](unicode-nli-diagnostic.md).

## Numerical behavior and resources

The standalone Python [forest loader](../src/triangle_memory/forest.py) checks
finite coordinate values but can overflow while calculating cosine similarity.
For example, opposing vectors with magnitudes around `1e308` can produce NaN,
which slips past its `< 0.8` rejection. Its coherence validation is therefore
not reliable for arbitrary finite numeric inputs. The native backend rejects
non-finite norms; this edge case was not an observed failure of the measured
MiniLM fixtures. It is separate from the full-model NaN behavior below.

Two early raw-context attempts were interrupted without complete answers. Their
exact causes are unresolved. A later full-input bounded run developed NaN
recurrent state/logits; its punctuation output is an invalid numerical control,
not a legitimate recall score. A separate checked replay and cache restore passed
8/8, but changed validation/synchronization do not isolate or permanently fix the
earlier trigger. Finite checks now stop invalid output and reject bad checkpoints.

Full-model measurements used one Intel Arc Pro B70 with Linux/PyTorch XPU. CUDA,
Windows and macOS full-model operation are unverified. RAM/cgroup counters and
device allocator measurements do not cover every driver or shared-cache
allocation. The guarded launcher requires working user-systemd controls and
refuses to run without the required memory limits. Clean-machine installation
has not been validated.

## Evaluation and artifact limits

The corpus is synthetic, with eight fixed questions and 16 distraction templates.
It does not measure broad task accuracy, long-term human use, wall-clock aging,
concurrent users or key recovery. Session labels are logical threads, with three
actual store reopen checkpoints. Raw controls answer all questions together;
retrieval controls answer one at a time, preventing a direct timing comparison.
Oracle evidence is an upper-bound baseline, not a deployable selection policy.

The merge is compiled, not trained on the questions, and its BF16 padding and
normalization can change base-model behavior. It is text-only and requires the
custom dispatch/runtime. Ordinary hosted inference and generic Transformers
generation are not validated substitutes. Source includes no weight binaries,
encrypted stores or runtime state. Snapshot integrity checks do not authenticate
adversarially replaced data and metadata. Abstract Bend laws do not prove the
compiler, runtime, model behavior or retrieval quality.
