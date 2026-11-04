# Known limitations

## Internal-context navigation and answer limits

The November 3, 2026 [validation record](../../docs/internal-context-validation.md)
measures the opt-in `--triangle-internal-context` mode. Its compiled source gate
passed in all 70 completed model rounds, but triangle-only automatic memory
scored 12/14: exact literal/citation fabrication and a denied write to an
incorrect absolute path remain failures. Boundary recovery scored 1/2 because
the long-source tail was omitted; separately instructed recall scored 3/3.
Mixed mode scored 14/14 while also receiving automatic hybrid evidence, so that
score does not isolate triangle excerpts as the reason for success.

Default chat retains an empty navigator forest. Internal mode keeps native
semantic selection, triangle construction, compaction and source storage on the
host. Complete selected closures above 64 records or depth eight fail explicitly;
there is no automatic paging. An actual zero-bank full-chat control was rejected
before completion. That verifies the source gate's failure path, not answer
entailment, unlimited memory or recovery of unselected source text.

## String-valued claims can bypass grounding

Reported upstream as [Rust-spine issue #40: Grounding: string-valued factual
claims can bypass NLI verification and repair](https://github.com/Jaredcastorena/Rust-spine/issues/40).
Status: open; a fix is deferred.

The new internal-context tests do not fix this defect. Their fabricated
literal/citation failure and the earlier
[Unicode diagnostic](../../docs/unicode-nli-diagnostic.md) remain separate
evidence for the boundary between successful navigation and answer validation.

The claim extractor can miss factual assertions about string identifiers,
including ASCII and Unicode labels. When it returns zero claims, NLI inference
is skipped and the current grounding gate does not request repair. An
unevaluated report is not evidence that an answer was verified. Supplying more
source text does not fix this extraction path.

Grounding therefore does not guarantee exact identifier, case, punctuation or
Unicode fidelity. Check important exact values against original source text.
The issue is not fixed in this unmaintained source snapshot. See the snapshot's
[full limitations](../../docs/known-limitations.md) for navigation, evaluation and
runtime boundaries.
