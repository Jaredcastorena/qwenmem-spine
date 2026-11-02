# Known limitations

## String-valued claims can bypass grounding

Reported upstream as [Rust-spine issue #40: Grounding: string-valued factual
claims can bypass NLI verification and repair](https://github.com/Jaredcastorena/Rust-spine/issues/40).
Status: open; a fix is deferred.

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
