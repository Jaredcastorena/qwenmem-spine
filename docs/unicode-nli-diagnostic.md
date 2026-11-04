# Exact Unicode diagnostic: extraction and evidence windows

Measured November 2, 2026 UTC using the real Rust claim extractor, MiniLM NLI
model and grounding predicate. The saved failing answer is replayed for the
historical measurements below; no language-model repair is generated. The host
defect is tracked in [Rust-spine issue #40](https://github.com/Jaredcastorena/Rust-spine/issues/40)
and is not fixed in this snapshot.

## November 3 internal-context follow-up

The [internal-context application matrix](internal-context-validation.md) adds
real compiled navigation to opt-in chat; it does not rerun or replace the NLI
measurements below. Triangle-only automatic memory scored 12/14 and still
fabricated the exact literal and citation despite passing its navigation gate.
Separately instructed recall recovered that literal with its original citation;
the full assisted set scored 3/3. The zero-claim extraction defect remains
deferred. Successful traversal authorizes selected source excerpts and cannot
establish exact-string fidelity or recover omitted evidence by itself.

## Fixed case and method

The question asks for the exact Zircon recovery label, preserving Unicode, case
and punctuation. The failing answer asserts `ZIRCON_RECOVERY_Ω`; the original
source specifies `Δ-sable_47/µ`. The original answer also contains a fabricated
citation. Citation authenticity is not independently evaluated by the forced
claim's NLI pair.

Rows A/B pass the original answer through `GroundingGate::verify`. Rows C/D bypass
only extraction by supplying one `AtomicClaim` with the *incorrect* value to the
same verifier and repair predicate. Its structured hypothesis is:

```text
The exact Zircon recovery label was equal to ZIRCON_RECOVERY_Ω.
```

Coverage threshold is 0.650, contradiction threshold 0.500 and evidence fanout
three. The original repair predicate is:

```text
claim_count > 0 AND (coverage < 0.650 OR contradiction >= 0.500)
```

Native pair order remains `(hypothesis, evidence)`. The short-source bound is
1,000 Unicode characters. The original event is only **108 characters / 110
UTF-8 bytes**, so short and full evidence are byte-identical. Each forced pair
is 56 tokens, below the NLI tokenizer's 512-token limit.

## Four-cell result

| Row | Extraction | Evidence | Claims | NLI pairs | Coverage | Contradiction | Repair |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| A | Original | Up to 1,000 characters | 0 | 0 | Not evaluated | Not evaluated | No |
| B | Original | Complete event | 0 | 0 | Not evaluated | Not evaluated | No |
| C | Forced wrong claim | Up to 1,000 characters | 1 | 1 | 0.01531738 | 0.12873006 | Yes |
| D | Forced wrong claim | Complete event | 1 | 1 | 0.01531738 | 0.12873006 | Yes |

Empty native reports carry default zero-valued numeric fields; they are not NLI
model scores. The aggregate JSON preserves null/not-evaluated for A/B. C/D request
repair because coverage is low, not because contradiction crosses its threshold.
The comparison confirms an extraction/gate bypass. It cannot measure truncation,
since the evidence windows are identical.

The actual failing request supplied six native triangle excerpts but omitted
both the correct label and its source event ID. None of those excerpts was
1,000 characters long. With that selected evidence, the original extractor again
yields no claims/comparisons/repair. Forcing the wrong claim yields three selected
pairs, coverage **0.12987405**, maximum contradiction **0.10546893**, and repair.
This is a source-selection miss plus an extraction bypass, not truncation of the
correct source. Older tool results visible elsewhere in a session are not
automatically part of this request's evidence.

## Exact-literal and pair-order controls

Forcing the **correct** value against the complete event also triggers repair:
coverage 0.00584489 and contradiction 0.00207689. Fixing extraction alone therefore
does not establish reliable literal verification or successful answer repair.

| Hypothesis/control | Pair order | Entailment | Contradiction |
| --- | --- | ---: | ---: |
| Incorrect literal | Hypothesis, evidence | 0.01531738 | 0.12873006 |
| Incorrect literal | Evidence, hypothesis | 0.73509771 | 0.14501232 |
| Correct literal | Hypothesis, evidence | 0.00584489 | 0.00207689 |
| Correct literal | Evidence, hypothesis | 0.84609514 | 0.00481583 |
| Source against itself | Identical texts | 0.97855455 | 0.00451711 |

Reversal materially changes the scores and also raises the wrong literal above
the coverage threshold. It is not a safe exact-string fix. The four-cell test
does not reverse pair order. The raw aggregate measurements are in
[nli-diagnostic.json](../evidence/nli-diagnostic.json).

## Reproduction scope

The repository supplies a minimal synthetic fixture at
[examples/unicode-nli-input.json](../examples/unicode-nli-input.json), the
[native diagnostic](../rust-spine/crates/spine-cli/examples/unicode_nli_probe.rs)
and a [runner](../scripts/unicode_nli_comparison.py). The diagnostic includes the
application's grounding implementation and only adds a wrapper for forced claims.

After dependency/model provisioning and native build:

```bash
./scripts/guarded-python.sh scripts/unicode_nli_comparison.py \
  --output artifacts/unicode-nli/reproduction
```

The included minimal answer omits the original fabricated citation and retains
the unsupported string claim. It supports a fresh A–D extraction/window test.
Its supplemental evidence contains the source event only, so it is **not** an
exact replay of the historical six-excerpt request. Historical aggregates and
new output should be labeled separately. This command loads the CPU NLI model;
it does not load the 4B checkpoint or generate an LLM repair. A fresh NLI run was
not performed during source packaging.

The historical CPU measurements peaked at 697,040,896 charged bytes for A–D and
683,778,048 for pair-order controls, with no recorded memory events. Those are
measurements for the tested host, not portable upper bounds.
