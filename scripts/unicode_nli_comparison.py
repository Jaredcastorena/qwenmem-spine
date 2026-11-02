#!/usr/bin/env python3
"""Replay a synthetic 2x2 NLI comparison through the unchanged native grounding gate.

Uses local MiniLM NLI weights on CPU. No language-model generation or saved chat logs.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/unicode-nli-input.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.input.read_text())
    (output / "input.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    subprocess.run([str(ROOT / "artifacts/spine-target/debug/examples/unicode_nli_probe"),
                    str(ROOT / "models/nli-minilm"), str(output / "input.json"),
                    str(output / "results.json")], check=True)
    result = json.loads((output / "results.json").read_text())
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / "models/nli-minilm/tokenizer.json"))
    tokenizer.no_truncation()
    for row in result["rows"] + result["supplemental"]:
        lengths = [len(tokenizer.encode(h, e).ids) for h in row["hypotheses"] for e in row["evidence"]]
        row["all_candidate_pair_tokens_before_native_truncation"] = lengths
        row["all_candidate_pairs_fit_native_limit"] = all(n <= 512 for n in lengths)
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("NLI comparison complete; all measured decisions are retained in results.json")


if __name__ == "__main__":
    main()
