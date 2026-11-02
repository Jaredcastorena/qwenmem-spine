# Licenses and upstream provenance

New qwenmem-spine code is MIT licensed; see [LICENSE](LICENSE). The unmaintained research-release notice describes maintenance and intended audience, and does not restrict uses allowed by the licenses.

## Rust Spine

The `rust-spine/` directory is an adapted source snapshot of [Rust Spine](https://github.com/Jaredcastorena/Rust-spine), based on commit `07857378505d2573b623da4f81a892bfb9b5f2cd`. It retains its **MIT OR Apache-2.0** licensing and copyright notices. Its dependency licenses, including vendored Candle components, remain in that directory. Changes add the triangle model backend/provider, native memory context-mode controls, checkpoint/session handling, documentation and tests. This repository does not import the upstream Git history; the exact source provenance is recorded here.

## Qwenputer

[Qwenputer](https://github.com/divagr18/qwenputer) is a separately downloaded, unchanged compiler dependency pinned at `01cf657d353bdc0c795c2d57a28a3e3f483f20a1` (archive SHA-256 `90808cfa03ac0f0ec7c5883985b8d4b49a456870d695204414d940ee5960984a`). Its MIT license, copyright 2026 Divyansh Agrawal, is included in `licenses/Qwenputer-MIT.txt`. The merge/dispatch implementations adapt patterns from `experimental/riscv/chipformer/rv/merge.py` and `rv/chat.py`. The Rust backend also carries its own attribution and upstream license copy.

## Models and other dependencies

The base Qwen/Qwen3.5-4B checkpoint and the comparison-only divagr1925/Qwen3.5-4B-Qwenputer-Filesystem checkpoint are Apache-2.0 licensed. The merged triangle checkpoint modifies the base Qwen weights by adding a compiled controller prefix, with associated configuration changes. Preserve the Apache-2.0 license and modification notice with derivative weights; the new-code MIT license does not replace those terms. The Apache license is included in `licenses/Qwen-Apache-2.0.txt`.

The embedding model sentence-transformers/all-MiniLM-L6-v2 and grounding model cross-encoder/nli-MiniLM2-L6-H768 are also Apache-2.0 licensed. Their exact revisions and Qwen revisions are in `models.lock.json`. Downloaded models keep their upstream notices. No model weights are bundled in this source repository.

PyTorch, Transformers, safetensors, NumPy, Bend2 and other dependencies retain their own licenses. Python dependency pins are in `requirements.lock.txt`; Rust licenses and inventory are in `rust-spine/THIRD_PARTY_NOTICES.md` and `rust-spine/THIRD_PARTY_LICENSES.html`.
