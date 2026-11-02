This backend adapts the project's local triangle-memory prototype. The circuit,
forest validation, protocol, and session integration are project code. The merged
runtime's language/circuit dispatch and staged XPU transfer follow Qwenputer's
`experimental/riscv/chipformer/rv/chat.py`, at commit
`01cf657d353bdc0c795c2d57a28a3e3f483f20a1`:
<https://github.com/divagr18/qwenputer/tree/01cf657d353bdc0c795c2d57a28a3e3f483f20a1>.

Qwenputer is MIT licensed, copyright (c) 2026 Divyansh Agrawal. Its license is
retained in [QWENPUTER-LICENSE.txt](QWENPUTER-LICENSE.txt). No compiler source,
model checkpoint, tokenizer, or third-party package is bundled here. Supply the
compiler and merged Qwen3.5 checkpoint explicitly; their licenses remain with
their distributions. This runtime does not download dependencies or model files.
