#!/usr/bin/env python3
"""Run the local Rust Spine merged-model backend; all paths are explicit."""
import argparse
import os
from pathlib import Path
import sys

# Prefer this runtime even when launched by a guard with an existing PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from triangle_backend.backend import Backend, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--compiler-dir", required=True, type=Path,
        help="Qwenputer checkout or its experimental/riscv directory")
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return serve(Backend(model_dir=args.model_dir.resolve(), compiler_dir=args.compiler_dir.resolve(),
        device=args.device), sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
