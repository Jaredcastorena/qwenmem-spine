#!/usr/bin/env python3
"""Add the Python backend, its notices and setup guide to a binary package."""
import argparse
from pathlib import Path
import shutil


def stage(destination):
    root = Path(__file__).resolve().parents[1]
    backend = Path("backends/triangle")
    files = [backend / name for name in (
        "server.py", "README.md", "NOTICE.md", "QWENPUTER-LICENSE.txt",
        "requirements-runtime.txt")]
    files += [p.relative_to(root) for p in sorted(
        (root / backend / "triangle_backend").glob("*.py"))]
    files.append(Path("docs/triangle-model.md"))
    if not (root / backend / "triangle_backend/backend.py").is_file():
        raise SystemExit("triangle backend source is missing")
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    print(f"Staged {len(files)} triangle runtime/doc files; models, compiler and Python packages are separate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    stage(parser.parse_args().destination)
