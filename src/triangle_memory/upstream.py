"""Load the pinned, locally vendored compiler without installing or modifying it."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor/qwenputer/experimental/riscv"
COMMIT = "01cf657d353bdc0c795c2d57a28a3e3f483f20a1"


def enable():
    if not (SOURCE / "chipformer/q35/ir.py").is_file():
        raise RuntimeError("Pinned Qwenputer source is missing; run .venv/bin/python scripts/download.py --compiler-only")
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))

