"""Use an explicitly selected, already available Qwenputer compiler checkout."""
from pathlib import Path
import sys

COMMIT = "01cf657d353bdc0c795c2d57a28a3e3f483f20a1"
_source = None


def configure(directory):
    global _source
    root = Path(directory).resolve()
    for source in (root / "experimental/riscv", root):
        if (source / "chipformer/q35/ir.py").is_file() and (source / "chipformer/rv/hf.py").is_file():
            if _source is not None and source != _source:
                raise ValueError("Compiler directory cannot change within a process")
            _source = source
            return
    raise ValueError("Compiler directory must contain chipformer/q35/ir.py and chipformer/rv/hf.py, directly or under experimental/riscv")


def enable():
    if _source is None:
        raise RuntimeError("Configure the explicit compiler directory before loading the model")
    sys.path.insert(0, str(_source)) if str(_source) not in sys.path else None
    loaded = sys.modules.get("chipformer")
    if loaded is not None:
        paths = getattr(loaded, "__path__", [])
        if not any(Path(p).resolve() == _source / "chipformer" for p in paths):
            raise RuntimeError("A different chipformer package is already loaded")
