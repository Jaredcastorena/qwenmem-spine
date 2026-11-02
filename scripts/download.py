"""Fetch pinned compiler source and explicitly selected model assets (no Git checkout)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {"base": "models/qwen3.5-4b", "filesystem": "models/qwenputer-filesystem",
               "embedding": "models/minilm", "nli": "models/nli-minilm"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compiler-only", action="store_true")
    mode.add_argument("--models", nargs="+", choices=[*MODEL_NAMES, "tokenizer", "all"],
                      help="Explicit opt-in; base and filesystem each download multi-GB weights")
    args = parser.parse_args()
    lock = json.loads((ROOT / "models.lock.json").read_text())
    compiler = lock["compiler"]
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    archive = vendor / (compiler["revision"] + ".tar.gz")
    if not archive.exists():
        urllib.request.urlretrieve(
            f"https://codeload.github.com/{compiler['repo']}/tar.gz/{compiler['revision']}", archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != compiler["archive_sha256"]:
        raise ValueError("Compiler archive checksum mismatch")
    if not (vendor / "qwenputer").exists():
        with tarfile.open(archive) as source:
            source.extractall(vendor, filter="data")
        (vendor / ("qwenputer-" + compiler["revision"])).rename(vendor / "qwenputer")
    selected = set(args.models or [])
    for item in lock["models"]:
        alias = next(k for k, v in MODEL_NAMES.items() if v == item["directory"])
        full = alias in selected or "all" in selected
        tokenizer = alias == "base" and "tokenizer" in selected
        if not full and not tokenizer:
            continue
        files = item.get("files", []) if full else ["config.json", "tokenizer*", "chat_template*"]
        subprocess.run([str(Path(sys.executable).with_name('hf')), "download", item["repo"],
                        *files, "--revision", item["revision"], "--local-dir", str(ROOT / item["directory"])],
                       check=True)


if __name__ == "__main__":
    main()
