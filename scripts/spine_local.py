"""Guarded local launch of the complete Rust Spine agent with the merged model."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
binary=ROOT/'artifacts/spine-target/debug/spine'
if not binary.is_file():
    raise SystemExit('Build first: cd rust-spine && CARGO_TARGET_DIR=../artifacts/spine-target cargo build -p spine-cli --locked --offline')
state=ROOT/'local-spine'
state.mkdir(mode=0o700,exist_ok=True)
# A local launch must not inherit remote endpoint defaults from this shell.
for variable in ('SPINE_LLM_URL','SPINE_LLM_MODEL','SPINE_LLM_API_KEY',
        'SPINE_LLAMA_MODEL','SPINE_LLAMA_SERVER','SPINE_REASONING_EFFORT','SPINE_MAX_CONTEXT_TOKENS'):
    os.environ.pop(variable, None)
# No saved remote-provider preferences are read or written. A dedicated local heart
# is used by this convenience launcher; the direct CLI accepts another heart path.
arguments=sys.argv[1:]
cmd=[str(binary),'--no-settings','chat',str(state/'heart.spine'),
    '--workspace',str(ROOT),'--model-dir',str(ROOT/'models/minilm'),
    '--nli-model-dir',str(ROOT/'models/nli-minilm'),'--no-download',
    '--triangle-backend',str(ROOT/'.venv/bin/python')]
for arg in [str(ROOT/'rust-spine/backends/triangle/server.py'),'--model-dir',str(ROOT/'models/qwen3.5-triangles'),
            '--compiler-dir',str(ROOT/'vendor/qwenputer'),'--device','xpu']:
    cmd.append('--triangle-backend-arg='+arg)
cmd.extend(arguments)
os.execv(binary,cmd)
