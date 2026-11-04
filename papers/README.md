# Technical note

[Read the PDF](qwenmem-spine-technical-note.pdf): *qwenmem-spine: Compiled Navigation Layers for Persistent Evidence Retrieval*, revised November 4, 2026. Measurements date from October 31 through November 3, 2026. This is a research note, not an arXiv submission.

The first ordinary-chat battery used an empty compiled forest with host-supplied context. The second battery enabled `--triangle-internal-context`, verifying the actual compiled navigation path across 70 completed rounds and 210 bank executions. The flag remains opt-in. The paper retains the measured failures and distinguishes navigation correctness from answer quality.

The editable content is [technical_note_content.json](technical_note_content.json); [create_technical_note.py](create_technical_note.py) renders it with ReportLab and the DejaVu Serif/Sans/Mono fonts. The checked PDF was built with Python 3.12 and ReportLab 4.4.9:

```bash
python3 -m venv /tmp/qwenmem-paper-venv
/tmp/qwenmem-paper-venv/bin/pip install reportlab==4.4.9
/tmp/qwenmem-paper-venv/bin/python papers/create_technical_note.py
```

On systems without `/usr/share/fonts/truetype/dejavu`, install the DejaVu font family and pass its directory with `--fonts`. Use `--output /path/to/note.pdf` to select another output. The script does not load the model or run inference. PDF metadata may vary between builds.

See [results](../docs/results.md), [internal-context validation](../docs/internal-context-validation.md), and the [machine-readable November 3 evidence](../evidence/internal-context-20261103.json). That evidence preserves its measurement-time source hashes and uncommitted status. The November 4 release additionally fixes the runner's final exit check for shortened runs; no new model measurements are claimed.
