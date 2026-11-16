# Conversation dataset viewer

Open `tools/dataset_viewer.html` in a browser and drop a JSONL/JSON file onto it.
There are no dependencies, external assets, or network requests. Files are read
in the browser; nothing is uploaded. The viewer does not edit training files.

Ready-to-open snapshots of the current curated data:

- `outputs/dataset-viewer/retail-sft-curated.html`: 199 training examples.
- `outputs/dataset-viewer/retail-validation.html`: 10 validation examples.

Use the conversation list, search box, task/type filters, or Previous/Next and
arrow keys. System messages, tool results and working memory collapse by default.
Tagged `<tool_call>`, `<tool_response>` and `<memory>` blocks render as formatted
JSON where valid. Native structured tool calls are supported too. Exact raw
messages remain available in a collapsible section.

If a record has `loss_message_indices`, the viewer marks those messages as
training targets and all other messages as masked context. Without that metadata,
it does not claim to know which messages receive loss. Targets-only hides
context; turn it off to read the complete conversation. Source indices and saved
token counts are shown when provided. The viewer does not calculate token counts.

Create another standalone embedded snapshot:

```bash
.venv/bin/python scripts/export_dataset_viewer.py PATH/TO/dataset.jsonl \
  --output outputs/dataset-viewer/my-dataset.html
```

Supported input: one messages-array record per JSONL line, a JSON array of
records, one saved episode with a messages array, a JSON array of messages, or a
single ShareGPT conversations-array record. Content is rendered as text, never
executable HTML. Invalid JSONL lines produce an explicit error.

If preferred, serve the snapshots locally:

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \
  --directory outputs/dataset-viewer
```

Then open `http://127.0.0.1:8765/retail-sft-curated.html` (forward port 8765 if the
workspace is on a remote machine).
