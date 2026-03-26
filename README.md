# AI Model Explorer

AI Model Explorer is a Textual-based terminal application for discovering GGUF models, checking hardware fit, and managing downloads from Ollama and Hugging Face.

## Features

- Search Ollama registry and Hugging Face GGUF repositories
- Estimate hardware fit based on live RAM/VRAM availability
- Queue, monitor, cancel, and delete downloads through a background service
- Track recent download history with status and details
- Filter by provider, use case, and hidden-gem models

## Requirements

- Python 3.10+
- Internet access for provider searches
- Ollama (optional; required only for local runtime and `ollama pull/run`)

## Installation

```bash
git clone https://github.com/kefyusuf/llm-terminal
cd llm-terminal

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Or with the installed script entrypoint:

```bash
ai-model-explorer
```

## Optional Hugging Face token

Use a read-only token for higher rate limits:

```env
AIMODEL_HF_TOKEN=hf_your_token_here
```

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `/` | Focus search |
| `p` | Cycle provider filter |
| `r` | Refresh current search |
| `h` | Toggle hidden gems |
| `q` | Quit |

## Project structure

```text
llm-terminal/
  app.py                # Main Textual app
  main.py               # Entry point
  cli.py                # Utility CLI commands
  download_service.py   # Background download service
  download_manager.py   # Download command builder
  providers/            # Provider integrations
  tests/                # Test suite
```

## Notes

- `terminal_ui/` is kept as a legacy experimental package and is not the primary product path.
- Download service data is stored in `downloads.db`.
- Metadata cache is stored in `cache.db`.
