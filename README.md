# AI Model Explorer (TUI)

AI Model Explorer is a Textual-based terminal UI that helps you find local-LLM models from Ollama and Hugging Face, then quickly estimate whether a model will fit your current hardware.

## Features

- Search models from both Ollama and Hugging Face GGUF listings.
- Show system snapshot: CPU, RAM, GPU VRAM, and Ollama process status.
- Estimate run fit (`Perfect`, `Partial`, `Slow`, `No Fit`) based on available memory.
- Filter results by source (`All`, `Ollama`, `Hugging Face`).
- Open row details in a modal with a ready-to-run command.

## Requirements

- Python 3.10+
- Optional NVIDIA GPU for VRAM detection
- Ollama (optional, for local model detection)

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Run

```bash
python main.py
```

## Run Tests

```bash
pytest -q
```

Run live platform checks (no mocks):

```bash
pytest -q --run-live
```

Release preflight check:

```bash
python scripts/release_check.py
```

## Current Limitations

- Network/API failures are currently handled conservatively and may return fewer diagnostics than ideal.
- Ollama model discovery uses web-page parsing, so UI changes on ollama.com can affect results.
- Hugging Face metadata lookups can be slow for large result sets.
