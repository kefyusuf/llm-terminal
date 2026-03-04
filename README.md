# LLM Terminal

A modern terminal user interface for discovering, browsing, and downloading AI language models from Ollama and HuggingFace.

![Python Version](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## About

LLM Terminal is a Textual-based TUI that helps you find local-LLM models from Ollama and Hugging Face, then quickly estimate whether a model will fit your current hardware.

## Features

- 🔍 **Search Models** - Search across Ollama and HuggingFace GGUF models
- 💻 **Hardware Fit** - See if models fit your GPU/RAM before downloading (Perfect / Partial / Slow / No Fit)
- ⬇️ **Download Manager** - Download models with progress tracking
- 🔄 **Progress Tracking** - Real-time download progress and status in table
- 🎨 **Modern UI** - Clean, colorful terminal interface with Textual
- 📊 **Model Details** - View parameters, quantization, size, use case, and more
- 🔐 **HuggingFace Token** - Optional token for faster downloads and higher rate limits
- 🖥️ **Local Detection** - Detects installed Ollama models

## Requirements

- Python 3.10+
- 4GB RAM minimum (8GB recommended)
- GPU recommended for large models
- Internet access for HuggingFace/Ollama registry search
- Ollama (optional, for local model detection and runtime)

> **Note:** The app UI runs without local Ollama. Only `ollama run` commands require Ollama running locally.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-terminal
cd llm-terminal

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the App

```bash
python app.py
```

## Optional: HuggingFace Token

For faster downloads and higher rate limits:

```bash
# Set token as environment variable
export HF_TOKEN=your_token_here
# Or add to .env file
echo "AIMODEL_HF_TOKEN=your_token" > .env
```

Get your token from: https://huggingface.co/settings/tokens

## Usage

### Search Models
- Press `/` to focus search
- Type model name (e.g., "llama", "qwen", "mistral")
- Press Enter to search

### Filter Providers
- Press `p` to cycle between Ollama and HuggingFace
- Or use the radio buttons in the UI

### Download Models
- Double-click or press Enter on a model row to open details
- Click "Download" to start downloading
- Click "Cancel" to stop an active download
- Click "Delete All" to remove model data completely

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `/` | Focus search |
| `p` | Cycle providers |
| `r` | Refresh search |
| `h` | Toggle hidden gems filter |
| `q` | Quit |

## Configuration

Create a `.env` file for custom settings:

```env
# HuggingFace settings
AIMODEL_HF_TOKEN=your_token_here
AIMODEL_HF_SEARCH_LIMIT=15

# Cache settings  
AIMODEL_SEARCH_CACHE_TTL_SECONDS=90

# Ollama settings
AIMODEL_OLLAMA_API_BASE=http://localhost:11434
```

## Project Structure

```
llm-terminal/
├── app.py                  # Main Textual application
├── main.py               # Entry point
├── config.py             # Configuration management
├── providers/           # Ollama and HuggingFace API integrations
│   ├── ollama_provider.py
│   └── hf_provider.py
├── download_service.py   # Background download service
├── download_manager.py  # Download command builder
├── hardware.py         # System hardware detection
├── utils.py           # Utility functions
├── cache_db.py        # SQLite cache
├── requirements.txt   # Python dependencies
└── README.md        # This file
```

## License

MIT License - see LICENSE file.

---

Made with ❤️ for AI enthusiasts