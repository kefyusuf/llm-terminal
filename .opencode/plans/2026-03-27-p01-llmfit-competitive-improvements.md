# Competitive Improvement Plan — AI Model Explorer v1.1

> **STATUS: COMPLETED** — All 3 phases fully implemented. 261 tests passing. 5 providers (Ollama, HuggingFace, LM Studio, Docker, MLX).

## Goal

Transform AI Model Explorer from a search/download tool into a comprehensive LLM model analysis and recommendation platform. Close the feature gap with llmfit by adding multi-dimensional scoring, hardware intelligence, provider expansion, REST API, and comparison capabilities.

## Current State vs Target

| Capability | Current | Target |
|---|---|---|
| Hardware fit | Binary (Perfect/Partial/Slow/No Fit) | 4-dimension scores (Quality/Speed/Fit/Context) |
| GPU support | NVIDIA only | NVIDIA + AMD + Intel Arc + Apple Silicon |
| MoE handling | Not detected | Auto-detect, adjust VRAM calc for active experts |
| Quant optimization | Fixed Q4_K_M preference | Dynamic: try Q8_0 to Q2_K, pick best fit |
| Speed estimate | None | Token/s estimate based on bandwidth + model size |
| Plan mode | None | Reverse analysis: "what HW for this model?" |
| Providers | Ollama + HuggingFace GGUF | + MLX, LM Studio, Docker Model Runner |
| REST API | Download service only (port 8765) | Full API on port 8787 |
| Comparison | None | Multi-select side-by-side comparison |
| Theming | None | 5-6 themes (Dracula, Nord, Solarized, etc.) |
| CLI mode | Basic (info, cache, version) | Comprehensive (fit, search, recommend, plan) |

---

## Phase 1: Scoring Engine & Hardware Intelligence (Highest Impact)

### 1.1 — Multi-Dimensional Scoring System

**New module:** `scoring.py`

Replace binary `calculate_fit()` with a 4-dimension scoring engine:

| Dimension | Range | What it measures |
|---|---|---|
| Quality | 0-100 | Parameter count, model reputation, quantization penalty |
| Speed | 0-100 | Estimated tok/s (bandwidth / model_size * efficiency) |
| Fit | 0-100 | Memory utilization efficiency (sweet spot: 50-80% VRAM) |
| Context | 0-100 | Context window capacity vs model capability |

**Use-case weights:**
```python
USE_CASE_WEIGHTS = {
    "chat":      {"quality": 0.25, "speed": 0.35, "fit": 0.25, "context": 0.15},
    "coding":    {"quality": 0.35, "speed": 0.30, "fit": 0.20, "context": 0.15},
    "reasoning": {"quality": 0.55, "speed": 0.15, "fit": 0.15, "context": 0.15},
    "vision":    {"quality": 0.30, "speed": 0.25, "fit": 0.25, "context": 0.20},
    "math":      {"quality": 0.45, "speed": 0.20, "fit": 0.20, "context": 0.15},
    "embedding": {"quality": 0.30, "speed": 0.40, "fit": 0.20, "context": 0.10},
    "general":   {"quality": 0.30, "speed": 0.25, "fit": 0.25, "context": 0.20},
}
```

**Composite score:** `weighted_avg = sum(dimension_score * weight)`

**Files to create/modify:**
- NEW: `scoring.py` — scoring engine (pure functions)
- `utils.py` — keep `calculate_fit()` as fallback, add bridge to new scoring
- `models.py` — add score fields to `ModelResult`
- `providers/hf_provider.py` — call scoring after size estimation
- `providers/ollama_provider.py` — call scoring after size estimation
- `results_presenter.py` — render compact score bar (e.g. Q72 S45 F88 C60)
- `results_view.py` — add composite score sort support
- `app.py` — wire up score display in table and detail modal

**Tests:** `tests/test_scoring.py`

### 1.2 — Speed Estimation

**Module:** `scoring.py` (part of 1.1)

**Formula:** `tok_s = (bandwidth_gb_s / model_size_gb) * efficiency_factor`

**GPU bandwidth lookup table** (~80 common GPUs):
```python
GPU_BANDWIDTH = {
    # NVIDIA
    "RTX 4090": 1008, "RTX 4080": 717, "RTX 4070 Ti": 504,
    "RTX 3090": 936, "RTX 3080": 760, "RTX 3070": 448,
    "RTX 3060": 360, "RTX 2080 Ti": 616, "A100 80GB": 2039,
    "H100": 3350, "L40S": 864, "T4": 300, "V100": 900,
    # AMD
    "RX 7900 XTX": 960, "RX 7900 XT": 800, "MI250X": 3277, "MI300X": 5200,
    # Apple Silicon
    "M2 Ultra": 800, "M2 Max": 400, "M3 Max": 400, "M3 Ultra": 800,
}
```

**Efficiency penalties:**
- CPU offload (Partial fit): x0.5
- CPU-only: x0.3
- Unknown GPU: backend default (CUDA: 220, Metal: 160, ROCm: 180)

**Files:** `scoring.py`, `hardware.py` (GPU name normalization)

### 1.3 — MoE Awareness

**Module:** `model_intelligence.py` (NEW)

Detect Mixture-of-Experts models by name patterns:
```python
MOE_PATTERNS = [
    r"mixtral", r"deepseek[_-]v[23]", r"qwen.*moe", r"jamba",
    r"grok", r"switch", r"nllb.*moe", r"olmoe",
]
```

**VRAM adjustment:** Only active experts load into VRAM at a time.
```python
def adjust_vram_for_moe(model_size_gb, total_experts, active_experts=2):
    return model_size_gb * (active_experts / total_experts)
```

**Files:**
- NEW: `model_intelligence.py`
- `utils.py` — use MoE-aware size calculation
- `scoring.py` — incorporate MoE factor into scores
- `models.py` — add `is_moe`, `total_experts`, `active_experts` fields

**Tests:** `tests/test_model_intelligence.py`

### 1.4 — Dynamic Quantization

**Module:** `model_intelligence.py`

**Quantization size table:**
```python
QUANT_MULTIPLIERS = {
    "FP16": 2.0, "Q8_0": 1.0, "Q6_K": 0.75, "Q5_K_M": 0.65,
    "Q5_0": 0.625, "Q4_K_M": 0.55, "Q4_0": 0.5, "Q3_K": 0.4, "Q2_K": 0.3,
}
```

**Algorithm:** Try Q8_0 to Q2_K, return highest quality that fits in VRAM.

**Files:** `model_intelligence.py`, `providers/hf_provider.py` (enhance GGUF selection)

### 1.5 — Expanded GPU Support

**Modify:** `hardware.py`

Add detection for:
- AMD via `rocm-smi --showmeminfo vram`
- Intel Arc via sysfs + lspci
- Apple Silicon via `system_profiler SPDisplaysDataType`

Add `backend` field to specs: `cuda`, `rocm`, `metal`, `sycl`, `cpu_x86`, `cpu_arm`

**Tests:** `tests/test_hardware.py` (mock-based)

---

## Phase 2: Plan Mode & Comparison (UX Features)

### 2.1 — Plan Mode (Reverse Analysis)

**New TUI screen:** `PlanModeScreen` in `app.py`

User enters a model name -> system returns minimum hardware requirements:
- Required VRAM for each quant level
- Required RAM for CPU offload
- Recommended GPU tier
- Estimated speed on different hardware configs

**Trigger:** `p` key opens plan mode modal

**Files:** `app.py` (new modal), `model_intelligence.py`, `scoring.py`

**Tests:** `tests/test_plan_mode.py`

### 2.2 — Model Comparison Mode

**New TUI feature:** Multi-select comparison

- `c` key enters comparison selection mode
- Space toggles models in/out of comparison set (max 4)
- Enter opens side-by-side comparison table with scores

**Files:** `app.py` (new `ComparisonModal`), `results_presenter.py`

### 2.3 — Theme Support

**New module:** `themes.py`

5 themes: default, dracula, nord, solarized, monokai

**Config:** `AIMODEL_THEME` env variable, runtime toggle with `t` key

**Files:** `themes.py`, `app.py` (CSS variable injection), `config.py`

---

## Phase 3: Provider Expansion & API (Infrastructure)

### 3.1 — LM Studio Provider

**New file:** `providers/lmstudio_provider.py`

- Detect on `localhost:1234` (configurable via `LMSTUDIO_HOST`)
- List installed models via `/v1/models` API
- Download via LM Studio HF integration

### 3.2 — Docker Model Runner Provider

**New file:** `providers/docker_provider.py`

- Detect on `localhost:12434`
- List models via API

### 3.3 — MLX Provider (Apple Silicon)

**New file:** `providers/mlx_provider.py`

- Detect MLX installation
- List models from `mlx-community` cache

### 3.4 — Provider Architecture Refactor

**Modify:** `providers/__init__.py`

Abstract base class with `search()`, `detect()`, `list_installed()` methods.
Provider registry for dynamic provider discovery.

### 3.5 — REST API

**New file:** `api_server.py` (port 8787)

```
GET  /health                      # Liveness
GET  /api/v1/system               # Hardware info
GET  /api/v1/models               # Filtered model list
GET  /api/v1/models/top           # Top N by composite score
GET  /api/v1/models/{name}/plan   # Hardware plan
GET  /api/v1/providers            # Available providers
GET  /api/v1/scores/{name}        # Score breakdown
```

**Files:** `api_server.py`, `config.py`, `main.py`

### 3.6 — Enhanced CLI

**Modify:** `cli.py`

```bash
ai-model-explorer-cli system              # Hardware info
ai-model-explorer-cli search "llama"      # Tabular results
ai-model-explorer-cli fit --perfect -n 5  # Top 5 fitting
ai-model-explorer-cli recommend --json    # JSON recommendations
ai-model-explorer-cli plan "llama-8b"     # Hardware plan
ai-model-explorer-cli scores "llama-8b"   # Score breakdown
```

---

## Implementation Order

| # | Task | Phase | Est. Scope |
|---|---|---|---|
| 1 | `model_intelligence.py` — MoE, quant table, size | 1 | Small module |
| 2 | `scoring.py` — 4-dimension scoring engine | 1 | Small module |
| 3 | `hardware.py` — AMD/Intel/Apple detection | 1 | Medium refactor |
| 4 | Speed estimation with GPU bandwidth table | 1 | Extension |
| 5 | `models.py` — extend ModelResult fields | 1 | Small |
| 6 | Wire scoring into HF + Ollama providers | 1 | Medium |
| 7 | `results_presenter.py` — render score bars | 1 | Small |
| 8 | `results_view.py` — composite score sorting | 1 | Small |
| 9 | Plan mode modal | 2 | Medium |
| 10 | Comparison mode | 2 | Medium |
| 11 | `themes.py` — theme system | 2 | Small |
| 12 | LM Studio provider | 3 | Small |
| 13 | Docker provider | 3 | Small |
| 14 | MLX provider | 3 | Small |
| 15 | Provider architecture refactor | 3 | Medium |
| 16 | REST API server | 3 | Medium |
| 17 | Enhanced CLI | 3 | Small |
| 18 | Tests for all modules | ALL | Continuous |

## New Files Summary

| File | Purpose |
|---|---|
| `scoring.py` | 4-dimension scoring engine |
| `model_intelligence.py` | MoE detection, dynamic quant, size estimation |
| `themes.py` | Theme definitions |
| `providers/lmstudio_provider.py` | LM Studio integration |
| `providers/docker_provider.py` | Docker Model Runner integration |
| `providers/mlx_provider.py` | MLX integration |
| `api_server.py` | REST API server |

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| GPU bandwidth accuracy | Allow user override via config; start with known values |
| MoE false positives | Strict regex + manual override list |
| Provider API instability | Timeout + retry with backoff; graceful degradation |
| API security | Localhost only by default; no auth needed for local use |
