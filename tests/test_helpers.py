import utils


def test_extract_params_detects_billions():
    assert utils.extract_params("llama-3.1-8b-instruct") == "8B"


def test_extract_params_returns_dash_when_missing():
    assert utils.extract_params("my-model") == "-"


def test_format_likes_compacts_thousands_and_millions():
    assert utils.format_likes(1500) == "1.5K"
    assert utils.format_likes(2_500_000) == "2.5M"


def test_determine_use_case_coding_priority():
    assert "Coding" in utils.determine_use_case("deepseek-coder-7b")


def test_determine_use_case_key_reasoning():
    assert utils.determine_use_case_key("deepseek-r1-7b") == "reasoning"


def test_infer_quant_from_name_detects_known_quantization():
    assert utils.infer_quant_from_name("qwen2.5-coder-q4_k_m-gguf") == "Q4_K_M"


def test_infer_quant_from_name_returns_default():
    assert utils.infer_quant_from_name("qwen2.5-coder", default="GGUF") == "GGUF"


def test_calculate_fit_perfect_gpu():
    specs = {
        "has_gpu": True,
        "vram_free": 12.0,
        "ram_free": 16.0,
    }
    fit, mode, resource = utils.calculate_fit(7.0, specs)
    assert "Perfect" in fit
    assert "GPU" in mode
    assert resource == "VRAM"


def test_calculate_fit_no_fit_when_memory_low():
    specs = {
        "has_gpu": False,
        "vram_free": 0.0,
        "ram_free": 2.0,
    }
    fit, _mode, resource = utils.calculate_fit(8.0, specs)
    assert "No Fit" in fit
    assert resource == "Insufficient"
