from search_orchestration import (
    build_query_key,
    cache_hit_suffix,
    has_more_pages_for_results,
    is_hf_provider_selection,
    page_info_suffix,
    provider_result_count,
    provider_search_status,
    providers_from_filter,
    validate_page_request,
)


def test_providers_from_filter_for_hf_label():
    assert providers_from_filter("Hugging Face") == ["huggingface"]


def test_build_query_key_uses_page_for_hf():
    assert build_query_key(["huggingface"], "Qwen", 2) == "hf:qwen:page2"


def test_build_query_key_omits_page_for_ollama():
    assert build_query_key(["ollama"], "Qwen", 2) == "ollama:qwen"


def test_cache_hit_suffix_for_hf_includes_page():
    assert cache_hit_suffix(["huggingface"], 1) == " (cached page 2)"


def test_validate_page_request_rejects_out_of_bounds():
    is_valid, message = validate_page_request(10, 10)
    assert is_valid is False
    assert message == "Reached page limit (10)."


def test_has_more_pages_for_hf_when_page_full():
    assert has_more_pages_for_results(
        ["huggingface"], hf_result_count=15, ollama_result_count=0, page_size=15
    )


def test_has_more_pages_for_ollama_always_false():
    assert not has_more_pages_for_results(
        ["ollama"], hf_result_count=0, ollama_result_count=20, page_size=15
    )


def test_provider_result_count_tracks_selected_provider():
    assert provider_result_count(["huggingface"], 11, 22) == 11
    assert provider_result_count(["ollama"], 11, 22) == 22


def test_provider_search_status_text_for_ollama():
    text = provider_search_status(["ollama"], result_count=8, has_more_pages=False, current_page=0)
    assert "pagination not supported" in text


def test_page_info_suffix_for_second_page():
    assert page_info_suffix(1) == " (Page 2)"


def test_is_hf_provider_selection():
    assert is_hf_provider_selection(["huggingface"]) is True
    assert is_hf_provider_selection(["ollama"]) is False
