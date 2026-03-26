from results_view import filter_results_for_view, result_unique_key


def test_result_unique_key_prefers_id_when_present():
    key = result_unique_key({"source": "Hugging Face", "id": "org/repo", "name": "repo"})
    assert key == "Hugging Face:org/repo"


def test_filter_results_for_view_filters_by_provider_and_use_case():
    all_results = [
        {"source": "Ollama", "name": "qwen", "use_case_key": "coding"},
        {"source": "Hugging Face", "name": "qwen", "use_case_key": "coding"},
        {"source": "Ollama", "name": "llama", "use_case_key": "chat"},
    ]

    filtered = filter_results_for_view(
        all_results,
        current_filter="Ollama",
        use_case_filter="coding",
        hidden_gems_only=False,
    )

    assert len(filtered) == 1
    assert filtered[0]["name"] == "qwen"


def test_filter_results_for_view_hidden_gems_only_sorts_by_score_then_downloads():
    all_results = [
        {
            "source": "Hugging Face",
            "name": "model-a",
            "is_hidden_gem": True,
            "gem_score": 0.8,
            "downloads": 100,
        },
        {
            "source": "Hugging Face",
            "name": "model-b",
            "is_hidden_gem": True,
            "gem_score": 0.9,
            "downloads": 50,
        },
        {
            "source": "Hugging Face",
            "name": "model-c",
            "is_hidden_gem": False,
            "gem_score": 1.0,
            "downloads": 500,
        },
    ]

    filtered = filter_results_for_view(
        all_results,
        current_filter="Hugging Face",
        use_case_filter="all",
        hidden_gems_only=True,
    )

    assert [item["name"] for item in filtered] == ["model-b", "model-a"]


def test_filter_results_for_view_sorts_by_downloads_descending():
    all_results = [
        {"source": "Ollama", "name": "a", "downloads": 10, "use_case_key": "general"},
        {"source": "Ollama", "name": "b", "downloads": 50, "use_case_key": "general"},
        {"source": "Ollama", "name": "c", "downloads": 20, "use_case_key": "general"},
    ]

    filtered = filter_results_for_view(
        all_results,
        current_filter="Ollama",
        use_case_filter="all",
        hidden_gems_only=False,
        sort_mode="downloads",
    )

    assert [item["name"] for item in filtered] == ["b", "c", "a"]


def test_filter_results_for_view_applies_fit_filter():
    all_results = [
        {"source": "Ollama", "name": "fit", "fit": "[bold green]Perfect[/bold green]"},
        {"source": "Ollama", "name": "partial", "fit": "[bold yellow]Partial[/bold yellow]"},
        {"source": "Ollama", "name": "nofit", "fit": "[bold red]No Fit[/bold red]"},
    ]

    filtered = filter_results_for_view(
        all_results,
        current_filter="Ollama",
        use_case_filter="all",
        hidden_gems_only=False,
        fit_filter="partial",
    )

    assert [item["name"] for item in filtered] == ["partial"]
