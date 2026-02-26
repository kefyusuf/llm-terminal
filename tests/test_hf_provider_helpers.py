from providers.hf_provider import classify_hidden_gem


def test_classify_hidden_gem_positive_case():
    is_gem, score = classify_hidden_gem(downloads=50000, likes=50)
    assert is_gem is True
    assert score > 0


def test_classify_hidden_gem_rejects_high_likes():
    is_gem, score = classify_hidden_gem(downloads=50000, likes=500)
    assert is_gem is False
    assert score == 0.0
