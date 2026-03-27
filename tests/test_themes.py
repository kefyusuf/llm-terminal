"""Tests for theme system."""

from themes import THEME_NAMES, THEMES, get_theme, next_theme, theme_css


class TestThemes:
    def test_all_themes_have_required_fields(self):
        for name, theme in THEMES.items():
            assert theme.name == name
            assert theme.primary.startswith("#")
            assert theme.background.startswith("#")
            assert theme.text.startswith("#")

    def test_default_theme_exists(self):
        assert "default" in THEMES

    def test_at_least_five_themes(self):
        assert len(THEMES) >= 5


class TestGetTheme:
    def test_returns_correct_theme(self):
        theme = get_theme("dracula")
        assert theme.name == "dracula"

    def test_unknown_name_returns_default(self):
        theme = get_theme("nonexistent")
        assert theme.name == "default"


class TestNextTheme:
    def test_cycles_through_themes(self):
        current = THEME_NAMES[0]
        nxt = next_theme(current)
        assert nxt == THEME_NAMES[1]

    def test_wraps_around(self):
        last = THEME_NAMES[-1]
        nxt = next_theme(last)
        assert nxt == THEME_NAMES[0]

    def test_unknown_returns_first(self):
        nxt = next_theme("nonexistent")
        assert nxt == THEME_NAMES[0]


class TestThemeCss:
    def test_returns_css_string(self):
        theme = get_theme("default")
        css = theme_css(theme)
        assert "background" in css
        assert theme.background in css

    def test_css_vars(self):
        theme = get_theme("dracula")
        vars = theme.css_vars
        assert "$primary" in vars
        assert vars["$primary"] == "#bd93f9"
