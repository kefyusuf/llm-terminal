"""Theme definitions for AI Model Explorer.

Provides color theme presets that can be applied at runtime via the ``t`` key
or configured via the ``AIMODEL_THEME`` environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Color theme with semantic color slots for UI components."""

    name: str
    primary: str  # Main accent color (buttons, highlights)
    secondary: str  # Secondary accent (headers, links)
    background: str  # App background
    surface: str  # Panel/card background
    text: str  # Primary text color
    text_muted: str  # Secondary/muted text
    success: str  # Fit/perfect indicators
    warning: str  # Partial/fit indicators
    error: str  # No fit/error indicators
    info: str  # Informational highlights

    @property
    def css_vars(self) -> dict[str, str]:
        """Return CSS variable overrides for Textual styling."""
        return {
            "$primary": self.primary,
            "$secondary": self.secondary,
            "$background": self.background,
            "$surface": self.surface,
            "$text": self.text,
        }


THEMES: dict[str, Theme] = {
    "default": Theme(
        name="default",
        primary="#00d4aa",
        secondary="#7c3aed",
        background="#0f172a",
        surface="#1e293b",
        text="#e2e8f0",
        text_muted="#94a3b8",
        success="#4fe08a",
        warning="#f2c46d",
        error="#ff7f8f",
        info="#7edfff",
    ),
    "dracula": Theme(
        name="dracula",
        primary="#bd93f9",
        secondary="#ff79c6",
        background="#282a36",
        surface="#44475a",
        text="#f8f8f2",
        text_muted="#6272a4",
        success="#50fa7b",
        warning="#f1fa8c",
        error="#ff5555",
        info="#8be9fd",
    ),
    "nord": Theme(
        name="nord",
        primary="#88c0d0",
        secondary="#81a1c1",
        background="#2e3440",
        surface="#3b4252",
        text="#eceff4",
        text_muted="#d8dee9",
        success="#a3be8c",
        warning="#ebcb8b",
        error="#bf616a",
        info="#8fbcbb",
    ),
    "solarized": Theme(
        name="solarized",
        primary="#268bd2",
        secondary="#b58900",
        background="#002b36",
        surface="#073642",
        text="#fdf6e3",
        text_muted="#93a1a1",
        success="#859900",
        warning="#b58900",
        error="#dc322f",
        info="#2aa198",
    ),
    "monokai": Theme(
        name="monokai",
        primary="#a6e22e",
        secondary="#f92672",
        background="#272822",
        surface="#3e3d32",
        text="#f8f8f2",
        text_muted="#75715e",
        success="#a6e22e",
        warning="#e6db74",
        error="#f92672",
        info="#66d9ef",
    ),
}

THEME_NAMES: list[str] = list(THEMES.keys())


def get_theme(name: str) -> Theme:
    """Return a theme by name, falling back to 'default'."""
    return THEMES.get(name, THEMES["default"])


def next_theme(current: str) -> str:
    """Return the next theme name in the cycle."""
    try:
        idx = THEME_NAMES.index(current)
        return THEME_NAMES[(idx + 1) % len(THEME_NAMES)]
    except ValueError:
        return THEME_NAMES[0]


def theme_css(theme: Theme) -> str:
    """Generate CSS string for a theme to inject into Textual app."""
    return f"""
    App {{
        background: {theme.background};
    }}
    #modal-container, #plan-container, #comparison-container, #job-modal {{
        background: {theme.surface};
    }}
    #modal-header, #plan-header, #comparison-header, #job-header {{
        background: {theme.secondary};
    }}
    """
