"""Lightweight error type hierarchy for provider search and API operations."""

from __future__ import annotations


class ProviderError(Exception):
    """Base error for all provider-related failures."""


class RateLimitError(ProviderError):
    """Remote API returned a rate-limit response (HTTP 429)."""


class NetworkError(ProviderError):
    """A transient network failure occurred (timeout, DNS, connection refused)."""


class ParseError(ProviderError):
    """Failed to parse provider response data."""


class AuthenticationError(ProviderError):
    """Provider authentication failed (invalid/missing token)."""


class NotFoundError(ProviderError):
    """Requested resource was not found on the provider."""
