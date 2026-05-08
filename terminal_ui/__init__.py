"""Legacy experimental terminal UI package."""

__all__ = ["ObsidianConsole", "main"]


def __getattr__(name):
	if name not in __all__:
		msg = f"module {__name__!r} has no attribute {name!r}"
		raise AttributeError(msg)

	from .app import ObsidianConsole, main

	exports = {
		"ObsidianConsole": ObsidianConsole,
		"main": main,
	}
	return exports[name]
