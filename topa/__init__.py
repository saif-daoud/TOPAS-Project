from .config import Config

__all__ = ["TOPA", "Config"]


def __getattr__(name):
    """Keep heavyweight training dependencies lazy for extraction-only runtimes."""
    if name == "TOPA":
        from .core import TOPA

        return TOPA
    raise AttributeError(name)
