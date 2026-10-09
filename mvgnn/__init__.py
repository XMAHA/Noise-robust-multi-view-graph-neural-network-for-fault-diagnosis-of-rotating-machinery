"""Public implementation of the multi-view graph neural network (MvGNN)."""

__all__ = ["MvGNN"]


def __getattr__(name: str):
    """Keep preprocessing usable before the graph-learning stack is imported."""
    if name == "MvGNN":
        from .model import MvGNN
        return MvGNN
    raise AttributeError(name)
