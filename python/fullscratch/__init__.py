"""
FullScratchML: A comprehensive C++ machine learning library with Python bindings
"""

__version__ = "0.1.0"

# Import C++ bindings when available
try:
    from .fullscratch import *  # noqa: F401, F403
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import C++ bindings: {e}")

__all__ = [
    "__version__",
]
