"""
Compatibility shim for BentoML and other libraries that expect the
`python-multipart` package to be importable as `multipart`.

The actual package installed in this project is `python-multipart`,
which exposes its functionality under the `python_multipart` namespace.

This package makes `import multipart` and `from multipart.multipart import ...`
work by delegating to `python_multipart.multipart`.
"""

from python_multipart import multipart as _multipart  # type: ignore[attr-defined]

# Re-export commonly used symbols at the package level, mirroring python-multipart.
from python_multipart.multipart import *  # type: ignore[assignment, misc]  # noqa: F401,F403

__all__ = getattr(_multipart, "__all__", [])

