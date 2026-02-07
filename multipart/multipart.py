"""
Submodule shim so that ``from multipart.multipart import ...`` works.

This module simply re-exports everything from ``python_multipart.multipart``.
"""

from python_multipart.multipart import *  # type: ignore[misc]  # noqa: F401,F403

