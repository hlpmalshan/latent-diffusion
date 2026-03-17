"""Expose the bundled taming-transformers package from this repository."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_bundled_taming = (
    Path(__file__).resolve().parent.parent / "src" / "taming-transformers" / "taming"
)
if _bundled_taming.is_dir():
    bundled_path = str(_bundled_taming)
    if bundled_path not in __path__:
        __path__.append(bundled_path)
