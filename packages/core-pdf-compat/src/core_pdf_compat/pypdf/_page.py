"""The import path pypdf gives its page class and Transformation."""

from __future__ import annotations

from importlib.util import find_spec

from ._doc_common import _VirtualList
from ._images import ImageFile, VirtualListImages
from ._page_merge import Transformation
from ._writer import PageObject

# pypdf's flag: without Pillow, the error is raised only when images are used.
try:
    pil_not_imported = find_spec("PIL") is None
except ValueError:  # A module set to None in sys.modules, as tests do to hide Pillow.
    pil_not_imported = True

__all__ = ["ImageFile", "PageObject", "Transformation", "VirtualListImages", "_VirtualList"]
