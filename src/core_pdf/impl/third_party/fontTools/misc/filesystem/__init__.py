"""Minimal, stdlib-only replacement for [`pyfilesystem2`][1] API for use by `fontTools.ufoLib`.

This package is a partial reimplementation of the `fs` package by Will McGugan, used under the
MIT license. See LICENSE.external for details.

Note this only exports a **subset** of the `pyfilesystem2` API, in particular the modules,
classes and functions that are currently used directly by `fontTools.ufoLib`.

As of version 4.59.0, the `fonttools[ufo]` extra no longer requires the `fs` package, thus
this `fontTools.misc.filesystem` package is used by default.

Client code can either replace `import fs` with `from fontTools.misc import filesystem as fs`
if that happens to work (no guarantee), or they can continue to use `fs` but they will have
to specify it as an explicit dependency of their project.

[1]: https://github.com/PyFilesystem/pyfilesystem2
"""

from __future__ import annotations

from . import _base as base
from . import _copy as copy
from . import _errors as errors
from . import _info as info
from . import _osfs as osfs
from . import _path as path
from . import _subfs as subfs
from . import _tempfs as tempfs
from . import _tools as tools
from . import _walk as walk
from . import _zipfs as zipfs

_haveFS = False


__all__ = [
    "base",
    "copy",
    "errors",
    "info",
    "osfs",
    "path",
    "subfs",
    "tempfs",
    "tools",
    "walk",
    "zipfs",
]
