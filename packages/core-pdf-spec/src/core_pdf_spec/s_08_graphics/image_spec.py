# SPDX-License-Identifier: AGPL-3.0-only
"""Passive inputs for PDF image XObjects and their soft masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SoftMask:
    """The /SMask plane accompanying an image XObject.

    Carried as its own field rather than smuggled through the image's PDF
    dictionary: that dictionary is the real object dictionary and is exported
    verbatim to display consumers, so private keys in it leak downstream.
    """

    raw: bytes | memoryview
    dictionary: dict[Any, Any]


@dataclass(slots=True, eq=False)
class ImageSource:
    """Resolved PDF image inputs, without device preparation policy."""

    raw: bytes | memoryview
    dictionary: dict[Any, Any]
    soft_mask: SoftMask | None = field(default=None, kw_only=True)


__all__ = (
    "SoftMask",
    "ImageSource",
)
