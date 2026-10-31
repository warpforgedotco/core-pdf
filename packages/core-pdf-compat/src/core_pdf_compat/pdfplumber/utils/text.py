from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .. import ObjectDict, _layout_text, plain_text
from .. import extract_words as extract_words


def extract_text(chars: Iterable[ObjectDict], **kwargs: Any) -> str:
    if not kwargs.get("layout"):
        return plain_text(chars, **kwargs)
    return _layout_text(
        chars,
        kwargs.get("layout_bbox", (0, 0, kwargs.get("layout_width", 1), 1)),
        width_chars=int(kwargs.get("layout_width_chars", 80)),
        height_chars=kwargs.get("layout_height_chars"),
    )


extract_text_simple = extract_text

__all__ = ("extract_text", "extract_text_simple", "extract_words")
