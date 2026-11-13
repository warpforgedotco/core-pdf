# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations


def is_garbage_text(text: str) -> bool:
    return all(ord(char) < 32 or 0xE000 <= ord(char) <= 0xF8FF for char in text)


def is_visible_text(text: str) -> bool:
    """Whether non-empty text holds a character other than controls and private use."""
    first = ord(text[0])
    return not ((first < 32 or 0xE000 <= first <= 0xF8FF) and is_garbage_text(text))
