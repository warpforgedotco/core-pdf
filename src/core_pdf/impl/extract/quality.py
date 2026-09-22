# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, Self

from core_pdf.impl.extract.contracts import TextQualityStats
from core_pdf.impl.records import internal_Record

internal_frozen_setattr = object.__setattr__


class TextAnalysis(internal_Record):
    __slots__ = ("quality", "characters", "suspicious_characters")

    quality: TextQualityStats
    characters: int
    suspicious_characters: int

    __fields__: ClassVar[tuple[str, ...]] = ("quality", "characters", "suspicious_characters")
    __match_args__ = ("quality", "characters", "suspicious_characters")

    def __init__(
        self,
        quality: TextQualityStats | None = None,
        characters: int = 0,
        suspicious_characters: int = 0,
    ) -> None:
        internal_frozen_setattr(self, "quality", TextQualityStats() if quality is None else quality)
        internal_frozen_setattr(self, "characters", characters)
        internal_frozen_setattr(self, "suspicious_characters", suspicious_characters)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"quality={self.quality!r}, "
            f"characters={self.characters!r}, "
            f"suspicious_characters={self.suspicious_characters!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.quality == other.quality
            and self.characters == other.characters
            and self.suspicious_characters == other.suspicious_characters
        )

    def __hash__(self) -> int:
        return hash((self.quality, self.characters, self.suspicious_characters))

    def __replace__(self, /, **changes: Any) -> Self:
        quality = changes.pop("quality", self.quality)
        characters = changes.pop("characters", self.characters)
        suspicious_characters = changes.pop("suspicious_characters", self.suspicious_characters)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(quality, characters, suspicious_characters)


internal_ASCII_VOWELS = frozenset("aeiouAEIOU")


def internal_analyze_text(text: str) -> TextAnalysis:
    tokens = text.split()
    if not tokens:
        return TextAnalysis()
    wordlike = 0
    short_tokens = 0
    digit_tokens = 0
    nonspace = 0
    symbols = 0
    non_ascii = 0
    suspicious = 0
    for token in tokens:
        if len(token) <= 2:
            short_tokens += 1
        if token.isascii() and token.isprintable():
            nonspace += len(token)
            if token.isalpha():
                if len(token) >= 3 and not internal_ASCII_VOWELS.isdisjoint(token):
                    wordlike += 1
                continue
            if token.isdigit():
                digit_tokens += 1
                continue
            has_digit = False
            letter_count = 0
            has_vowel = False
            for character in token:
                if character.isalnum():
                    if character.isdigit():
                        has_digit = True
                    else:
                        letter_count += 1
                        if not has_vowel and character in "aeiouAEIOU":
                            has_vowel = True
                else:
                    symbols += 1
            if has_digit:
                digit_tokens += 1
            if letter_count >= 3 and has_vowel:
                wordlike += 1
            continue
        has_digit = False
        letter_count = 0
        has_vowel = False
        for character in token:
            codepoint = ord(character)
            nonspace += 1
            if character.isdigit():
                has_digit = True
            if character.isalpha():
                letter_count += 1
                if not has_vowel and character.casefold() in "aeiou":
                    has_vowel = True
            if not character.isalnum():
                symbols += 1
            if codepoint > 127:
                non_ascii += 1
            if (
                character == "\ufffd"
                or 0xE000 <= codepoint <= 0xF8FF
                or (not character.isprintable() and not character.isspace())
            ):
                suspicious += 1
        if has_digit:
            digit_tokens += 1
        if letter_count >= 3 and has_vowel:
            wordlike += 1
    if not nonspace:
        return TextAnalysis(TextQualityStats(token_count=len(tokens)))
    return TextAnalysis(
        quality=TextQualityStats(
            token_count=len(tokens),
            wordlike_ratio=wordlike / len(tokens),
            short_token_ratio=short_tokens / len(tokens),
            symbol_ratio=symbols / nonspace,
            non_ascii_ratio=non_ascii / nonspace,
            digit_token_ratio=digit_tokens / len(tokens),
        ),
        characters=nonspace,
        suspicious_characters=suspicious,
    )
