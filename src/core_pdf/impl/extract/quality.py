# SPDX-License-Identifier: AGPL-3.0-only
"""Native text quality analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from core_pdf.impl.extract.contracts import TextQualityStats


@dataclass(frozen=True, slots=True)
class TextAnalysis:
    quality: TextQualityStats = field(default_factory=TextQualityStats)
    characters: int = 0
    suspicious_characters: int = 0


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
        # `tokens` comes from a `\S+` regex match, so every character in `token` already
        # satisfies `not character.isspace()` (CPython's Unicode `\s`/`str.isspace()` share
        # the same whitespace classification) -- iterate `token` directly instead of
        # rebuilding an always-identical filtered copy.
        if len(token) <= 2:
            short_tokens += 1
        if token.isascii() and token.isprintable():
            # Printable ASCII contributes no non-ASCII or suspicious counts,
            # and the common all-letter / all-digit tokens resolve with
            # whole-string C checks instead of six method calls per character.
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
