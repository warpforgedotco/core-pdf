# SPDX-License-Identifier: AGPL-3.0-only
"""Shared PDF lexical tokens and compact syntax aliases."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.standards import PdfVersion, SemanticContext

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]{}/%"

SEPARATOR_TABLE = bytes([1 if i in WHITESPACE or i in DELIMITERS else 0 for i in range(256)])
WS_TABLE = bytes([1 if i in WHITESPACE else 0 for i in range(256)])


@dataclass(frozen=True, slots=True)
class LexicalRules:
    """Immutable token rules; readers may construct an explicit recovery policy."""

    whitespace: bytes
    name_escapes: bool
    whitespace_table: bytes = field(init=False, repr=False)
    separator_table: bytes = field(init=False, repr=False)
    separator_re: re.Pattern[bytes] = field(init=False, repr=False)
    ignored_re: re.Pattern[bytes] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "whitespace_table", bytes(int(i in self.whitespace) for i in range(256))
        )
        object.__setattr__(
            self,
            "separator_table",
            bytes(int(i in self.whitespace or i in DELIMITERS) for i in range(256)),
        )
        object.__setattr__(
            self, "separator_re", re.compile(b"[" + re.escape(self.whitespace + DELIMITERS) + b"]")
        )
        object.__setattr__(
            self,
            "ignored_re",
            re.compile(
                b"(?:[" + re.escape(self.whitespace) + b"]+|%[^\r\n]*(?:\r\n|\n\r|\r|\n)?)*"
            ),
        )


internal_EARLY_RULES = LexicalRules(b"\t\n\x0c\r ", name_escapes=False)
internal_PDF12_RULES = LexicalRules(b"\t\n\x0c\r ", name_escapes=True)
internal_CURRENT_RULES = LexicalRules(WHITESPACE, name_escapes=True)


def lexical_rules(context: SemanticContext | None = None) -> LexicalRules:
    """Select PDF lexical rules without assuming an unknown document version.

    Adobe PDF Reference 1.2, 4.5 introduces hexadecimal escapes in names.
    Its 4.4 whitespace list lacks NUL, which appears in PDF 1.3, Table 3.1.
    The PDF Association confirms the boundary in "PDF malformations and more":
    https://pdfa.org/pdf-malformations-and-more/ (2023-09-06).
    Omitting context retains the existing modern lexical API.
    """
    if context is None:
        return internal_CURRENT_RULES
    version = context.version
    if version is None or not version.recognized:
        raise PdfUnsupportedError("lexical semantics require a recognized PDF version")
    if version < PdfVersion(1, 2):
        return internal_EARLY_RULES
    if version < PdfVersion(1, 3):
        return internal_PDF12_RULES
    return internal_CURRENT_RULES


__all__ = (
    "LexicalRules",
    "lexical_rules",
    "DELIMITERS",
    "SEPARATOR_TABLE",
    "WHITESPACE",
    "WS_TABLE",
)
