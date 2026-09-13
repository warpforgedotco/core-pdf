# SPDX-License-Identifier: AGPL-3.0-only
"""Shared PDF lexical tokens and compact syntax aliases."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.standards import PdfVersion, SemanticContext

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]/%"
internal_LEGACY_DELIMITERS = b"()<>[]{}/%"

SEPARATOR_TABLE = bytes([1 if i in WHITESPACE or i in DELIMITERS else 0 for i in range(256)])
WS_TABLE = bytes([1 if i in WHITESPACE else 0 for i in range(256)])


@dataclass(frozen=True, slots=True)
class LexicalRules:
    """Immutable token rules; readers may construct an explicit recovery policy."""

    whitespace: bytes
    name_escapes: bool
    delimiters: bytes = DELIMITERS
    canonical_identifiers: bool = True
    whitespace_table: bytes = field(init=False, repr=False)
    separator_table: bytes = field(init=False, repr=False)
    separator_re: re.Pattern[bytes] = field(init=False, repr=False)
    ignored_re: re.Pattern[bytes] = field(init=False, repr=False)
    split_whitespace_compatible: bool = field(init=False, repr=False)
    content_token_re: re.Pattern[bytes] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "whitespace_table", bytes(int(i in self.whitespace) for i in range(256))
        )
        object.__setattr__(
            self,
            "separator_table",
            bytes(int(i in self.whitespace or i in self.delimiters) for i in range(256)),
        )
        object.__setattr__(
            self,
            "separator_re",
            re.compile(b"[" + re.escape(self.whitespace + self.delimiters) + b"]"),
        )
        object.__setattr__(
            self,
            "split_whitespace_compatible",
            all(byte in self.whitespace for byte in b"\t\n\f\r ")
            and not any(self.separator_table[byte] for byte in b"+-0123456789.")
            and 93 in self.delimiters
            and 93 not in self.whitespace,
        )
        object.__setattr__(
            self,
            "ignored_re",
            re.compile(
                b"(?:[" + re.escape(self.whitespace) + b"]+|%[^\r\n]*(?:\r\n|\n\r|\r|\n)?)*"
            ),
        )
        # One match classifies the common content-stream tokens -- numbers, plain
        # names and operators -- in C. Anything else (strings, arrays,
        # dictionaries, escaped names, malformed numbers) is left to the
        # byte-by-byte scanner so its diagnostics and recovery are unchanged.
        separator = b"[" + re.escape(self.whitespace + self.delimiters) + b"]"
        boundary = b"(?=" + separator + b"|$)"
        name_body = b"[^" + re.escape(self.whitespace + self.delimiters)
        name_body += b"#]*" if self.name_escapes else b"]*"
        operator_start = b"[^" + re.escape(self.whitespace + self.delimiters) + b"+\\-.0-9]"
        operator_rest = b"[^" + re.escape(self.whitespace + self.delimiters) + b"]*"
        ignored = b"(?:[" + re.escape(self.whitespace) + b"]+|%[^\\r\\n]*)*"
        object.__setattr__(
            self,
            "content_token_re",
            re.compile(
                ignored
                + b"(?:(?P<num>[+-]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+))"
                + boundary
                + b"|(?P<name>/"
                + name_body
                + b")"
                + boundary
                + b"|(?P<op>"
                + operator_start
                + operator_rest
                + b")"
                + boundary
                + b")"
            ),
        )


internal_EARLY_RULES = LexicalRules(
    b"\t\n\x0c\r ", False, internal_LEGACY_DELIMITERS, canonical_identifiers=False
)
internal_PDF12_RULES = LexicalRules(
    b"\t\n\x0c\r ", True, internal_LEGACY_DELIMITERS, canonical_identifiers=False
)
internal_PDF1X_RULES = LexicalRules(
    WHITESPACE, True, internal_LEGACY_DELIMITERS, canonical_identifiers=False
)
internal_CURRENT_RULES = LexicalRules(WHITESPACE, name_escapes=True)


def lexical_rules(context: SemanticContext | None = None) -> LexicalRules:
    """Select PDF lexical rules without assuming an unknown document version.

    Adobe PDF Reference 1.2, 4.5 introduces hexadecimal escapes in names.
    Its 4.4 whitespace list lacks NUL, which appears in PDF 1.3, Table 3.1.
    The PDF Association confirms the boundary in "PDF malformations and more":
    https://pdfa.org/pdf-malformations-and-more/ (2023-09-06).
    ISO 32000-2:2020, 7.2.3 limits brace delimiters to Type 4 calculators;
    its corrected 7.3.10 defines canonical object identifiers. Earlier editions
    include braces among ordinary delimiters and specify integer identifiers.
    Omitting context selects the corrected PDF 2.0 grammar.
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
    if version < PdfVersion(2, 0):
        return internal_PDF1X_RULES
    return internal_CURRENT_RULES


__all__ = (
    "LexicalRules",
    "lexical_rules",
    "DELIMITERS",
    "SEPARATOR_TABLE",
    "WHITESPACE",
    "WS_TABLE",
)
