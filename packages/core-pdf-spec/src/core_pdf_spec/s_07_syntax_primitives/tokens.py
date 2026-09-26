# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from typing import Any, ClassVar, Self

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_records import FrozenFields, PickleFields, ReprFields, frozen_setattr

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]/%"
LEGACY_DELIMITERS = b"()<>[]{}/%"

SEPARATOR_TABLE = bytes([1 if i in WHITESPACE or i in DELIMITERS else 0 for i in range(256)])
WS_TABLE = bytes([1 if i in WHITESPACE else 0 for i in range(256)])


class LexicalRules(FrozenFields, PickleFields, ReprFields):
    __slots__ = (
        "whitespace",
        "name_escapes",
        "delimiters",
        "canonical_identifiers",
        "whitespace_table",
        "separator_table",
        "separator_re",
        "ignored_re",
        "split_whitespace_compatible",
        "content_token_re",
    )

    whitespace: bytes
    name_escapes: bool
    delimiters: bytes
    canonical_identifiers: bool
    whitespace_table: bytes
    separator_table: bytes
    separator_re: re.Pattern[bytes]
    ignored_re: re.Pattern[bytes]
    split_whitespace_compatible: bool
    content_token_re: re.Pattern[bytes]

    __fields__: ClassVar[tuple[str, ...]] = (
        "whitespace",
        "name_escapes",
        "delimiters",
        "canonical_identifiers",
        "whitespace_table",
        "separator_table",
        "separator_re",
        "ignored_re",
        "split_whitespace_compatible",
        "content_token_re",
    )
    __repr_fields__: ClassVar[tuple[str, ...]] = (
        "whitespace",
        "name_escapes",
        "delimiters",
        "canonical_identifiers",
    )
    __match_args__ = ("whitespace", "name_escapes", "delimiters", "canonical_identifiers")

    def __init__(
        self,
        whitespace: bytes,
        name_escapes: bool,
        delimiters: bytes = DELIMITERS,
        canonical_identifiers: bool = True,
    ) -> None:
        frozen_setattr(self, "whitespace", whitespace)
        frozen_setattr(self, "name_escapes", name_escapes)
        frozen_setattr(self, "delimiters", delimiters)
        frozen_setattr(self, "canonical_identifiers", canonical_identifiers)
        self._post_init()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.whitespace == other.whitespace
            and self.name_escapes == other.name_escapes
            and self.delimiters == other.delimiters
            and self.canonical_identifiers == other.canonical_identifiers
            and self.whitespace_table == other.whitespace_table
            and self.separator_table == other.separator_table
            and self.separator_re == other.separator_re
            and self.ignored_re == other.ignored_re
            and self.split_whitespace_compatible == other.split_whitespace_compatible
            and self.content_token_re == other.content_token_re
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.whitespace,
                self.name_escapes,
                self.delimiters,
                self.canonical_identifiers,
                self.whitespace_table,
                self.separator_table,
                self.separator_re,
                self.ignored_re,
                self.split_whitespace_compatible,
                self.content_token_re,
            )
        )

    def __replace__(self, /, **changes: Any) -> Self:
        whitespace = changes.pop("whitespace", self.whitespace)
        name_escapes = changes.pop("name_escapes", self.name_escapes)
        delimiters = changes.pop("delimiters", self.delimiters)
        canonical_identifiers = changes.pop("canonical_identifiers", self.canonical_identifiers)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(whitespace, name_escapes, delimiters, canonical_identifiers)

    def _post_init(self) -> None:
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


EARLY_RULES = LexicalRules(b"\t\n\x0c\r ", False, LEGACY_DELIMITERS, canonical_identifiers=False)
PDF12_RULES = LexicalRules(b"\t\n\x0c\r ", True, LEGACY_DELIMITERS, canonical_identifiers=False)
PDF1X_RULES = LexicalRules(WHITESPACE, True, LEGACY_DELIMITERS, canonical_identifiers=False)
CURRENT_RULES = LexicalRules(WHITESPACE, name_escapes=True)

PDF_1_2 = PdfVersion(1, 2)
PDF_1_3 = PdfVersion(1, 3)
PDF_2_0 = PdfVersion(2, 0)


def lexical_rules(context: SemanticContext | None = None) -> LexicalRules:
    if context is None:
        return CURRENT_RULES
    version = context.version
    if version is None or not version.recognized:
        raise PdfUnsupportedError("lexical semantics require a recognized PDF version")
    if version < PDF_1_2:
        return EARLY_RULES
    if version < PDF_1_3:
        return PDF12_RULES
    if version < PDF_2_0:
        return PDF1X_RULES
    return CURRENT_RULES


__all__ = (
    "LexicalRules",
    "lexical_rules",
    "DELIMITERS",
    "SEPARATOR_TABLE",
    "WHITESPACE",
    "WS_TABLE",
)
