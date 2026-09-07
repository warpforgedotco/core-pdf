"""PyMuPDF object inspection over the native cross-reference resolver."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from core_pdf.api.compat._shared import float32
from core_pdf.api.document import PdfDocument
from core_pdf.impl._impl.document.recovery.text_strings import decode_pdf_text_string
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.types import PdfName, PdfReference, PdfString


def internal_number(value: float) -> str:
    rounded = float32(value)
    for precision in range(1, 10):
        candidate = format(rounded, f".{precision}g")
        if float32(float(candidate)) == rounded:
            break
    result = format(Decimal(candidate), "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    if result.startswith("0."):
        result = result[1:]
    elif result.startswith("-0."):
        result = "-" + result[2:]
    return "0" if result == "-0" else result


def internal_name(value: object) -> str:
    data = value.value_bytes if isinstance(value, PdfName) else str(value).encode("latin-1")
    return "/" + "".join(
        chr(c) if 33 <= c <= 126 and c not in b"()<>[]{}/%#" else f"#{c:02X}" for c in data
    )


def internal_string(data: bytes, *, ascii_only: bool = False) -> str:
    if ascii_only and any(not 32 <= c < 127 and c not in b"\n\r\t" for c in data):
        return "<" + data.hex().upper() + ">"
    escapes = {10: r"\n", 13: r"\r", 9: r"\t", 8: r"\b", 12: r"\f", 40: r"\(", 41: r"\)", 92: r"\\"}
    escaped = "".join(escapes.get(c, chr(c) if 32 <= c < 127 else f"\\{c:03o}") for c in data)
    return "(" + escaped + ")" if len(escaped) <= 2 * len(data) else "<" + data.hex().upper() + ">"


def internal_join(tokens: list[str]) -> str:
    output = ""
    delimiters = "()<>[]{}/%"
    for token in tokens:
        if output and token and output[-1] not in delimiters and token[0] not in delimiters:
            output += " "
        output += token
    return output


def format_object(
    value: object,
    *,
    compressed: bool = False,
    ascii_only: bool = False,
    depth: int = 0,
    column: int = 0,
) -> str:
    if isinstance(value, PdfStream):
        value = value.dictionary
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, PdfReference):
        return str(value)
    if isinstance(value, PdfName):
        return internal_name(value)
    if isinstance(value, PdfString):
        return internal_string(value.data, ascii_only=ascii_only)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return internal_number(value)
    if isinstance(value, (list, tuple)):
        if compressed:
            return (
                "["
                + internal_join(
                    [format_object(v, compressed=True, ascii_only=ascii_only) for v in value]
                )
                + "]"
            )
        output = "[ "
        current = column + 2
        for index, item in enumerate(value):
            if index:
                separator = "\n" + " " * (2 * depth + (4 if depth else 2)) if current > 60 else " "
                output += separator
                current = len(separator.rsplit("\n", 1)[-1]) if "\n" in separator else current + 1
            token = format_object(item, depth=depth + 1, column=current, ascii_only=ascii_only)
            output += token
            current = len(token.rsplit("\n", 1)[-1]) if "\n" in token else current + len(token)
        return output + (" ]" if value else "]")
    if isinstance(value, dict):
        if compressed:
            return (
                "<<"
                + internal_join(
                    [
                        part
                        for k, v in value.items()
                        for part in (
                            internal_name(k),
                            format_object(
                                v, compressed=True, depth=depth + 1, ascii_only=ascii_only
                            ),
                        )
                    ]
                )
                + ">>"
            )
        indent = "  " * depth
        return (
            "<<\n"
            + "".join(
                indent
                + "  "
                + internal_name(k)
                + " "
                + format_object(
                    v,
                    depth=depth + 1,
                    column=len(indent) + 3 + len(internal_name(k)),
                    ascii_only=ascii_only,
                )
                + "\n"
                for k, v in value.items()
            )
            + indent
            + ">>"
        )
    raise TypeError(f"unsupported PDF object: {type(value).__name__}")


class ObjectAccess:
    def __init__(self, source: PdfDocument | None) -> None:
        self.source = source
        # Catalog defaults of the pinned 1.28.2 reference, before adding pages.
        self.empty: dict[int, Any] = {
            1: {
                "Type": PdfName.of("Catalog"),
                "Pages": PdfReference(2),
                "Info": {"Producer": PdfString(b"MuPDF 1.28.2")},
            },
            2: {"Type": PdfName.of("Pages"), "Count": 0, "Kids": []},
        }

    @property
    def length(self) -> int:
        if self.source is None:
            return 3
        declared_size = self.source.trailer_dict.get("Size", 0)
        return max(
            int(declared_size) if isinstance(declared_size, (int, float)) else 0,
            max((key >> 16 for key in self.source.xref), default=0) + 1,
        )

    @property
    def trailer(self) -> dict[Any, Any]:
        if self.source is None:
            return {"Size": 3, "Root": PdfReference(1)}
        trailer = self.source.trailer_dict
        return trailer if "Size" in trailer else {"Size": self.length, **trailer}

    def resolve(self, value: object) -> object:
        if self.source is not None:
            return self.source.resolver.resolve(value)
        return self.empty.get(value.object_number) if isinstance(value, PdfReference) else value

    def get(self, xref: int) -> object:
        if xref == -1:
            return self.trailer
        if not 0 < xref < self.length:
            raise ValueError("bad xref")
        if self.source is None:
            return self.empty.get(xref)
        entries = [
            (key & 65535, entry) for key, entry in self.source.xref.items() if key >> 16 == xref
        ]
        if not entries:
            return None
        generation, entry = max(entries, key=lambda item: item[0])
        if not entry.in_use:
            return None
        return self.resolve(PdfReference(xref, generation))

    def keys(self, xref: int) -> list[str]:
        obj = self.get(xref)
        if isinstance(obj, PdfStream):
            obj = obj.dictionary
        return (
            [str(key).encode("latin-1").decode("utf-8", "surrogateescape") for key in obj]
            if isinstance(obj, dict)
            else []
        )

    def key(self, xref: int, key: str) -> tuple[str, str]:
        obj = self.get(xref)
        for part in key.split("/"):
            obj = self.resolve(obj)
            if isinstance(obj, PdfStream):
                obj = obj.dictionary
            name = part.encode("utf-8", "surrogateescape").decode("latin-1")
            obj = obj.get(name) if isinstance(obj, dict) else None
        if isinstance(obj, PdfString):
            return "string", decode_pdf_text_string(obj.data).split("\0", 1)[0]
        if isinstance(obj, PdfName):
            return "name", "/" + obj.value_bytes.decode("utf-8", "surrogateescape")
        kind = (
            "null"
            if obj is None
            else "bool"
            if isinstance(obj, bool)
            else "xref"
            if isinstance(obj, PdfReference)
            else "int"
            if isinstance(obj, int)
            else "float"
            if isinstance(obj, float)
            else "array"
            if isinstance(obj, (list, tuple))
            else "dict"
        )
        return kind, format_object(obj, compressed=True)


__all__ = ("ObjectAccess", "format_object")
