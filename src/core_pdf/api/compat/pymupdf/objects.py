"""PyMuPDF object inspection over the native cross-reference resolver."""

from __future__ import annotations

import zlib
from decimal import Decimal
from typing import Any, cast
from uuid import uuid4

from core_pdf.api.compat._shared import float32
from core_pdf.api.document import PdfDocument
from core_pdf.impl._impl.document.recovery.text_strings import decode_pdf_text_string
from core_pdf.impl._impl.document.write import write_pdf
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.serialize import serialize_name
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.types import PdfName, PdfReference, PdfString


def internal_metadata_value(value: object) -> str:
    if not value or value in ("none", "null"):
        return "null"
    characters = list(cast(Any, value))
    if any(ord(character) > 255 for character in characters):
        data = b"\xfe\xff" + "".join(characters).encode("utf-16-be")
    else:
        data = bytes(
            ord(character) if ord(character) >= 32 or character in "\b\t\n\f\r" else 183
            for character in characters
        )
    return "<" + data.hex() + ">"


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
    return serialize_name(value).decode("ascii")


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
            token = format_object(
                item, depth=depth + (2 if depth else 1), column=current, ascii_only=ascii_only
            )
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


def internal_clone(value: object) -> Any:
    if isinstance(value, dict):
        return {key: internal_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [internal_clone(item) for item in value]
    return value


class ObjectAccess:
    def __init__(self, source: PdfDocument | None) -> None:
        self.source = source
        self.overrides: dict[int, object] = {}
        self.extra_count = 0
        self.trailer_override: dict[Any, Any] | None = None
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
            return 3 + self.extra_count
        declared_size = self.source.trailer_dict.get("Size", 0)
        return (
            max(
                int(declared_size) if isinstance(declared_size, (int, float)) else 0,
                max((key >> 16 for key in self.source.xref), default=0) + 1,
            )
            + self.extra_count
        )

    @property
    def trailer(self) -> dict[Any, Any]:
        if self.trailer_override is not None:
            return self.trailer_override
        if self.source is None:
            return {"Size": 3, "Root": PdfReference(1)}
        trailer = self.source.trailer_dict
        return trailer if "Size" in trailer else {"Size": self.length, **trailer}

    def resolve(self, value: object) -> object:
        if isinstance(value, PdfReference) and value.object_number in self.overrides:
            return self.overrides[value.object_number]
        if self.source is not None:
            return self.source.resolver.resolve(value)
        return self.empty.get(value.object_number) if isinstance(value, PdfReference) else value

    def get(self, xref: int) -> object:
        if xref == -1:
            return self.trailer
        if not 0 < xref < self.length:
            raise ValueError("bad xref")
        if xref in self.overrides:
            return self.overrides[xref]
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

    def page_references(self) -> list[PdfReference]:
        catalog = self.resolve(self.trailer.get("Root"))
        if not isinstance(catalog, dict):
            return []
        result: list[PdfReference] = []
        pending = [catalog.get("Pages")]
        seen: set[int] = set()
        while pending:
            ref = pending.pop()
            if isinstance(ref, PdfReference):
                if ref.object_number in seen:
                    continue
                seen.add(ref.object_number)
            obj = self.resolve(ref)
            if not isinstance(obj, dict):
                continue
            if str(obj.get("Type")) == "Page" and isinstance(ref, PdfReference):
                result.append(ref)
            else:
                kids = self.resolve(obj.get("Kids"))
                if isinstance(kids, list):
                    pending.extend(reversed(kids))
        return result

    def insert_blank_page(self, index: int, width: float, height: float) -> None:
        catalog = self.resolve(self.trailer.get("Root"))
        if not isinstance(catalog, dict):
            raise ValueError("invalid page tree")
        parent_ref = catalog.get("Pages")
        if not isinstance(parent_ref, PdfReference):
            raise ValueError("invalid page tree")
        parent = internal_clone(self.resolve(parent_ref))
        if not isinstance(parent, dict):
            raise ValueError("invalid page tree")
        pages = self.page_references()
        if index == 0 and pages:
            labels = self.resolve(catalog.get("PageLabels"))
            old_nums = labels.get("Nums") if isinstance(labels, dict) else None
            nums = (
                internal_clone(old_nums)
                if isinstance(old_nums, list)
                else [0, {"S": PdfName.of("D")}]
            )
            for offset in range(0, len(nums), 2):
                number = nums[offset]
                if not isinstance(number, int):
                    raise ValueError("invalid page label index")
                nums[offset] = number + 1
            updated_catalog = internal_clone(catalog)
            updated_catalog["PageLabels"] = {"Nums": [0, {"S": PdfName.of("D")}, *nums]}
            root = self.trailer["Root"]
            if isinstance(root, PdfReference):
                self.overrides[root.object_number] = updated_catalog
        resources = self.allocate()
        self.overrides[resources] = {}
        page = self.allocate()
        self.overrides[page] = {
            "Type": PdfName.of("Page"),
            "MediaBox": [0, 0, width, height],
            "Rotate": 0,
            "Resources": PdfReference(resources),
            "Parent": parent_ref,
        }
        pages.insert(index, PdfReference(page))
        parent["Kids"], parent["Count"] = pages, len(pages)
        self.overrides[parent_ref.object_number] = parent
        for ref in pages:
            child = internal_clone(self.get(ref.object_number))
            if isinstance(child, dict):
                ancestor = child.get("Parent")
                visited: set[int] = set()
                while isinstance(ancestor, PdfReference) and ancestor.object_number not in visited:
                    visited.add(ancestor.object_number)
                    inherited = self.resolve(ancestor)
                    if not isinstance(inherited, dict):
                        break
                    for key in ("Resources", "MediaBox", "CropBox", "Rotate"):
                        if key not in child and key in inherited:
                            child[key] = internal_clone(inherited.get(key))
                    ancestor = inherited.get("Parent")
                child["Parent"] = parent_ref
                self.overrides[ref.object_number] = child

    def allocate(self) -> int:
        number = self.length
        self.extra_count += 1
        self.overrides[number] = None
        return number

    def update(self, xref: int, text: str) -> None:
        if not 0 < xref < self.length:
            raise Exception("bad xref")
        previous = self.get(xref)
        obj = PdfLexer(text.encode("utf-8")).parse_object()
        if isinstance(previous, PdfStream) and isinstance(obj, dict):
            obj = previous.replace(dictionary=obj)
        self.overrides[xref] = obj

    def set_key(self, xref: int, key: str, value: str) -> None:
        obj = self.get(xref)
        stream = obj if isinstance(obj, PdfStream) else None
        dictionary = internal_clone(stream.dictionary if stream is not None else obj)
        if not isinstance(dictionary, dict):
            raise ValueError("not a dict (null)")
        parts = key.encode("utf-8", "surrogateescape").decode("latin-1").split("/")
        parent = dictionary
        for name in parts[:-1]:
            child = parent.get(name)
            if isinstance(child, PdfReference):
                raise NotImplementedError(
                    "writing through indirect dictionary paths is not implemented"
                )
            if not isinstance(child, dict):
                child = {}
                parent[name] = child
            parent = child
        parent[parts[-1]] = PdfLexer(value.encode("utf-8")).parse_object()
        replacement = stream.replace(dictionary=dictionary) if stream is not None else dictionary
        if xref == -1:
            self.trailer_override = dictionary
        else:
            self.overrides[xref] = replacement

    def update_stream(self, xref: int, data: bytes, *, compress: bool) -> None:
        obj = self.get(xref)
        dictionary = dict(obj.dictionary) if isinstance(obj, PdfStream) else internal_clone(obj)
        if not isinstance(dictionary, dict):
            raise ValueError("object is no PDF dict")
        dictionary.pop("Filter", None)
        dictionary.pop("DecodeParms", None)
        encoded = zlib.compress(data, 9) if compress else data
        if compress and len(encoded) < len(data):
            dictionary["Filter"] = PdfName.of("FlateDecode")
        else:
            encoded = data
        dictionary["Length"] = len(encoded)
        self.overrides[xref] = PdfStream(dictionary, encoded, spec=dictionary)

    def tobytes(self, *, no_new_id: bool = False, version: str = "1.7") -> bytes:
        objects: dict[int, tuple[int, object]] = {}
        if self.source is not None:
            for key, entry in self.source.xref.items():
                number, generation = key >> 16, key & 65535
                if (
                    number
                    and entry.in_use
                    and (number not in objects or generation >= objects[number][0])
                ):
                    objects[number] = (generation, self.get(number))
        else:
            objects.update((number, (0, obj)) for number, obj in self.empty.items())
        for number, obj in self.overrides.items():
            objects[number] = (objects.get(number, (0, None))[0], obj)
        trailer = dict(self.trailer)
        if not no_new_id:
            old_id = trailer.get("ID")
            first = old_id[0] if isinstance(old_id, list) and old_id else PdfString(uuid4().bytes)
            trailer["ID"] = [first, PdfString(uuid4().bytes)]
            self.trailer_override = trailer
        return write_pdf(objects, trailer, size=self.length, version=version)

    def metadata_text(self, key: str) -> str:
        info = self.resolve(self.trailer.get("Info"))
        value = self.resolve(info.get(key)) if isinstance(info, dict) else None
        if not isinstance(value, PdfString):
            return ""
        text = decode_pdf_text_string(value.data).split("\0", 1)[0]
        if not value.data.startswith((b"\xfe\xff", b"\xff\xfe", b"\xef\xbb\xbf")):
            text = text.replace("\x9f", "\0").replace("\xad", "\0")
        return text

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
