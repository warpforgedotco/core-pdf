# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import PdfExtension, PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfString


def parse_header_version(data: bytes | memoryview) -> PdfVersion:
    header = bytes(data[:9])
    if len(header) < 9 or header[:5] != b"%PDF-" or header[8:9] not in (b"\r", b"\n"):
        raise ValueError("invalid PDF header")
    try:
        return PdfVersion.parse(header[5:8].decode("ascii"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid PDF header version") from exc


def parse_catalog_version(value: object) -> PdfVersion:
    if not isinstance(value, PdfName):
        raise ValueError("catalog Version must be a PDF name")
    return PdfVersion.parse(value.value)


def effective_pdf_version(
    header: PdfVersion | None,
    catalog: PdfVersion | None,
    *,
    previous: PdfVersion | None = None,
) -> PdfVersion | None:
    return max(
        (version for version in (header, catalog, previous) if version is not None), default=None
    )


def optional_string(dictionary: dict, key: str) -> str | None:
    value = dictionary.get(key)
    if value is None:
        return None
    if not isinstance(value, PdfString):
        raise ValueError(f"developer extension {key} must be a direct PDF string")
    return decode_pdf_text_string(value.data)


def parse_extension(prefix: str, value: object) -> PdfExtension:
    if not isinstance(prefix, str) or not prefix or prefix == "Type":
        raise ValueError("invalid developer extension prefix")
    if not isinstance(value, dict):
        raise ValueError("developer extension must be a direct dictionary")
    kind = value.get("Type")
    if kind is not None and (not isinstance(kind, PdfName) or kind.value != "DeveloperExtensions"):
        raise ValueError("invalid developer extension Type")
    base = value.get("BaseVersion")
    if not isinstance(base, PdfName):
        raise ValueError("developer extension BaseVersion must be a direct PDF name")
    level = value.get("ExtensionLevel")
    if type(level) is not int or level <= 0:
        raise ValueError("developer extension ExtensionLevel must be a positive direct integer")
    return PdfExtension(
        prefix,
        PdfVersion.parse(base.value),
        level,
        optional_string(value, "URL"),
        optional_string(value, "ExtensionRevision"),
    )


def parse_extensions(
    value: object, *, context: SemanticContext | None = None
) -> tuple[PdfExtension, ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("Extensions must be a direct dictionary")
    kind = value.get("Type")
    if kind is not None and (not isinstance(kind, PdfName) or kind.value != "Extensions"):
        raise ValueError("invalid Extensions Type")
    extensions: list[PdfExtension] = []
    for key, raw in value.items():
        if isinstance(key, PdfName):
            prefix = key.value
        elif isinstance(key, str):
            prefix = key
        elif isinstance(key, bytes):
            prefix = key.decode("latin-1")
        else:
            raise ValueError("invalid Extensions dictionary key")
        if prefix == "Type":
            continue
        if isinstance(raw, list):
            if not raw:
                raise ValueError("developer extensions array must not be empty")
            if (
                context is not None
                and context.version is not None
                and context.version < PdfVersion(2, 0)
            ):
                raise ValueError("developer extensions arrays require PDF 2.0")
            values = raw
        else:
            values = [raw]
        for declaration in values:
            extension = parse_extension(prefix, declaration)
            if context is not None and context.version is not None:
                if extension.base_version > context.version:
                    raise ValueError("extension BaseVersion exceeds effective PDF version")
                if context.version >= PdfVersion(2, 0) and extension.url is None:
                    raise ValueError("PDF 2.0 developer extensions require URL")
            extensions.append(extension)
    return tuple(extensions)


__all__ = (
    "effective_pdf_version",
    "parse_catalog_version",
    "parse_extension",
    "parse_extensions",
    "parse_header_version",
)
