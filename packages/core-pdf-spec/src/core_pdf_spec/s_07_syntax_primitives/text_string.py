# SPDX-License-Identifier: AGPL-3.0-only
"""PDF 7.9.2.2 text strings: PDFDocEncoding and the BOM-prefixed encodings."""

from __future__ import annotations

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.standards import PdfVersion, SemanticContext

PDFDOC_ENCODING_OVERRIDES: dict[int, str] = {
    24: "˘",
    25: "ˇ",
    26: "ˆ",
    27: "˙",
    28: "˝",
    29: "˛",
    30: "˚",
    31: "˜",
    128: "•",
    129: "†",
    130: "‡",
    131: "…",
    132: "—",
    133: "–",
    134: "ƒ",
    135: "⁄",
    136: "‹",
    137: "›",
    138: "−",
    139: "‰",
    140: "„",
    141: "“",
    142: "”",
    143: "‘",
    144: "’",
    145: "‚",
    146: "™",
    147: "ﬁ",
    148: "ﬂ",
    149: "Ł",
    150: "Œ",
    151: "Š",
    152: "Ÿ",
    153: "Ž",
    154: "ı",
    155: "ł",
    156: "œ",
    157: "š",
    158: "ž",
    160: "€",
}

CHR_TABLE: list[str] = [chr(i) for i in range(256)]
PDFDOC_ENCODING_TABLE: list[str] = [
    PDFDOC_ENCODING_OVERRIDES.get(i, CHR_TABLE[i]) for i in range(256)
]


def decode_pdf_text_string(
    data: bytes | memoryview, *, context: SemanticContext | None = None
) -> str:
    """Decode a text string under the declared PDF version when context is supplied.

    Adobe PDF 1.2, 4.4 adds UTF-16BE BOM encoding; ISO 32000-2:2020, 7.9.2.2
    adds UTF-8 BOM encoding to PDF 2.0. Before each introduction the same bytes
    are ordinary PDFDocEncoding. Omitting context retains the historical
    all-encodings API; applications own best-effort recovery.

    Adobe PDF 1.3, Annex D.1 note 1 assigns the previously unused PDFDocEncoding
    byte A0 to Euro. With an earlier explicit version that byte has no assigned
    character, so decoding it as PDFDocEncoding raises ``ValueError``.
    """
    if context is not None and (context.version is None or not context.version.recognized):
        raise PdfUnsupportedError("text-string semantics require a recognized PDF version")
    if type(data) is memoryview:
        data = data.tobytes()
    version = context.version if context is not None else None
    if data.startswith(b"\xfe\xff") and (version is None or version >= PdfVersion(1, 2)):
        try:
            return data[2:].decode("utf-16-be")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16BE data") from exc
    if data.startswith(b"\xef\xbb\xbf") and (
        context is None or context.version == PdfVersion(2, 0)
    ):
        try:
            return data[3:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 data") from exc
    if version is not None and version < PdfVersion(1, 3) and 0xA0 in data:
        raise ValueError("undefined PDFDocEncoding byte 0xA0 before PDF 1.3")
    return "".join(PDFDOC_ENCODING_TABLE[b] for b in data)


__all__ = ("PDFDOC_ENCODING_TABLE", "decode_pdf_text_string")
