# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.9.2.2: text strings in PDFDocEncoding, UTF-16BE or UTF-8."""

from __future__ import annotations

import codecs

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.standards import PdfVersion, SemanticContext, require_recognized_version

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
PDFDOC_DECODE_MAP: str = "".join(PDFDOC_ENCODING_TABLE)


@cites(
    "ISO 32000-2",
    "7.9.2.2",
    since="1.2",
    rule="UTF-16BE text strings, after the byte order mark FE FF",
)
@cites(
    "ISO 32000-2",
    "7.9.2.2",
    since="2.0",
    rule="UTF-8 text strings, after the byte order mark EF BB BF",
)
@cites("ISO 32000-2", "D.3", since="1.3", rule="PDFDocEncoding defines byte A0 (Euro sign)")
def decode_pdf_text_string(
    data: bytes | memoryview, *, context: SemanticContext | None = None
) -> str:
    version = require_recognized_version(
        context, "text-string semantics require a recognized PDF version"
    )
    if type(data) is memoryview:
        data = data.tobytes()
    if data.startswith(b"\xfe\xff") and (version is None or version >= PdfVersion(1, 2)):
        try:
            return data[2:].decode("utf-16-be")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16BE data") from exc
    if data.startswith(b"\xef\xbb\xbf") and (version is None or version == PdfVersion(2, 0)):
        try:
            return data[3:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 data") from exc
    if version is not None and version < PdfVersion(1, 3) and 0xA0 in data:
        raise ValueError("undefined PDFDocEncoding byte 0xA0 before PDF 1.3")
    return codecs.charmap_decode(data, "strict", PDFDOC_DECODE_MAP)[0]


__all__ = ("PDFDOC_ENCODING_TABLE", "decode_pdf_text_string")
