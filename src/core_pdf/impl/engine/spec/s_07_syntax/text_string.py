# SPDX-License-Identifier: AGPL-3.0-only
"""PDF 7.9.2.2 text strings: PDFDocEncoding and the BOM-prefixed encodings."""

from __future__ import annotations

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


def decode_pdf_text_string(data: bytes | memoryview) -> str:
    if type(data) is memoryview:
        data = data.tobytes()
    if data.startswith(b"\xfe\xff"):
        try:
            return data[2:].decode("utf-16-be")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16BE data") from exc
    if data.startswith(b"\xff\xfe"):
        try:
            return data[2:].decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16LE data") from exc
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data[3:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 data") from exc
    return "".join(PDFDOC_ENCODING_TABLE[b] for b in data)
