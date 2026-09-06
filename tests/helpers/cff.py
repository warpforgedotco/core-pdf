"""Small CFF programs used by format and font-service tests."""

import struct


def index(items: list[bytes], off_size: int = 1) -> bytes:
    """Serialise a CFF INDEX (section 5)."""
    if not items:
        return struct.pack(">H", 0)
    out = struct.pack(">H", len(items)) + bytes([off_size])
    offset = 1
    for item in items:
        out += offset.to_bytes(off_size, "big")
        offset += len(item)
    out += offset.to_bytes(off_size, "big")
    return out + b"".join(items)


def dict_op(operands: list[int], op: int) -> bytes:
    """Serialise DICT operands followed by a one-byte operator."""
    out = b""
    for value in operands:
        # 5-byte integer form keeps offsets patchable at a fixed width.
        out += b"\x1d" + struct.pack(">i", value)
    return out + bytes([op])


def build_cff(encoding_bytes: bytes, glyph_names: list[str]) -> bytes:
    """Build a minimal CFF with a custom Encoding and a format 0 charset."""
    header = bytes([1, 0, 4, 1])
    name_index = index([b"TestFont"])
    strings = [name.encode() for name in glyph_names]
    string_index = index(strings)
    global_subrs = index([])
    # One empty charstring per glyph, plus .notdef.
    charstrings = index([b"\x0e"] * (len(glyph_names) + 1))
    # Custom SIDs start after the 391 standard strings.
    charset = bytes([0]) + b"".join(struct.pack(">H", 391 + i) for i in range(len(glyph_names)))

    # Lay the pieces out after a fixed-size top DICT, then patch the offsets.
    top_placeholder = dict_op([0], 15) + dict_op([0], 16) + dict_op([0], 17)
    top_index_size = len(index([top_placeholder]))
    base = len(header) + len(name_index) + top_index_size + len(string_index) + len(global_subrs)
    charset_off = base
    encoding_off = charset_off + len(charset)
    charstrings_off = encoding_off + len(encoding_bytes)
    top = dict_op([charset_off], 15) + dict_op([encoding_off], 16) + dict_op([charstrings_off], 17)
    assert len(top) == len(top_placeholder)
    return (
        header
        + name_index
        + index([top])
        + string_index
        + global_subrs
        + charset
        + encoding_bytes
        + charstrings
    )
