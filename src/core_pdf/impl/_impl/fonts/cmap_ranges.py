"""Application Unicode replacement and legacy range expansion."""

MAX_CMAP_RANGE_SPAN = 65536


def unicode_scalar_or_replacement(codepoint: int) -> str:
    if 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF:
        return chr(codepoint)
    return "\ufffd"


def expand_range(start: int, end: int, source_hex_len: int, base_dst: str) -> dict[bytes, str]:
    mapping: dict[bytes, str] = {}
    if (
        source_hex_len <= 0
        or end >= 1 << (source_hex_len * 8)
        or end < start
        or end - start + 1 > MAX_CMAP_RANGE_SPAN
    ):
        raise ValueError("invalid ToUnicode CMap bfrange")
    for i in range(start, end + 1):
        offset = i - start
        if not base_dst:
            mapping[i.to_bytes(source_hex_len, "big")] = ""
            continue
        units = [ord(c) for c in base_dst]
        units[-1] += offset
        mapping[i.to_bytes(source_hex_len, "big")] = "".join(
            unicode_scalar_or_replacement(u) for u in units
        )
    return mapping
