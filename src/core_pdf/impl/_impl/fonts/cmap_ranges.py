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
    prefix = "".join(unicode_scalar_or_replacement(ord(c)) for c in base_dst[:-1])
    final_scalar = ord(base_dst[-1]) if base_dst else None
    for i in range(start, end + 1):
        mapping[i.to_bytes(source_hex_len, "big")] = (
            ""
            if final_scalar is None
            else prefix + unicode_scalar_or_replacement(final_scalar + i - start)
        )
    return mapping
