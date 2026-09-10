# SPDX-License-Identifier: AGPL-3.0-only
"""T.88 arithmetic direct symbol dictionaries and symbol-coded text regions."""

from core_pdf.impl.spec.s_07_filters.jbig2.arithmetic import (
    GENERIC_TEMPLATE_0_DEFAULT_AT,
    ArithmeticIntegers,
    JBIG2MQDecoder,
    decode_generic_bitmap,
    decode_refinement_bitmap,
)
from core_pdf.impl.spec.s_07_filters.jbig2.bitmap_kernels import compose_packed_bitmap_data
from core_pdf.impl.spec.s_07_filters.jbig2.structures import (
    MAX_BITMAP_PIXELS,
    MAX_INSTANCES,
    MAX_INTEGER_RUNS,
    MAX_SYMBOLS,
    JBIG2Image,
    Jbig2ParseError,
    Jbig2UnsupportedError,
    read_be_i8,
    read_be_u32,
)


def decode_symbol_dictionary(data: bytes, imported: list[JBIG2Image]) -> list[JBIG2Image]:
    """Decode direct arithmetic symbols and their export runs (6.5, 7.4.2)."""
    if len(data) < 2:
        raise Jbig2ParseError("truncated JBIG2 symbol dictionary")
    flags = int.from_bytes(data[:2], "big")
    if flags & 3:
        raise Jbig2UnsupportedError(
            "JBIG2 Huffman or aggregate symbol dictionaries are unsupported"
        )
    if flags & 0xF0FC:
        raise Jbig2ParseError("invalid JBIG2 arithmetic symbol dictionary flags")
    if flags & 0xF00:
        raise Jbig2UnsupportedError(
            "JBIG2 symbol template or inherited bitmap contexts unsupported"
        )
    if len(data) < 18:
        raise Jbig2ParseError("truncated JBIG2 symbol dictionary")
    at = tuple((read_be_i8(data, pos), read_be_i8(data, pos + 1)) for pos in range(2, 10, 2))
    if at != GENERIC_TEMPLATE_0_DEFAULT_AT:
        raise Jbig2UnsupportedError("unsupported JBIG2 symbol adaptive template")
    exported_count, new_count = read_be_u32(data, 10), read_be_u32(data, 14)
    total = len(imported) + new_count
    if exported_count > total:
        raise Jbig2ParseError("JBIG2 exported symbol count exceeds dictionary size")
    if total > MAX_SYMBOLS:
        raise Jbig2UnsupportedError("JBIG2 dictionary exceeds supported symbol limit")
    decoder = JBIG2MQDecoder.from_segment(data[18:])
    integers = ArithmeticIntegers(decoder)
    symbols = list(imported)
    height = 0
    decoded_pixels = 0
    height_classes = 0
    while len(symbols) < total:
        height_classes += 1
        if height_classes > MAX_INTEGER_RUNS:
            raise Jbig2UnsupportedError("JBIG2 dictionary exceeds supported height class limit")
        height += integers.integer("IADH")
        width = 0
        while (delta := integers.decode("IADW")) is not None:
            width += delta
            if len(symbols) >= total:
                raise Jbig2ParseError("JBIG2 height class exceeds declared symbol count")
            decoded_pixels += width * height
            if decoded_pixels > MAX_BITMAP_PIXELS:
                raise Jbig2UnsupportedError("JBIG2 dictionary exceeds supported pixel limit")
            symbols.append(decode_generic_bitmap(decoder, width, height))
    exported: list[JBIG2Image] = []
    index = 0
    export = False
    # Zero-length export runs are legal. Bound their number as well as symbols
    # so malformed arithmetic data cannot produce an unbounded no-progress loop.
    for _ in range(MAX_INTEGER_RUNS):
        if index == total:
            break
        count = integers.integer("IAEX")
        if count < 0 or count > total - index:
            raise Jbig2ParseError("invalid JBIG2 symbol export run")
        if export:
            exported.extend(symbols[index : index + count])
        index += count
        export = not export
    if index != total:
        raise Jbig2UnsupportedError("JBIG2 dictionary exceeds supported export run limit")
    if len(exported) != exported_count:
        raise Jbig2ParseError("JBIG2 symbol exports disagree with declared count")
    return exported


def internal_refined_symbol(
    integers: ArithmeticIntegers, contexts: list[int], symbol: JBIG2Image
) -> JBIG2Image:
    refinement = integers.integer("IARI")
    if refinement == 0:
        return symbol
    if refinement != 1:
        raise Jbig2ParseError("invalid JBIG2 symbol refinement indicator")
    dw, dh = integers.integer("IARDW"), integers.integer("IARDH")
    dx, dy = integers.integer("IARDX"), integers.integer("IARDY")
    return decode_refinement_bitmap(
        integers.decoder,
        contexts,
        symbol,
        symbol.width + dw,
        symbol.height + dh,
        dw // 2 + dx,
        dh // 2 + dy,
    )


def decode_text_bitmap(data: bytes, symbols: list[JBIG2Image]) -> JBIG2Image:
    """Decode arithmetic text strips, placement, and template-0 refinements."""
    if len(data) < 23:
        raise Jbig2ParseError("truncated JBIG2 text region header")
    flags = int.from_bytes(data[17:19], "big")
    if flags & 1:
        raise Jbig2UnsupportedError("JBIG2 Huffman text regions are unsupported")
    if flags & 0x8000:
        raise Jbig2UnsupportedError("unsupported JBIG2 text refinement template")
    operator = (flags >> 7) & 3
    if operator not in (0, 2):
        raise Jbig2UnsupportedError("unsupported JBIG2 text combination operator")
    refinement = bool(flags & 2)
    pos = 19
    if refinement:
        if len(data) < 27:
            raise Jbig2ParseError("truncated JBIG2 text refinement header")
        if data[pos : pos + 4] != b"\xff\xff\xff\xff":
            raise Jbig2UnsupportedError("unsupported JBIG2 text refinement adaptive template")
        pos += 4
    count = read_be_u32(data, pos)
    if count > MAX_INSTANCES or len(symbols) > MAX_SYMBOLS:
        raise Jbig2UnsupportedError("JBIG2 text region exceeds supported symbol limit")
    if count and not symbols:
        raise Jbig2ParseError("JBIG2 text region has no symbol dictionary")
    image = JBIG2Image.create(read_be_u32(data, 0), read_be_u32(data, 4))
    image.fill((flags >> 9) & 1)
    integers = ArithmeticIntegers(JBIG2MQDecoder.from_segment(data[pos + 4 :]), len(symbols))
    refinement_contexts = [0] * 8192
    strips = 1 << ((flags >> 2) & 3)
    corner = (flags >> 4) & 3
    transposed = bool(flags & 64)
    offset = (flags >> 10) & 31
    if offset >= 16:
        offset -= 32
    strip_t = -integers.integer("IADT") * strips
    first_s = 0
    decoded = 0
    decoded_pixels = 0
    while decoded < count:
        strip_t += integers.integer("IADT") * strips
        first_s += integers.integer("IAFS")
        current_s = first_s
        while True:
            if decoded >= count:
                raise Jbig2ParseError("JBIG2 strip exceeds declared instance count")
            current_t = integers.integer("IAIT") if strips > 1 else 0
            if current_t < 0 or current_t >= strips:
                raise Jbig2ParseError("invalid JBIG2 within-strip coordinate")
            symbol_id = integers.symbol_id()
            if symbol_id >= len(symbols):
                raise Jbig2ParseError("JBIG2 text symbol ID exceeds dictionary size")
            symbol = symbols[symbol_id]
            if refinement:
                symbol = internal_refined_symbol(integers, refinement_contexts, symbol)
            decoded_pixels += symbol.width * symbol.height
            if decoded_pixels > MAX_BITMAP_PIXELS:
                raise Jbig2UnsupportedError("JBIG2 text region exceeds supported pixel limit")
            before = (not transposed and corner >= 2) or (transposed and not corner & 1)
            advance = (symbol.height if transposed else symbol.width) - 1
            if before:
                current_s += advance
            x, y = (
                (strip_t + current_t, current_s) if transposed else (current_s, strip_t + current_t)
            )
            if corner >= 2:
                x -= symbol.width - 1
            if not corner & 1:
                y -= symbol.height - 1
            compose_packed_bitmap_data(
                symbol.data,
                symbol.height,
                symbol.width,
                x,
                y,
                image.width,
                image.height,
                image.stride,
                image.data,
                operator,
            )
            if not before:
                current_s += advance
            decoded += 1
            delta = integers.decode("IADS")
            if delta is None:
                break
            current_s += delta + offset
    return image
