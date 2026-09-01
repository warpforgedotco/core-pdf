from __future__ import annotations

import struct
import zlib

import imagecodecs
import numpy

TIFF_BITS_NUMPY_THRESHOLD = 1024
PNG_CODEC_THRESHOLD = 1024
# PDF predictor Colors -> PNG color type with identical sample layout.
internal_PNG_COLOR_TYPES = {1: 0, 3: 2, 4: 6}
internal_PNG_MAX_DIMENSION = 1_000_000
internal_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
internal_TIFF_SAMPLE_LUTS = {
    bits: numpy.asarray(
        [
            [
                (value >> (shift * bits)) & ((1 << bits) - 1)
                for shift in range(8 // bits - 1, -1, -1)
            ]
            for value in range(256)
        ],
        dtype=numpy.uint8,
    )
    for bits in (1, 2, 4)
}
for internal_table in internal_TIFF_SAMPLE_LUTS.values():
    internal_table.setflags(write=False)


class PredictorError(ValueError):
    """Invalid predictor parameters or data."""


class UnsupportedPngFilterError(PredictorError):
    """Unsupported PNG predictor row filter."""


def tiff_predict_8(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    bytes_per_row = colors * columns
    if bytes_per_row <= 0:
        return b""
    n = len(data)
    complete = (n // bytes_per_row) * bytes_per_row
    if complete == 0:
        return b""
    rows = (
        numpy.frombuffer(data, dtype=numpy.uint8, count=complete)
        .copy()
        .reshape(
            -1,
            columns,
            colors,
        )
    )
    rows[:] = numpy.cumsum(rows, axis=1, dtype=numpy.uint16).astype(numpy.uint8)
    return rows.tobytes()


def tiff_predict_16(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    bytes_per_row = colors * columns * 2
    if bytes_per_row <= 0:
        return b""
    n = len(data)
    complete = (n // bytes_per_row) * bytes_per_row
    rows = (
        numpy.frombuffer(data, dtype=">u2", count=complete // 2)
        .copy()
        .reshape(
            -1,
            columns,
            colors,
        )
    )
    decoded = numpy.cumsum(rows, axis=1, dtype=numpy.uint32).astype(">u2", copy=False)
    return decoded.tobytes()


def tiff_predict_bits(data: bytes | memoryview, columns: int, colors: int, bits: int) -> bytes:
    if len(data) >= TIFF_BITS_NUMPY_THRESHOLD:
        return internal_tiff_predict_bits_numpy(data, columns, colors, bits)

    return internal_tiff_predict_bits_scalar(data, columns, colors, bits)


def internal_tiff_predict_bits_scalar(
    data: bytes | memoryview, columns: int, colors: int, bits: int
) -> bytes:
    sample_count = colors * columns
    sample_mask = (1 << bits) - 1
    row_bit_length = sample_count * bits
    row_byte_length = max(1, (row_bit_length + 7) // 8)
    out = bytearray()
    pos = 0
    n = len(data)

    def unpack_samples(row_bytes: bytes) -> list[int]:
        samples: list[int] = []
        bit_buffer = 0
        bits_in_buffer = 0
        for byte in row_bytes:
            bit_buffer = (bit_buffer << 8) | byte
            bits_in_buffer += 8
            while bits_in_buffer >= bits and len(samples) < sample_count:
                bits_in_buffer -= bits
                samples.append((bit_buffer >> bits_in_buffer) & sample_mask)
                if bits_in_buffer:
                    bit_buffer &= (1 << bits_in_buffer) - 1
                else:
                    bit_buffer = 0
        if len(samples) < sample_count:
            samples.extend([0] * (sample_count - len(samples)))
        return samples

    def pack_samples(samples: list[int]) -> bytes:
        packed = bytearray()
        bit_buffer = 0
        bits_in_buffer = 0
        for sample in samples:
            bit_buffer = (bit_buffer << bits) | (sample & sample_mask)
            bits_in_buffer += bits
            while bits_in_buffer >= 8:
                bits_in_buffer -= 8
                packed.append((bit_buffer >> bits_in_buffer) & 0xFF)
                if bits_in_buffer:
                    bit_buffer &= (1 << bits_in_buffer) - 1
                else:
                    bit_buffer = 0
        if bits_in_buffer:
            packed.append((bit_buffer << (8 - bits_in_buffer)) & 0xFF)
        if len(packed) < row_byte_length:
            packed.extend(b"\x00" * (row_byte_length - len(packed)))
        return bytes(packed[:row_byte_length])

    while pos < n:
        if pos + row_byte_length > n:
            break
        row = bytes(data[pos : pos + row_byte_length])
        pos += row_byte_length
        samples = unpack_samples(row)
        for i in range(colors, sample_count):
            samples[i] = (samples[i] + samples[i - colors]) & sample_mask
        out.extend(pack_samples(samples))
    return bytes(out)


def internal_tiff_predict_bits_numpy(
    data: bytes | memoryview, columns: int, colors: int, bits: int
) -> bytes:
    sample_count = colors * columns
    row_byte_length = max(1, (sample_count * bits + 7) // 8)
    complete_rows = len(data) // row_byte_length
    if complete_rows == 0:
        return b""

    encoded = numpy.frombuffer(
        data,
        dtype=numpy.uint8,
        count=complete_rows * row_byte_length,
    ).reshape(complete_rows, row_byte_length)
    samples = internal_TIFF_SAMPLE_LUTS[bits][encoded].reshape(complete_rows, -1)[:, :sample_count]
    decoded_samples = (
        numpy.cumsum(
            samples.reshape(complete_rows, columns, colors),
            axis=1,
            dtype=numpy.uint16,
        )
        & ((1 << bits) - 1)
    ).reshape(complete_rows, -1)
    # Pack samples arithmetically (samples-per-byte shifted and ORed) instead
    # of expanding to one array element per bit for packbits.
    samples_per_byte = 8 // bits
    pad = (-decoded_samples.shape[1]) % samples_per_byte
    if pad:
        decoded_samples = numpy.pad(decoded_samples, ((0, 0), (0, pad)))
    grouped = decoded_samples.reshape(complete_rows, -1, samples_per_byte)
    packed = numpy.zeros(grouped.shape[:2], dtype=numpy.uint16)
    for sample_index in range(samples_per_byte):
        packed |= grouped[:, :, sample_index] << (bits * (samples_per_byte - 1 - sample_index))
    return packed.astype(numpy.uint8).tobytes()


def tiff_predict(
    data: bytes | memoryview, *, columns: int, colors: int, bits_per_component: int
) -> bytes:
    if bits_per_component == 8:
        return tiff_predict_8(data, columns, colors)
    if bits_per_component == 16:
        return tiff_predict_16(data, columns, colors)
    if bits_per_component not in {1, 2, 4}:
        raise PredictorError(f"invalid TIFF predictor bits {bits_per_component}")
    return tiff_predict_bits(data, columns, colors, bits_per_component)


def internal_png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload))
    )


def internal_png_predict_codec(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
) -> bytes | None:
    """Unfilter PNG-predicted rows with libpng via a minimal PNG container.

    The filtered stream is byte-for-byte PNG scanline data, so wrapping it in
    IHDR/IDAT/IEND (stored-mode zlib, ~memcpy cost) lets imagecodecs run the
    row unfilter in C. Returns ``None`` when the parameter combination has no
    PNG equivalent; damaged data raises and the caller falls back to the
    scalar path, which reproduces the exact error/partial-output semantics.
    """
    color_type = internal_PNG_COLOR_TYPES.get(colors)
    if color_type is None:
        return None
    # Sub-byte depths exist only for grayscale, and the decoder's sample
    # expansion drops row padding bits, so require byte-aligned rows to
    # stay byte-identical with the scalar path.
    if bits_per_component not in (8, 16) and (
        color_type != 0 or (columns * bits_per_component) % 8
    ):
        return None
    if not 1 <= columns <= internal_PNG_MAX_DIMENSION:
        return None
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    rows = len(data) // (row_length + 1)
    if not 1 <= rows <= internal_PNG_MAX_DIMENSION:
        return None
    body = memoryview(data)[: rows * (row_length + 1)]
    header = struct.pack(">IIBBBBB", columns, rows, bits_per_component, color_type, 0, 0, 0)
    png = b"".join(
        (
            internal_PNG_SIGNATURE,
            internal_png_chunk(b"IHDR", header),
            internal_png_chunk(b"IDAT", zlib.compress(body, 0)),
            internal_png_chunk(b"IEND", b""),
        )
    )
    decoded = numpy.asarray(imagecodecs.png_decode(png))
    if bits_per_component == 16:
        return decoded.astype(">u2", copy=False).tobytes()
    if bits_per_component == 8:
        return decoded.tobytes()
    # Sub-byte gray comes back expanded to one byte per sample, scaled by the
    # exact factor 255 // (2**bits - 1); undo the scaling and repack.
    bits = bits_per_component
    samples = decoded.reshape(rows, columns) // (255 // ((1 << bits) - 1))
    per_byte = 8 // bits
    grouped = samples.reshape(rows, -1, per_byte)
    packed = numpy.zeros(grouped.shape[:2], dtype=numpy.uint8)
    for sample_index in range(per_byte):
        packed |= grouped[:, :, sample_index] << (bits * (per_byte - 1 - sample_index))
    return packed.tobytes()


def png_predict(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
    damaged_rows_before_error: int = 0,
) -> bytes:
    if bits_per_component not in {1, 2, 4, 8, 16}:
        raise PredictorError(f"invalid PNG predictor bits {bits_per_component}")
    if len(data) >= PNG_CODEC_THRESHOLD:
        try:
            decoded = internal_png_predict_codec(
                data,
                columns=columns,
                colors=colors,
                bits_per_component=bits_per_component,
            )
        except Exception:
            decoded = None
        if decoded is not None:
            return decoded
    bytes_per_pixel = max(1, (colors * bits_per_component + 7) // 8)
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    n = len(data)
    out = bytearray((n // (row_length + 1)) * row_length)
    out_view = numpy.frombuffer(out, dtype=numpy.uint8)
    out_pos = 0
    pos = 0
    previous: bytes | bytearray | memoryview | numpy.ndarray = bytearray(row_length)
    bpp = bytes_per_pixel
    rl = row_length
    while pos < n:
        if pos + 1 > n:
            break
        filter_type = data[pos]
        pos += 1
        if pos + rl > n:
            break
        if filter_type == 0:
            raw_row = memoryview(data)[pos : pos + rl]
            pos += rl
            out_view[out_pos : out_pos + rl] = numpy.frombuffer(raw_row, dtype=numpy.uint8)
            out_pos += rl
            previous = raw_row
            continue
        if filter_type == 1:
            row_array = numpy.frombuffer(data, dtype=numpy.uint8, count=rl, offset=pos).copy()
            for offset in range(min(bpp, len(row_array))):
                row_array[offset::bpp] = numpy.cumsum(
                    row_array[offset::bpp],
                    dtype=numpy.uint16,
                ).astype(numpy.uint8, copy=False)
            row: bytes | bytearray | memoryview | numpy.ndarray = row_array
        elif filter_type == 2:
            row_array = numpy.frombuffer(data, dtype=numpy.uint8, count=rl, offset=pos).copy()
            row_array[:] = (
                row_array.astype(numpy.uint16) + numpy.frombuffer(previous, dtype=numpy.uint8)
            ).astype(numpy.uint8)
            row = row_array
        elif filter_type == 3:
            row_bytes = bytearray(data[pos : pos + rl])
            n_row = len(row_bytes)
            previous_array = previous.tobytes() if isinstance(previous, numpy.ndarray) else previous
            first = min(bpp, n_row)
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + (previous_array[i] >> 1)) & 0xFF
            for i in range(bpp, n_row):
                row_bytes[i] = (
                    row_bytes[i] + ((row_bytes[i - bpp] + previous_array[i]) >> 1)
                ) & 0xFF
            row = row_bytes
        elif filter_type == 4:
            row_bytes = bytearray(data[pos : pos + rl])
            n_row = len(row_bytes)
            previous_array = previous.tobytes() if isinstance(previous, numpy.ndarray) else previous
            first = min(bpp, n_row)
            for i in range(first):
                row_bytes[i] = (row_bytes[i] + previous_array[i]) & 0xFF
            for i in range(bpp, n_row):
                left, up, up_left = (
                    row_bytes[i - bpp],
                    previous_array[i],
                    previous_array[i - bpp],
                )
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                if pa <= pb and pa <= pc:
                    row_bytes[i] = (row_bytes[i] + left) & 0xFF
                elif pb <= pc:
                    row_bytes[i] = (row_bytes[i] + up) & 0xFF
                else:
                    row_bytes[i] = (row_bytes[i] + up_left) & 0xFF
            row = row_bytes
        else:
            if damaged_rows_before_error:
                break
            raise UnsupportedPngFilterError(f"Unsupported PNG predictor filter {filter_type}")
        pos += rl
        out_view[out_pos : out_pos + rl] = row
        out_pos += rl
        previous = row
    return bytes(out[:out_pos])


__all__ = (
    "PredictorError",
    "UnsupportedPngFilterError",
    "png_predict",
    "tiff_predict",
    "tiff_predict_8",
    "tiff_predict_16",
    "tiff_predict_bits",
)
