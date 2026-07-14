from __future__ import annotations


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
    out = bytearray(data[:complete])
    for row_start in range(0, complete, bytes_per_row):
        row_end = row_start + bytes_per_row
        for i in range(row_start + colors, row_end):
            out[i] = (out[i] + out[i - colors]) & 0xFF
    return bytes(out)


def tiff_predict_16(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    bytes_per_row = colors * columns * 2
    if bytes_per_row <= 0:
        return b""
    n = len(data)
    complete = (n // bytes_per_row) * bytes_per_row
    out = bytearray(data[:complete])
    samples_per_row = colors * columns
    for row_start in range(0, complete, bytes_per_row):
        for i in range(colors, samples_per_row):
            offset = row_start + i * 2
            prev_offset = row_start + (i - colors) * 2
            cur = (out[offset] << 8) | out[offset + 1]
            prev_sample = (out[prev_offset] << 8) | out[prev_offset + 1]
            val = (cur + prev_sample) & 0xFFFF
            out[offset] = (val >> 8) & 0xFF
            out[offset + 1] = val & 0xFF
    return bytes(out)


def tiff_predict_bits(
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


def png_predict(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
    damaged_rows_before_error: bool = False,
) -> bytes:
    if bits_per_component not in {1, 2, 4, 8, 16}:
        raise PredictorError(f"invalid PNG predictor bits {bits_per_component}")
    bytes_per_pixel = max(1, (colors * bits_per_component + 7) // 8)
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    out = bytearray()
    n = len(data)
    pos = 0
    previous: bytes | bytearray = bytearray(row_length)
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
            row_bytes = data[pos : pos + rl]
            pos += rl
            out.extend(row_bytes)
            previous = (
                row_bytes if type(row_bytes) is bytearray else bytes(row_bytes)
            )
            continue
        row = bytearray(data[pos : pos + rl])
        pos += rl
        if filter_type == 1:
            for i in range(bpp, len(row)):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif filter_type == 2:
            for i in range(len(row)):
                row[i] = (row[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            n_row = len(row)
            for i in range(min(bpp, n_row)):
                row[i] = (row[i] + (previous[i] >> 1)) & 0xFF
            for i in range(bpp, n_row):
                row[i] = (row[i] + ((row[i - bpp] + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            n_row = len(row)
            for i in range(min(bpp, n_row)):
                row[i] = (row[i] + previous[i]) & 0xFF
            for i in range(bpp, n_row):
                left, up, up_left = row[i - bpp], previous[i], previous[i - bpp]
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                if pa <= pb and pa <= pc:
                    row[i] = (row[i] + left) & 0xFF
                elif pb <= pc:
                    row[i] = (row[i] + up) & 0xFF
                else:
                    row[i] = (row[i] + up_left) & 0xFF
        else:
            if damaged_rows_before_error:
                break
            raise UnsupportedPngFilterError(
                f"Unsupported PNG predictor filter {filter_type}"
            )
        out.extend(row)
        previous = row
    return bytes(out)


__all__ = (
    "PredictorError",
    "UnsupportedPngFilterError",
    "png_predict",
    "tiff_predict",
    "tiff_predict_8",
    "tiff_predict_16",
    "tiff_predict_bits",
)
