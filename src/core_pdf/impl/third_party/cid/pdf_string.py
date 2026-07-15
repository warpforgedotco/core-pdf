from __future__ import annotations

STRING_ESCAPE = {
    98: b"\b",
    102: b"\f",
    110: b"\n",
    114: b"\r",
    116: b"\t",
    40: b"(",
    41: b")",
    92: b"\\",
}


def decode_pdf_literal_string(data: bytes | bytearray | memoryview) -> bytes:
    raw = memoryview(data)
    n = len(raw)
    if n < 2 or raw[0] != 40:
        raise ValueError("invalid PDF literal string")

    pos = 1
    out = bytearray()
    depth = 1
    while pos < n:
        byte = raw[pos]
        pos += 1
        if byte == 40:
            depth += 1
            out.append(byte)
        elif byte == 41:
            depth -= 1
            if depth == 0:
                return bytes(out)
            out.append(byte)
        elif byte == 92:
            if pos < n:
                esc = raw[pos]
                pos += 1
                if 48 <= esc <= 55:
                    oct_val = esc - 48
                    count = 1
                    while count < 3 and pos < n and 48 <= raw[pos] <= 55:
                        oct_val = (oct_val << 3) | (raw[pos] - 48)
                        pos += 1
                        count += 1
                    out.append(oct_val & 0xFF)
                elif esc == 10:
                    if pos < n and raw[pos] == 13:
                        pos += 1
                elif esc == 13:
                    if pos < n and raw[pos] == 10:
                        pos += 1
                else:
                    mapped = STRING_ESCAPE.get(esc)
                    if mapped is None:
                        out.append(esc)
                    else:
                        out.extend(mapped)
        elif byte == 13 or byte == 10:
            out.append(10)
            if pos < n:
                next_byte = raw[pos]
                if (byte == 13 and next_byte == 10) or (byte == 10 and next_byte == 13):
                    pos += 1
        else:
            out.append(byte)
    raise ValueError("unterminated PDF literal string")
