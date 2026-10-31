from core_pdf_spec.s_07_filters.codecs import apply_ascii85, apply_ascii_hex


def ascii85decode(data: bytes) -> bytes:
    start = data.find(b"<~")
    body = data[start + 2 :] if start >= 0 else data
    end = body.find(b"~>")
    return apply_ascii85((body if end < 0 else body[:end]) + b"~>", None)


def asciihexdecode(data: bytes) -> bytes:
    return apply_ascii_hex(data if b">" in data else data + b">", None)


__all__ = ("ascii85decode", "asciihexdecode")
