from core_pdf_spec.s_07_filters.codecs import apply_ascii85, apply_ascii_hex


def ascii85decode(data: bytes) -> bytes:
    # pdfminer accepts a missing or broken end marker and the optional opening marker.
    body = data[2:] if data.startswith(b"<~") else data
    return apply_ascii85(body.partition(b"~")[0] + b"~>", None)


def asciihexdecode(data: bytes) -> bytes:
    return apply_ascii_hex(data if b">" in data else data + b">", None)


__all__ = ("ascii85decode", "asciihexdecode")
