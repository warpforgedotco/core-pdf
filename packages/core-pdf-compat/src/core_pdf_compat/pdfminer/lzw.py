from core_pdf_spec.s_07_filters.codecs import apply_lzw


def lzwdecode(data: bytes) -> bytes:
    return apply_lzw(data, None)


__all__ = ("lzwdecode",)
