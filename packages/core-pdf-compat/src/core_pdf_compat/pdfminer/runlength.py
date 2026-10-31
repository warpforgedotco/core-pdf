from core_pdf_spec.s_07_filters.codecs import apply_run_length


def rldecode(data: bytes) -> bytes:
    return apply_run_length(data, None)


__all__ = ("rldecode",)
