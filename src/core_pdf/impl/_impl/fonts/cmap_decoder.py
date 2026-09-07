"""Application CMap aliases around the PDF CMap decoder."""

from __future__ import annotations

from core_pdf.impl._impl.fonts.cmap_tokenizer import decode_cmap_hex_token
from core_pdf.impl.spec.s_09_fonts.cmap_decoder import (
    CMapDecoder as PdfCMapDecoder,
)
from core_pdf.impl.spec.s_09_fonts.cmap_decoder import (
    CMapResourceResolver,
)


class CMapDecoder(PdfCMapDecoder):
    __slots__ = ()

    @staticmethod
    def decode_codespace_token(token: bytes) -> bytes:
        return decode_cmap_hex_token(token)

    @staticmethod
    def resolve_usecmap(
        name: str, *, usecmap_resolver: CMapResourceResolver | None, depth: int
    ) -> PdfCMapDecoder | None:
        if name in {"OneByteIdentityH", "OneByteIdentityV"}:
            return CMapDecoder.identity(byte_width=1, wmode=int(name.endswith("V")))
        if usecmap_resolver is not None:
            resolved = usecmap_resolver(name)
            if isinstance(resolved, PdfCMapDecoder):
                return resolved
            if resolved is not None:
                return CMapDecoder(
                    resolved, usecmap_resolver=usecmap_resolver, internal_depth=depth
                )
        return PdfCMapDecoder.resolve_usecmap(name, usecmap_resolver=None, depth=depth)
