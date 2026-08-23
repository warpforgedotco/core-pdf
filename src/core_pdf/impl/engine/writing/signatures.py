# SPDX-License-Identifier: AGPL-3.0-only
"""PDF signature containers with externally supplied CMS signers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PdfSignatureProvider(Protocol):
    """Produce a detached CMS/PKCS#7 signature for PDF ByteRange data."""

    def sign(self, data: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class PdfSignaturePlan:
    """External signer and reserved contents size used while building a PDF."""

    provider: PdfSignatureProvider
    contents_length: int = 8192

    def __post_init__(self) -> None:
        if self.contents_length <= 0:
            raise ValueError("signature contents length must be positive")


def apply_signature_plan(
    pdf: bytes,
    plan: PdfSignaturePlan,
    *,
    byte_range_placeholder: bytes = b"[0 0000000000 0000000000 0000000000]",
) -> bytes:
    """Replace signature placeholders and invoke the external signer."""
    contents_placeholder = b"<" + b"0" * (plan.contents_length * 2) + b">"
    contents_position = internal_unique_position(pdf, contents_placeholder, "signature contents")
    byte_range_position = internal_unique_position(
        pdf, byte_range_placeholder, "signature ByteRange"
    )
    contents_end = contents_position + len(contents_placeholder)
    byte_range = (
        f"[0 {contents_position:010d} {contents_end:010d} {len(pdf) - contents_end:010d}]".encode(
            "ascii"
        )
    )
    if len(byte_range) != len(byte_range_placeholder):
        raise ValueError("PDF signature ByteRange exceeds fixed placeholder width")
    result = bytearray(pdf)
    result[byte_range_position : byte_range_position + len(byte_range)] = byte_range
    signed_data = bytes(result[:contents_position] + result[contents_end:])
    signature = plan.provider.sign(signed_data)
    if len(signature) > plan.contents_length:
        raise ValueError("external signature exceeds reserved PDF contents space")
    signed_contents = b"<" + signature.hex().upper().encode("ascii")
    signed_contents += b"0" * (plan.contents_length * 2 - len(signature) * 2) + b">"
    result[contents_position:contents_end] = signed_contents
    return bytes(result)


def internal_unique_position(data: bytes, marker: bytes, label: str) -> int:
    position = data.find(marker)
    if position < 0 or data.find(marker, position + 1) >= 0:
        raise ValueError(f"expected exactly one {label} placeholder")
    return position


__all__ = ("PdfSignaturePlan", "PdfSignatureProvider", "apply_signature_plan")
