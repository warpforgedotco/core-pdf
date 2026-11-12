# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.3: PDF objects as bytes, written by the compiled ObjectWriter.

The writer writes what core_pdf.impl.spec.s_07_syntax.serialize does byte for byte and hands
it the values it does not write itself (reals among them), so core and the facades get the
spec's bytes from compiled code. scripts/check_object_writer.py compares the two.
"""

from __future__ import annotations

from functools import partial

from core_pdf.impl.spec.s_07_syntax import serialize as spec
from core_pdf.impl.spec.s_07_syntax.serialize import ObjectLayout
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.types import PdfName, PdfReference, PdfString
from core_pdf_cythonized import ObjectWriter

FALLBACKS = {layout: partial(spec.serialize_object, layout=layout) for layout in ObjectLayout}


def serialize_object(value: object, layout: ObjectLayout = ObjectLayout.COMPACT) -> bytes:
    """serialize_object of the spec, compiled."""
    return ObjectWriter(
        layout is ObjectLayout.SPACED,
        PdfName,
        PdfString,
        PdfReference,
        PdfStream,
        FALLBACKS[layout],
    ).serialize(value)


def serialize_indirect_object(
    reference: PdfReference, value: object, layout: ObjectLayout = ObjectLayout.COMPACT
) -> bytes:
    """serialize_indirect_object of the spec, compiled."""
    header = b"%d %d obj\n" % (reference.object_number, reference.generation_number)
    return header + serialize_object(value, layout) + b"\nendobj\n"


__all__ = ["ObjectLayout", "serialize_indirect_object", "serialize_object"]
