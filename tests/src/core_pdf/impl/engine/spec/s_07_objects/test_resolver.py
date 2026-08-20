from typing import cast

from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.objects import PdfReference
from core_pdf.impl.types import PdfDict


def test_resolves_demanded_object_missing_from_damaged_xref() -> None:
    data = b"%PDF-1.7\n154 0 obj\n<< /Type /Font >>\nendobj\n"
    resolver = ObjectResolver(data, {}, {}, recover_missing=True)

    try:
        resolved = resolver.resolve(PdfReference(154))
    finally:
        resolver.close()

    assert isinstance(resolved, dict)
    assert str(cast(PdfDict, resolved)["Type"]) == "Font"
