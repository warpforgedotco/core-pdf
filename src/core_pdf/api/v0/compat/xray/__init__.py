from __future__ import annotations

import re

from core_pdf.api.types import PdfInput
from core_pdf.api.v0.compat._common import project_document
from core_pdf.api.v0.operations import BadRedactionOperation

from .._common import flip_box, open_source

_DATE_ONLY = re.compile(r"^[0-3]?\d[/\-][0-3]?\d[/\-]\d{2,4}$")


def inspect(source: PdfInput) -> dict[int, list[dict[str, object]]]:
    """Return x-ray-shaped bad-redaction results using only core-pdf locally."""
    with open_source(source) as document:
        report = BadRedactionOperation().run(project_document(document))
        output: dict[int, list[dict[str, object]]] = {}
        for finding in report.findings:
            bbox = finding.bbox
            if bbox is None or finding.page_number is None or not finding.evidence:
                continue
            page_height = bbox.space.height or 0.0
            output.setdefault(finding.page_number, []).append(
                {
                    "bbox": flip_box((bbox.x0, bbox.y0, bbox.x1, bbox.y1), page_height),
                    "text": finding.evidence[0].value,
                }
            )
        if output and all(
            _DATE_ONLY.fullmatch(str(item["text"]).strip())
            for findings in output.values()
            for item in findings
        ):
            return {}
        return output


__all__ = ("inspect",)
