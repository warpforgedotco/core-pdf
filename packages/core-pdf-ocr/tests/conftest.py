from typing import Any, cast

import pytest

from core_pdf.impl.capture.program import PageProgram
from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf_ocr.impl.extract.contracts import PageAnalysis, PageEvidence


@pytest.fixture
def ocr_capture() -> PageAnalysis:
    return PageAnalysis(
        page=cast(Any, None),
        width=600,
        height=800,
        rotation=0,
        fields=(),
        annotations=(),
        program=PageProgram(),
        observations=ObservationBatch.empty(),
        evidence=PageEvidence(
            page_area=480_000,
            native_characters=0,
            visible_native_characters=0,
            suspicious_characters=0,
            image_count=0,
            image_area_ratio=0,
        ),
    )
