from __future__ import annotations

import os

import pytest

from core_pdf.impl.extract import pipeline as parse_pipeline
from core_pdf.impl.extract.contracts import PagePlanReason, PageRoute, WorkPlan

from .support import FULL_ENV


@pytest.fixture(autouse=True)
def native_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every page natively while comparing against reference libraries.

    The reference readers never OCR, and the facades drop recognized text before
    comparing, so OCR passes only add raster and Tesseract time to output the
    equality assertion cannot depend on. The full corpus run keeps real routing.
    """
    if os.environ.get(FULL_ENV) != "1":
        monkeypatch.setattr(
            parse_pipeline,
            "plan_page",
            lambda capture: WorkPlan(PageRoute.NATIVE, reason=PagePlanReason.UNSPECIFIED),
        )
