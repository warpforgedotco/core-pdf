from __future__ import annotations

from typing import Any, cast

import pytest

from core_pdf.impl.engine.extraction.ocr.session import OcrPageSession, PreparedOcrImage
from core_pdf.impl.engine.extraction.ocr.types import OcrImage, OcrTextResult


def test_scaled_single_region_keeps_source_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = OcrPageSession()
    backend = cast(Any, object())
    session._backend = backend
    session._backend_loaded = True
    prepared = PreparedOcrImage(1, False, "source", ("variant",))
    monkeypatch.setattr(session, "prepared_image", lambda _image: prepared)
    captured: dict[str, object] = {}

    def recognize(
        _backend: object,
        _prepared: PreparedOcrImage,
        _image: OcrImage,
        *,
        psm: int,
        variables: object,
        rectangle: tuple[int, int, int, int] | None = None,
    ) -> OcrTextResult:
        captured.update(psm=psm, variables=variables, rectangle=rectangle)
        return OcrTextResult("row", 90)

    monkeypatch.setattr(session, "_image_to_text_result_from_prepared", recognize)
    image = OcrImage(
        b"\xff" * 10_000,
        100,
        100,
        1,
        100,
        target_width=400,
        target_height=400,
    )

    result = session.image_region_to_text_result(
        image,
        (10, 20, 80, 90),
        psm=7,
    )

    assert result.text == "row"
    assert captured["rectangle"] == (10, 20, 80, 90)
