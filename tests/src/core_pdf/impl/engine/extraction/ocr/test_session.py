from __future__ import annotations

from typing import Any, cast

import pytest
from core_ocr.impl import deskew
from core_ocr.impl.types import OcrImage, OcrTextResult

from core_pdf.impl.engine.extraction.ocr.session import OcrPageSession, PreparedOcrImage


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


class _DestroyingLeptonica:
    def __init__(self) -> None:
        self.destroyed: list[int] = []

    def pixDestroy(self, pix: Any) -> None:
        self.destroyed.append(int(pix._obj.value))


class _DeskewingBackend:
    def __init__(self) -> None:
        self.leptonica = _DestroyingLeptonica()
        self.source_calls = 0
        self.deskew_calls = 0

    def should_use_pix(self, _image: OcrImage) -> bool:
        return True

    def source_pix_from_image(self, _image: OcrImage) -> int:
        self.source_calls += 1
        return 10

    def scale_source_pix(self, source_pix: int, _image: OcrImage) -> int:
        return source_pix

    def deskew_pix(
        self,
        _pix: int,
        *,
        source: str,
        width: int,
        height: int,
    ) -> tuple[int, object]:
        self.deskew_calls += 1
        return (
            20,
            deskew.deskew_info(
                source=source,
                angle_degrees=1.25,
                confidence=7.5,
                applied=True,
                reason="applied",
                image_width=width,
                image_height=height,
            ),
        )


def test_full_page_deskew_is_cached_across_ocr_retries() -> None:
    session = OcrPageSession()
    backend = _DeskewingBackend()
    session._backend = cast(Any, backend)
    session._backend_loaded = True
    image = OcrImage(
        b"encoded",
        800,
        1000,
        4,
        3200,
        encoded=b"encoded",
        source="rendered_page_300dpi",
        cache_key="page",
        resolution=300,
    )

    first = session.prepared_image(image)
    second = session.prepared_image(image)

    assert first is second
    assert first is not None
    assert first.pix == 20
    assert first.deskew_info is not None
    assert first.deskew_info.applied
    assert backend.source_calls == 1
    assert backend.deskew_calls == 1
    assert session.deskew_diagnostics() == (
        {
            "source": "rendered_page_300dpi",
            "angle_degrees": 1.25,
            "confidence": 7.5,
            "applied": True,
            "reason": "applied",
            "image_width": 800,
            "image_height": 1000,
            "source_to_deskew": first.deskew_info.source_to_deskew,
            "deskew_to_source": first.deskew_info.deskew_to_source,
        },
    )

    session.close()

    assert sorted(backend.leptonica.destroyed) == [10, 20]


def test_non_page_image_is_not_deskewed() -> None:
    session = OcrPageSession()
    backend = _DeskewingBackend()
    session._backend = cast(Any, backend)
    session._backend_loaded = True
    image = OcrImage(
        b"encoded",
        100,
        100,
        4,
        400,
        encoded=b"encoded",
        source="embedded_figure",
        cache_key="figure",
    )

    prepared = session.prepared_image(image)

    assert prepared is not None
    assert prepared.pix == 10
    assert prepared.deskew_info is None
    assert backend.deskew_calls == 0
