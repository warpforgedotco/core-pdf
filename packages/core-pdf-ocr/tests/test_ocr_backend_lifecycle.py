import pytest

from core_pdf_ocr.impl.extract.ocr import tesseract as ocr_tesseract


def test_backend_is_created_for_each_session(monkeypatch: pytest.MonkeyPatch) -> None:
    import tesserocr

    class FakeApi:
        def __init__(self, **internal_kwargs: object) -> None:
            pass

        def SetVariable(self, *internal_args: object) -> None:
            pass

        def SetPageSegMode(self, internal_mode: int) -> None:
            pass

    monkeypatch.setattr(tesserocr, "PyTessBaseAPI", FakeApi)
    # Resolving real tessdata is not part of this lifecycle test.
    monkeypatch.setattr(ocr_tesseract, "internal_tessdata_path", lambda: "")
    first = ocr_tesseract.internal_api(3)
    second = ocr_tesseract.internal_api(3)

    assert first is not second


def test_shared_scorer_rejects_legacy_ocr_flag() -> None:
    from scripts.score_unstructured_bench import ScoreBench

    with pytest.raises(SystemExit):
        ScoreBench.from_cli(["--ocr"])
