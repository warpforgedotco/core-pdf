from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import cast

import pytest

from core_pdf.impl.engine.execution import TaskScope
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.parse import (
    CapturedPage,
    ObservationBatch,
    ObservationSource,
    OcrPass,
    OcrPassScope,
    PageEvidence,
    PagePlanReason,
    PageRoute,
    RecognitionReport,
    RecognitionResult,
    WorkPlan,
    ocr,
)
from core_pdf.impl.engine.parse import capture as parse_capture
from core_pdf.impl.engine.parse import ocr as parse_ocr
from core_pdf.impl.engine.parse import pipeline as parse_pipeline
from core_pdf.impl.engine.parse.model import StrokedVectorTextEvidence
from core_pdf.impl.engine.render.raster_image import RasterImage


def raster(
    data: bytes, width: int, height: int, channels: int, resolution: int
) -> ocr.internal_Raster:
    return ocr.internal_Raster(RasterImage(data, width, height, channels), resolution)


def candidate_observations(text: str, confidence: float) -> ObservationBatch:
    return ObservationBatch.from_columns(
        (text,),
        ((0.0, 0.0, 1.0, 1.0),),
        source=ObservationSource.OCR,
        confidence=(confidence,),
    )


def token_observations(
    tokens: tuple[str, ...],
    *,
    offset_x: float = 0.0,
    source: ObservationSource = ObservationSource.OCR,
) -> ObservationBatch:
    return ObservationBatch.from_columns(
        tokens,
        (
            (offset_x + index * 10.0, 10.0, offset_x + index * 10.0 + 8.0, 18.0)
            for index in range(len(tokens))
        ),
        source=source,
        confidence=(95.0 for _ in tokens),
    )


def internal_recognize_with_report(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
) -> tuple[ObservationBatch, RecognitionReport]:
    trace = ocr.internal_RecognitionTrace.create()
    observations = ocr.internal_recognize_page_with_reserved_raster(
        capture,
        plan,
        context,
        trace=trace,
    )
    return observations, trace.report()


def internal_report_mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


def test_tessdata_prefix_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tessdata = tmp_path / "configured-tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"test")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tessdata))
    ocr.internal_tessdata_path.cache_clear()

    try:
        assert ocr.internal_tessdata_path() == str(tessdata)
    finally:
        ocr.internal_tessdata_path.cache_clear()


def test_invalid_tessdata_prefix_has_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    ocr.internal_tessdata_path.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="eng.traineddata"):
            ocr.internal_tessdata_path()
    finally:
        ocr.internal_tessdata_path.cache_clear()


def test_tessdata_path_falls_back_to_tesseract_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tessdata = tmp_path / "cli-tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"test")
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    monkeypatch.setattr(ocr.internal_TESSEROCR, "get_languages", lambda: ("./", ()))
    monkeypatch.setattr(ocr.shutil, "which", lambda internal_name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        ocr.subprocess,
        "run",
        lambda *internal_args, **internal_kwargs: SimpleNamespace(
            stdout=f'List of available languages in "{tessdata}" (1):\neng\n',
            stderr="",
        ),
    )
    ocr.internal_tessdata_path.cache_clear()

    try:
        assert ocr.internal_tessdata_path() == str(tessdata)
    finally:
        ocr.internal_tessdata_path.cache_clear()


def test_prewarm_runtime_starts_workers_and_initializes_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ocr.RUNTIME, "prewarm", lambda: calls.append("workers"))
    monkeypatch.setattr(parse_ocr, "internal_prepare_ocr", lambda: calls.append("ocr"))

    ocr.prewarm_runtime()

    assert calls == ["workers", "ocr"]


def test_low_confidence_standalone_punctuation_is_rejected() -> None:
    assert not ocr.internal_acceptable_text("|", 65.0)
    assert ocr.internal_acceptable_text("R1", 65.0)
    assert ocr.internal_acceptable_text("R-1", 65.0)


def test_single_non_ascii_symbol_observations_are_rejected() -> None:
    # Braille cells and other non-ASCII symbols are common OCR garbage from
    # blank/decorative regions and should never form their own observations.
    assert not ocr.internal_acceptable_text("⠭", 99.0)
    assert not ocr.internal_acceptable_text("⠬", 90.0)
    assert not ocr.internal_acceptable_text("©", 99.0)
    # Acute-accented letters are alphanumeric and remain acceptable.
    assert ocr.internal_acceptable_text("é", 99.0)
    # ASCII punctuation keeps the existing confidence gate (no behavior change).
    assert not ocr.internal_acceptable_text("|", 65.0)
    assert ocr.internal_acceptable_text("|", 75.0)


def page_evidence(*, image_count: int = 0, image_area_ratio: float = 0.0) -> PageEvidence:
    return PageEvidence(
        page_area=100.0,
        native_characters=0,
        visible_native_characters=0,
        suspicious_characters=0,
        image_count=image_count,
        image_area_ratio=image_area_ratio,
        vector_complexity=0,
    )


def test_dominant_image_prefers_source_resolution_for_equal_display_area(
    monkeypatch,
) -> None:
    background = raster(bytes((255, 255, 255)), 1, 1, 3, 70)
    scan = raster(bytes(300 * 400 * 3), 300, 400, 3, 100)
    page_box = (0.0, 0.0, 600.0, 800.0)
    monkeypatch.setattr(
        parse_ocr,
        "internal_page_image_regions",
        lambda capture, minimum_area_ratio, **internal_kwargs: (
            ocr.internal_RasterRegion(background, page_box),
            ocr.internal_RasterRegion(scan, page_box),
        ),
    )

    capture = cast(CapturedPage, SimpleNamespace())
    region = ocr.internal_dominant_image_region(capture)
    assert region is not None
    assert region.raster is scan


def test_decoded_image_falls_back_when_shared_source_cannot_decode(monkeypatch) -> None:
    image = SimpleNamespace(
        image_source=SimpleNamespace(decode=lambda: None),
        raw_data=b"encoded",
        dictionary={"Width": 2, "Height": 1},
    )
    monkeypatch.setattr(
        parse_ocr,
        "decode_pdf_image",
        lambda internal_raw, internal_dictionary: SimpleNamespace(
            data=bytes((1, 2, 3, 4, 5, 6)),
            width=2,
            height=1,
            channels=3,
        ),
    )

    decoded = ocr.internal_decoded_image_raster(image, 2.0)

    assert decoded is not None
    assert (decoded.width, decoded.height, decoded.image.channels) == (11, 5, 3)


def test_dominant_image_defers_layered_scans_to_compositor(monkeypatch) -> None:
    left = raster(bytes(300 * 400 * 3), 300, 400, 3, 100)
    right = raster(bytes(600 * 800 * 3), 600, 800, 3, 100)
    page_box = (0.0, 0.0, 600.0, 800.0)
    monkeypatch.setattr(
        parse_ocr,
        "internal_page_image_regions",
        lambda capture, minimum_area_ratio, **internal_kwargs: (
            ocr.internal_RasterRegion(left, page_box),
            ocr.internal_RasterRegion(right, page_box),
        ),
    )

    capture = cast(CapturedPage, SimpleNamespace())
    assert ocr.internal_dominant_image_region(capture) is None


def test_safe_image_crop_only_crops_image_dominated_pages() -> None:
    capture = SimpleNamespace(
        page=SimpleNamespace(width=100.0, height=100.0),
        evidence=replace(
            page_evidence(image_count=1, image_area_ratio=0.70),
            image_boxes=((10.0, 20.0, 80.0, 90.0),),
        ),
    )

    assert ocr.internal_safe_image_crop(cast(CapturedPage, capture)) == (10.0, 20.0, 80.0, 90.0)
    sparse = SimpleNamespace(
        page=capture.page,
        evidence=replace(capture.evidence, image_area_ratio=0.20),
    )
    assert ocr.internal_safe_image_crop(cast(CapturedPage, sparse)) is None


def test_direct_image_mapping_accepts_orthogonal_orientation() -> None:
    normal = SimpleNamespace(
        items=(("quad", ((0.0, 0.0), (100.0, 0.0), (0.0, 200.0), (100.0, 200.0))),)
    )
    rotated = SimpleNamespace(
        items=(("quad", ((0.0, 200.0), (0.0, 0.0), (100.0, 200.0), (100.0, 0.0))),)
    )

    assert ocr.internal_direct_image_orientation(normal) == "identity"
    assert ocr.internal_direct_image_orientation(rotated) == "transpose-flip-y"


def test_direct_image_mapping_accepts_bounded_near_axis_orientation() -> None:
    near_axis = SimpleNamespace(
        items=(("quad", ((0.0, 1.0), (100.0, 0.0), (1.0, 201.0), (101.0, 200.0))),)
    )
    skewed = SimpleNamespace(
        items=(("quad", ((0.0, 4.0), (100.0, 0.0), (1.0, 201.0), (101.0, 197.0))),)
    )

    assert ocr.internal_direct_image_orientation(near_axis) is None
    assert (
        ocr.internal_direct_image_orientation(
            near_axis,
            maximum_axis_deviation=0.01,
        )
        == "identity"
    )
    assert (
        ocr.internal_direct_image_orientation(
            skewed,
            maximum_axis_deviation=0.01,
        )
        is None
    )


def test_direct_image_raster_normalizes_orthogonal_orientation() -> None:
    image = SimpleNamespace(items=(("quad", ((0.0, 3.0), (0.0, 0.0), (2.0, 3.0), (2.0, 0.0))),))
    source = raster(bytes((1, 2, 3, 4, 5, 6)), 3, 2, 1, 72)

    oriented = ocr.internal_orient_direct_image_raster(image, source)

    assert (oriented.width, oriented.height) == (2, 3)
    assert oriented.image.array()[:, :, 0].tolist() == [[4, 1], [5, 2], [6, 3]]


def test_tile_tasks_share_one_raster_and_select_rectangles() -> None:
    data = bytes(100 * 120 * 3)
    image = RasterImage(data, 100, 120, 3)
    raster = ocr.internal_Raster(image, 100)
    ocr_pass = OcrPass("primary", OcrPassScope.TILES, 2.0, (3,), tiles=3)

    tasks = ocr.internal_tile_tasks(raster, (0.0, 0.0, 100.0, 120.0), ocr_pass)

    assert len(tasks) == 3
    assert all(task.image is image for task in tasks)
    assert all(task.image.height == raster.height for task in tasks)
    assert tasks[0].rectangle[1] == 0
    assert tasks[-1].rectangle[1] > 0
    assert tasks[-1].rectangle[1] + tasks[-1].rectangle[3] == raster.height


def test_ocr_groups_small_same_raster_tasks_but_splits_large_regions() -> None:
    small = ocr.internal_Raster(RasterImage(bytes(100 * 120), 100, 120, 1), 100)
    ocr_pass = OcrPass("primary", OcrPassScope.TILES, 2.0, (3,), tiles=3)
    small_tasks = ocr.internal_tile_tasks(small, (0.0, 0.0, 100.0, 120.0), ocr_pass)

    large_image = RasterImage(bytes(4000 * 4000), 4000, 4000, 1)
    large = ocr.internal_Raster(large_image, 300)
    large_tasks = ocr.internal_tile_tasks(
        large,
        (0.0, 0.0, 4000.0, 4000.0),
        replace(ocr_pass, tiles=2),
    )

    assert ocr.internal_ocr_task_groups(small_tasks) == (small_tasks,)
    assert ocr.internal_ocr_task_groups(large_tasks) == tuple((task,) for task in large_tasks)


def test_ocr_groups_sixteen_small_regions_per_image_upload() -> None:
    image = RasterImage(bytes(100 * 400), 100, 400, 1)
    tasks = tuple(
        ocr.internal_OcrTask(
            mode=7,
            image=image,
            rectangle=(0, index * 10, 100, 10),
            page_box=(0.0, float(index * 10), 100.0, float((index + 1) * 10)),
            resolution=100,
        )
        for index in range(17)
    )

    groups = ocr.internal_ocr_task_groups(tasks)

    assert tuple(map(len, groups)) == (16, 1)


def test_recognize_group_reuses_api_and_image_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    image = RasterImage(bytes(100 * 120), 100, 120, 1)
    tasks = tuple(
        ocr.internal_OcrTask(
            mode=11,
            image=image,
            rectangle=(0, index * 20, 100, 20),
            page_box=(0.0, float(index * 20), 100.0, float((index + 1) * 20)),
            resolution=100,
        )
        for index in range(3)
    )
    api = object()
    calls: list[tuple[object | None, bool]] = []

    monkeypatch.setattr(parse_ocr, "internal_api", lambda internal_mode: api)

    def recognize(
        task: ocr.internal_OcrTask,
        *,
        api_override: object | None = None,
        image_prepared: bool = False,
    ) -> ocr.internal_OcrTask:
        calls.append((api_override, image_prepared))
        return task

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    assert ocr.internal_recognize_group(tasks) == tasks
    assert calls == [(api, False), (api, True), (api, True)]


def test_compact_ocr_image_removes_redundant_channels() -> None:
    samples = b"".join(bytes((value, value, value, 255)) for value in range(4) for _ in range(4))
    image = RasterImage(samples, 4, 4, 4)

    gray = ocr.internal_compact_ocr_image(image)

    assert gray.channels == 1
    assert gray.pixels == bytes(value for value in range(4) for _ in range(4))


def test_compact_ocr_image_removes_opaque_grayscale_alpha() -> None:
    samples = b"".join(bytes((value, 255)) for value in range(16))
    image = RasterImage(samples, 4, 4, 2)

    gray = ocr.internal_compact_ocr_image(image)

    assert gray.channels == 1
    assert gray.pixels == bytes(range(16))


def test_compact_ocr_image_composites_grayscale_alpha_onto_white() -> None:
    image = RasterImage(bytes((0, 0, 0, 255, 128, 128, 255, 0)), 2, 2, 2)

    gray = ocr.internal_compact_ocr_image(image)

    assert gray.channels == 1
    assert gray.pixels == bytes((255, 0, 191, 255))


def test_compact_ocr_image_drops_only_opaque_alpha() -> None:
    samples = b"".join(bytes((value, 2 * value, 3 * value, 255)) for value in range(16))
    image = RasterImage(samples, 4, 4, 4)

    compact = ocr.internal_compact_ocr_image(image)

    assert compact.channels == 3
    assert compact.pixels == b"".join(bytes((value, 2 * value, 3 * value)) for value in range(16))


def test_compact_ocr_image_keeps_large_rgba_buffer() -> None:
    image = RasterImage(bytes((32, 64, 96, 255)) * 1_000_000, 1_000, 1_000, 4)

    compact = ocr.internal_compact_ocr_image(image)

    assert compact is image


def test_raster_text_signal_rejects_blank_image() -> None:
    signal = ocr.internal_raster_text_signal(RasterImage(bytes([255]) * (200 * 100), 200, 100, 1))

    assert signal.likely_text is False
    assert signal.reason == "low-edge-density"


def test_raster_text_signal_rejects_continuous_tone_image() -> None:
    width = 200
    height = 100
    pixels = b"".join(
        bytes((value, value, value))
        for y in range(height)
        for x in range(width)
        for value in (((x // 8) * 40 + y * 37) % 224,)
    )

    signal = ocr.internal_raster_text_signal(RasterImage(pixels, width, height, 3))

    assert signal.likely_text is False
    assert signal.reason == "continuous-tone-image"


def test_raster_text_signal_preserves_repeated_glyph_edges() -> None:
    width = 200
    height = 100
    pixels = bytearray([255]) * (width * height)
    for line_y in (5, 23, 41, 59, 77):
        for character_x in range(5, 185, 9):
            for y in range(line_y, line_y + 7):
                start = y * width + character_x
                pixels[start : start + 4] = bytes(4)

    signal = ocr.internal_raster_text_signal(RasterImage(pixels, width, height, 1))

    assert signal.likely_text is True
    assert signal.reason == "text-structure"


def test_single_ocr_candidate_skips_merge_rescan() -> None:
    candidate = ocr.internal_candidate(11, candidate_observations("text", 90.0))

    assert ocr.internal_merge_candidate_batches((candidate,)) is candidate


def test_raster_image_normalizes_to_read_only_zero_copy_view() -> None:
    owner = bytearray(range(16))
    image = RasterImage(owner, 4, 4, 1)

    assert isinstance(image.pixels, memoryview)
    assert image.pixels.readonly
    assert image.pixels.obj is owner
    assert not image.array().flags.writeable
    owner[0] = 99
    assert image.array()[0, 0, 0] == 99


def test_text_acceptance_uses_route_confidence_floor() -> None:
    assert ocr.internal_acceptable_text("readable text", 39.0, 20.0)
    assert not ocr.internal_acceptable_text("readable text", 39.0, 40.0)


def test_recognize_maps_rectangle_bounding_boxes_from_full_raster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Iterator:
        def GetUTF8Text(self, internal_level: object) -> str:
            return "recognized"

        def Confidence(self, internal_level: object) -> float:
            return 90.0

        def BoundingBox(self, internal_level: object) -> tuple[int, int, int, int]:
            return (10, 50, 90, 80)

        def Next(self, internal_level: object) -> bool:
            return False

    class Api:
        def __init__(self) -> None:
            self.image: bytes | None = None
            self.rectangle: tuple[int, int, int, int] | None = None

        def SetImageBytes(self, data: bytes, *internal_args: object) -> None:
            self.image = data

        def SetRectangle(self, left: int, top: int, width: int, height: int) -> None:
            self.rectangle = (left, top, width, height)

        def SetSourceResolution(self, internal_resolution: int) -> None:
            pass

        def Recognize(self, **internal_kwargs: object) -> bool:
            return True

        def GetIterator(self) -> Iterator:
            return Iterator()

        def Clear(self) -> None:
            pass

    api = Api()
    monkeypatch.setattr(parse_ocr, "internal_api", lambda internal_mode: api)
    data = bytes(100 * 100)
    task = ocr.internal_OcrTask(
        mode=3,
        image=RasterImage(data, 100, 100, 1),
        rectangle=(0, 40, 100, 60),
        page_box=(0.0, 0.0, 200.0, 200.0),
        resolution=144,
    )

    candidate = ocr.internal_recognize(task)

    assert api.image == data
    assert api.rectangle == (0, 40, 100, 60)
    assert tuple(candidate.observations.bbox[0]) == (20.0, 40.0, 180.0, 100.0)


def test_recognize_clamps_rectangle_before_tesseract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def __init__(self) -> None:
            self.rectangle: tuple[int, int, int, int] | None = None

        def SetImageBytes(self, data: bytes, *internal_args: object) -> None:
            pass

        def SetRectangle(self, left: int, top: int, width: int, height: int) -> None:
            self.rectangle = (left, top, width, height)

        def SetSourceResolution(self, internal_resolution: int) -> None:
            pass

        def Recognize(self, **internal_kwargs: object) -> bool:
            return False

        def Clear(self) -> None:
            pass

    api = Api()
    monkeypatch.setattr(parse_ocr, "internal_api", lambda internal_mode: api)
    task = ocr.internal_OcrTask(
        mode=3,
        image=RasterImage(bytes(100 * 100), 100, 100, 1),
        rectangle=(-20, -10, 100, 50),
        page_box=(0.0, 0.0, 100.0, 100.0),
        resolution=144,
    )

    ocr.internal_recognize(task)

    assert api.rectangle == (0, 0, 80, 40)


def test_recognize_words_uses_word_level_confidence_and_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels: list[object] = []

    class Iterator:
        def GetUTF8Text(self, internal_level: object) -> str:
            levels.append(internal_level)
            return "GPIO12"

        def Confidence(self, internal_level: object) -> float:
            levels.append(internal_level)
            return 92.0

        def BoundingBox(self, internal_level: object) -> tuple[int, int, int, int]:
            levels.append(internal_level)
            return (10, 20, 50, 40)

        def Next(self, internal_level: object) -> bool:
            levels.append(internal_level)
            return False

    class Api:
        def SetImageBytes(self, *internal_args: object) -> None:
            pass

        def SetRectangle(self, *internal_args: object) -> None:
            pass

        def SetSourceResolution(self, internal_resolution: int) -> None:
            pass

        def Recognize(self, **internal_kwargs: object) -> bool:
            return True

        def GetIterator(self) -> Iterator:
            return Iterator()

    monkeypatch.setattr(parse_ocr, "internal_api", lambda internal_mode: Api())
    task = ocr.internal_OcrTask(
        mode=11,
        image=RasterImage(bytes(100 * 100), 100, 100, 1),
        rectangle=(0, 0, 100, 100),
        page_box=(0.0, 0.0, 200.0, 200.0),
        resolution=144,
        minimum_confidence=80.0,
        recognize_words=True,
    )

    candidate = ocr.internal_recognize(task)

    word_level = ocr.internal_TESSEROCR.RIL.WORD
    assert levels == [word_level] * 4
    assert candidate.observations.text == ("GPIO12",)
    assert tuple(candidate.observations.bbox[0]) == (20.0, 120.0, 100.0, 160.0)


def test_recognize_words_collects_symbols_without_another_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Iterator:
        def __init__(self, entries: tuple[tuple[str, float, tuple[int, int, int, int]], ...]):
            self.entries = entries
            self.index = 0

        def GetUTF8Text(self, internal_level: object) -> str:
            return self.entries[self.index][0]

        def Confidence(self, internal_level: object) -> float:
            return self.entries[self.index][1]

        def BoundingBox(self, internal_level: object) -> tuple[int, int, int, int]:
            return self.entries[self.index][2]

        def Next(self, internal_level: object) -> bool:
            self.index += 1
            return self.index < len(self.entries)

    class Api:
        def __init__(self) -> None:
            self.recognitions = 0
            self.iterators = 0

        def SetImageBytes(self, *internal_args: object) -> None:
            pass

        def SetSourceResolution(self, internal_resolution: int) -> None:
            pass

        def Recognize(self, **internal_kwargs: object) -> bool:
            self.recognitions += 1
            return True

        def GetIterator(self) -> Iterator:
            self.iterators += 1
            if self.iterators == 1:
                return Iterator((("AB", 93.0, (10, 20, 50, 40)),))
            return Iterator(
                (
                    ("A", 96.0, (10, 20, 28, 40)),
                    ("B", 94.0, (30, 20, 50, 40)),
                )
            )

    api = Api()
    monkeypatch.setattr(parse_ocr, "internal_api", lambda internal_mode: api)
    task = ocr.internal_OcrTask(
        mode=11,
        image=RasterImage(bytes(100 * 100), 100, 100, 1),
        rectangle=(0, 0, 100, 100),
        page_box=(0.0, 0.0, 200.0, 200.0),
        resolution=144,
        recognize_words=True,
        collect_symbols=True,
    )

    candidate = ocr.internal_recognize(task)

    assert api.recognitions == 1
    assert api.iterators == 2
    assert candidate.observations.text == ("AB",)
    assert candidate.symbols.text == ("A", "B")
    assert candidate.symbols.bbox.tolist() == [
        [20.0, 120.0, 56.0, 160.0],
        [60.0, 120.0, 100.0, 160.0],
    ]


def test_recognize_words_preserves_tesseract_line_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Iterator:
        texts = ("first", "line", "second")
        starts = (True, False, True)

        def __init__(self) -> None:
            self.index = 0

        def IsAtBeginningOf(self, internal_level: object) -> bool:
            return self.starts[self.index]

        def GetUTF8Text(self, internal_level: object) -> str:
            return self.texts[self.index]

        def Confidence(self, internal_level: object) -> float:
            return 95.0

        def BoundingBox(self, internal_level: object) -> tuple[int, int, int, int]:
            return (10, 20, 50, 40)

        def Next(self, internal_level: object) -> bool:
            self.index += 1
            return self.index < len(self.texts)

    class Api:
        def SetImageBytes(self, *internal_args: object) -> None:
            pass

        def SetSourceResolution(self, internal_resolution: int) -> None:
            pass

        def Recognize(self, **internal_kwargs: object) -> bool:
            return True

        def GetIterator(self) -> Iterator:
            return Iterator()

    monkeypatch.setattr(parse_ocr, "internal_api", lambda internal_mode: Api())
    task = ocr.internal_OcrTask(
        mode=11,
        image=RasterImage(bytes(100 * 100), 100, 100, 1),
        rectangle=(0, 0, 100, 100),
        page_box=(0.0, 0.0, 100.0, 100.0),
        resolution=144,
        recognize_words=True,
    )

    candidate = ocr.internal_recognize(task)

    assert candidate.observations.text == ("first", "line", "second")
    assert candidate.observations.line_break_before.tolist() == [True, False, True]


def test_recognize_does_not_reset_tesseract_rectangle_for_full_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Api:
        def SetImageBytes(self, *internal_args: object) -> None:
            pass

        def SetRectangle(self, *internal_args: object) -> None:
            raise AssertionError("full-page OCR already uses the default rectangle")

        def SetSourceResolution(self, internal_resolution: int) -> None:
            pass

        def Recognize(self, **internal_kwargs: object) -> bool:
            return False

        def GetIterator(self) -> None:
            return None

        def Clear(self) -> None:
            pass

    monkeypatch.setattr(parse_ocr, "internal_api", lambda internal_mode: Api())
    task = ocr.internal_OcrTask(
        mode=3,
        image=RasterImage(bytes(100), 10, 10, 1),
        rectangle=(0, 0, 10, 10),
        page_box=(0.0, 0.0, 10.0, 10.0),
        resolution=72,
    )

    assert not len(ocr.internal_recognize(task).observations)


def test_candidate_merge_prefers_complete_overlapping_text() -> None:
    complete = ocr.internal_candidate(3, candidate_observations("column transformation", 90.0))
    clipped = ocr.internal_candidate(3, candidate_observations("transformation", 95.0))

    merged = ocr.internal_merge_candidate_batches((clipped, complete))

    assert merged.observations.text == ("column transformation",)


def test_candidate_diagnostics_are_typed_and_identify_selection() -> None:
    first = ocr.internal_candidate(
        3,
        candidate_observations("alpha + beta", 80.0),
    )
    second = ocr.internal_candidate(
        11,
        candidate_observations("noise", 40.0),
    )
    trace = ocr.internal_RecognitionTrace.create()

    ocr.internal_record_candidates(
        (("primary", first), ("fallback", second)),
        "primary",
        trace,
    )

    diagnostics = trace.report().candidates
    assert diagnostics[0]["selected"] is True
    assert diagnostics[0]["characters"] == len("alpha + beta")
    assert diagnostics[0]["mean_confidence"] == 80.0
    assert diagnostics[1]["selected"] is False


def test_hidden_text_verification_requires_semantic_and_spatial_agreement() -> None:
    tokens = tuple(f"cell{index:02d}" for index in range(30))
    hidden = token_observations(tokens, source=ObservationSource.NATIVE)

    aligned = ocr.internal_hidden_text_verification(hidden, token_observations(tokens))
    displaced = ocr.internal_hidden_text_verification(
        hidden,
        token_observations(tokens, offset_x=500.0),
    )
    unrelated = ocr.internal_hidden_text_verification(
        hidden,
        token_observations(tuple(f"other{index:02d}" for index in range(30))),
    )

    assert aligned.accepted is True
    assert aligned.matched_tokens == 30
    assert aligned.spatially_matched_tokens == 30
    assert displaced.accepted is False
    assert displaced.reason == "low-spatial-overlap"
    assert unrelated.accepted is False
    assert unrelated.reason == "insufficient-matched-tokens"


def test_verified_hidden_text_bypasses_full_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = tuple(f"cell{index:02d}" for index in range(30))
    runs = tuple(
        TextRun(
            token,
            index * 10.0,
            10.0,
            index * 10.0 + 8.0,
            18.0,
            0.0,
            0.0,
            8.0,
            3.0,
            index,
            index,
            0,
            visible=False,
            seqno=index,
        )
        for index, token in enumerate(tokens)
    )
    hidden = parse_capture.internal_observations_from_runs(runs)
    raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(
            raster,
            (0.0, 0.0, 300.0, 100.0),
        ),
    )
    recognized_word_passes = 0

    def recognize(task: ocr.internal_OcrTask) -> ocr.internal_Candidate:
        nonlocal recognized_word_passes
        assert task.recognize_words is True
        recognized_word_passes += 1
        return ocr.internal_candidate(task.mode, token_observations(tokens))

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=300.0, height=100.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            runs=runs,
            observations=hidden,
            evidence=page_evidence(image_count=1, image_area_ratio=1.0),
        ),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(OcrPass("primary", OcrPassScope.PAGE, 3.0, (3,)),),
        verify_hidden_text=True,
    )

    result, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert recognized_word_passes == 1
    assert result.text == tokens
    assert result.visible.tolist() == [True] * len(tokens)
    assert result.source.tolist() == [int(ObservationSource.NATIVE)] * len(tokens)
    assert report.hidden_text_verification["accepted"] is True
    diagnostics = report.passes
    assert [item["name"] for item in diagnostics] == ["hidden-text-verification"]
    assert diagnostics[0]["raster_pixels"] == 100


def test_candidate_utility_rejects_larger_low_confidence_symbol_noise() -> None:
    readable = ocr.internal_candidate(3, candidate_observations("readable text", 92.0))
    noisy = ocr.internal_candidate(6, candidate_observations("// :: -- == ~~ ###", 25.0))

    assert readable.metrics.utility > noisy.metrics.utility


def test_character_filtered_candidate_preserves_recall_when_filtering_is_costly() -> None:
    raw = ocr.internal_candidate(3, candidate_observations("schematic +5V R1", 80.0))
    filtered = ocr.internal_candidate(3, candidate_observations("schematic +5V", 80.0))

    assert ocr.internal_select_character_filtered_candidate(raw, filtered) is raw


def test_character_filtered_candidate_accepts_bounded_utility_gain() -> None:
    raw = ocr.internal_candidate(3, candidate_observations("raw text", 80.0))
    filtered = replace(
        raw,
        metrics=replace(raw.metrics, utility=raw.metrics.utility * 1.05),
    )

    assert ocr.internal_select_character_filtered_candidate(raw, filtered) is filtered


def test_weak_region_tasks_target_ink_without_primary_text() -> None:
    pixels = bytearray([255]) * (40 * 40)
    for y in range(20, 40):
        for x in range(20, 40):
            pixels[y * 40 + x] = 0
    raster = ocr.internal_Raster(RasterImage(pixels, 40, 40, 1), 72)
    primary = ObservationBatch.from_columns(
        ("known",),
        ((2.0, 2.0, 10.0, 10.0),),
        source=ObservationSource.OCR,
        confidence=(90.0,),
    )
    ocr_pass = OcrPass(
        "weak",
        OcrPassScope.WEAK_REGIONS,
        1.0,
        (6,),
        tiles=2,
        region_columns=2,
        max_regions=1,
    )

    tasks = ocr.internal_weak_region_tasks(raster, (0.0, 0.0, 40.0, 40.0), ocr_pass, primary)

    assert len(tasks) == 1
    x, y, width, height = tasks[0].rectangle
    assert x + width / 2 > 20
    assert y + height / 2 > 20


def test_observation_coverage_spreads_line_utility_across_intersected_cells() -> None:
    observations = ObservationBatch.from_columns(
        ("long recognized line",),
        ((0.0, 0.0, 100.0, 10.0),),
        source=ObservationSource.OCR,
        confidence=(95.0,),
    )

    coverage = ocr.internal_observation_coverage_grid(
        observations,
        (0.0, 0.0, 100.0, 10.0),
        1,
        2,
    )

    assert coverage[0] == pytest.approx(coverage[1])
    assert float(coverage.sum()) == pytest.approx(
        ocr.internal_observation_utility("long recognized line", 95.0)
    )


def test_adaptive_rescue_skips_when_primary_covers_visual_ink() -> None:
    width = 60
    height = 60
    pixels = bytearray([255]) * (width * height)
    for y in range(10):
        pixels[y * width : (y + 1) * width] = bytes(width)
    image = RasterImage(pixels, width, height, 1)
    task = ocr.internal_OcrTask(
        mode=3,
        image=image,
        rectangle=(0, 0, width, height),
        page_box=(0.0, 0.0, 60.0, 60.0),
        resolution=72,
    )
    candidate = ocr.internal_candidate(
        3,
        ObservationBatch.from_columns(
            ("recognized" * 40,),
            ((0.0, 50.0, 60.0, 60.0),),
            source=ObservationSource.OCR,
            confidence=(96.0,),
        ),
        median_text_height=20.0,
    )
    ocr_pass = OcrPass("primary", OcrPassScope.PAGE, 1.0, (3,), adaptive_scale=True)

    run, decision = ocr.internal_adaptive_rescue_decision(candidate, (task,), ocr_pass)

    assert run is False
    assert decision["reason"] == "primary-covers-ink"
    assert decision["weak_ink_ratio"] == 0.0


def test_adaptive_rescue_skips_saturated_ink_for_dense_reliable_text() -> None:
    width = 60
    height = 60
    task = ocr.internal_OcrTask(
        mode=3,
        image=RasterImage(bytes(width * height), width, height, 1),
        rectangle=(0, 0, width, height),
        page_box=(0.0, 0.0, 60.0, 60.0),
        resolution=72,
    )
    candidate = ocr.internal_candidate(
        3,
        ObservationBatch.from_columns(
            ("x" * 2_000,),
            ((0.0, 0.0, 60.0, 60.0),),
            source=ObservationSource.OCR,
            confidence=(92.0,),
        ),
        median_text_height=20.0,
    )
    ocr_pass = OcrPass("primary", OcrPassScope.PAGE, 1.0, (3,), adaptive_scale=True)

    run, decision = ocr.internal_adaptive_rescue_decision(candidate, (task,), ocr_pass)

    assert run is False
    assert decision["reason"] == "ink-map-saturated"
    assert decision["mean_ink"] == 1.0


def test_estimated_text_height_uses_repeated_horizontal_bands() -> None:
    width = 120
    height = 100
    pixels = bytearray([255]) * (width * height)
    for band_y in (10, 25, 40, 55, 70):
        for y in range(band_y, band_y + 4):
            pixels[y * width : (y + 1) * width] = bytes(width)
    preview = ocr.internal_Raster(RasterImage(pixels, width, height, 1), 72)

    assert ocr.internal_estimated_text_height(preview) == 4.0


def test_raster_rectangle_maps_top_left_pixels_to_pdf_page_space() -> None:
    raster = ocr.internal_Raster(RasterImage(bytes(100 * 200), 100, 200, 1), 72)

    page_box = ocr.internal_raster_rectangle_page_box(
        raster,
        (10.0, 20.0, 210.0, 420.0),
        (25, 50, 50, 100),
    )

    assert page_box == (60.0, 120.0, 160.0, 320.0)


def test_region_augmentation_keeps_only_confident_uncovered_text() -> None:
    primary = ocr.internal_candidate(3, candidate_observations("known", 90.0))
    supplement = ocr.internal_candidate(
        6,
        ObservationBatch.from_columns(
            ("recovered", "duplicate", "uncertain"),
            (
                (20.0, 0.0, 30.0, 10.0),
                (0.0, 0.0, 1.0, 1.0),
                (40.0, 0.0, 50.0, 10.0),
            ),
            source=ObservationSource.OCR,
            confidence=(92.0, 95.0, 45.0),
        ),
    )

    augmented, added = ocr.internal_augment_candidate(primary, supplement, minimum_confidence=20.0)

    assert added == 1
    assert augmented.observations.text == ("known", "recovered")


def test_dominant_image_ocr_preserves_its_page_space_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    region_box = (12.0, 18.0, 92.0, 78.0)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(raster, region_box),
    )
    monkeypatch.setattr(
        parse_ocr,
        "internal_candidate_ocr_regions",
        lambda internal_capture: (ocr.internal_OcrRegion(region_box, 1.0, ("image",)),),
    )
    observed_boxes: list[tuple[float, float, float, float]] = []

    def recognize(task: ocr.internal_OcrTask, **internal_kwargs: object) -> ocr.internal_Candidate:
        observed_boxes.append(task.page_box)
        return ocr.internal_candidate(task.mode, candidate_observations("mapped", 90.0))

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=100.0, height=100.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            evidence=page_evidence(image_count=1, image_area_ratio=0.70),
        ),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(OcrPass("primary", OcrPassScope.PAGE, 1.0, (3,)),),
    )

    ocr.internal_recognize_page_with_reserved_raster(capture, plan, cast(TaskScope, Context()))

    assert observed_boxes == [region_box]


def test_explicit_fallback_pass_runs_only_for_weak_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(
            raster, (0.0, 0.0, 10.0, 10.0)
        ),
    )
    executed_modes: list[int] = []

    def recognize(task: ocr.internal_OcrTask) -> ocr.internal_Candidate:
        executed_modes.append(task.mode)
        text = "x" if task.mode == 3 else "strong fallback"
        return ocr.internal_candidate(
            task.mode,
            candidate_observations(text, 90.0),
        )

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=10.0, height=10.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=page, evidence=page_evidence()),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass("primary", OcrPassScope.PAGE, 1.0, (3,)),
            OcrPass(
                "fallback",
                OcrPassScope.PAGE,
                1.0,
                (6,),
                run_if_characters_below=5,
            ),
        ),
    )

    context = cast(TaskScope, Context())
    observations, report = internal_recognize_with_report(capture, plan, context)

    assert executed_modes == [3, 6]
    assert observations.text[0] == "strong fallback"
    diagnostics = report.passes
    assert diagnostics[0]["selected"] is False
    assert diagnostics[1]["selected"] is True


def test_large_high_confidence_primary_skips_full_page_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(
            raster, (0.0, 0.0, 10.0, 10.0)
        ),
    )
    executed_modes: list[int] = []

    def recognize(task: ocr.internal_OcrTask) -> ocr.internal_Candidate:
        executed_modes.append(task.mode)
        if task.mode != 3:
            raise AssertionError("full-page fallback should not run")
        return ocr.internal_candidate(
            task.mode,
            candidate_observations("large heading", 96.0),
            median_text_height=40.0,
        )

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=10.0, height=10.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=page, evidence=page_evidence()),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass("primary", OcrPassScope.PAGE, 1.0, (3,)),
            OcrPass(
                "fallback",
                OcrPassScope.PAGE,
                1.0,
                (6,),
                run_if_characters_below=32,
            ),
        ),
    )

    observations, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert executed_modes == [3]
    assert observations.text == ("large heading",)
    assert [item["name"] for item in report.passes] == ["primary"]


def test_named_fallback_page_runs_when_primary_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(
            raster, (0.0, 0.0, 10.0, 10.0)
        ),
    )
    executed_modes: list[int] = []

    def recognize(task: ocr.internal_OcrTask) -> ocr.internal_Candidate:
        executed_modes.append(task.mode)
        text = "" if task.mode == 3 else "recovered page text"
        return ocr.internal_candidate(
            task.mode,
            candidate_observations(text, 92.0),
        )

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=10.0, height=10.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=page, evidence=page_evidence()),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass("primary-page", OcrPassScope.PAGE, 1.0, (3,)),
            OcrPass(
                "fallback-page",
                OcrPassScope.PAGE,
                1.0,
                (6,),
                run_if_characters_below=5,
            ),
        ),
    )

    observations = ocr.internal_recognize_page_with_reserved_raster(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert executed_modes == [3, 6]
    assert observations.text[0] == "recovered page text"


def test_weak_region_pass_augments_instead_of_replacing_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(
            raster, (0.0, 0.0, 10.0, 10.0)
        ),
    )

    def recognize(task: ocr.internal_OcrTask) -> ocr.internal_Candidate:
        if task.mode == 3:
            return ocr.internal_candidate(3, candidate_observations("x", 90.0))
        return ocr.internal_candidate(
            6,
            ObservationBatch.from_columns(
                ("recovered",),
                ((5.0, 5.0, 9.0, 9.0),),
                source=ObservationSource.OCR,
                confidence=(92.0,),
            ),
        )

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    def weak_region_crops(
        internal_capture,
        source_tasks,
        ocr_pass,
        primary,
        **internal_kwargs,
    ):
        task = ocr.internal_OcrTask(
            mode=ocr_pass.modes[0],
            image=raster.image,
            rectangle=(0, 0, 10, 10),
            page_box=(0.0, 0.0, 10.0, 10.0),
            resolution=raster.resolution,
        )
        return (task,), 100, None, ((0.0, 0.0, 10.0, 10.0),)

    monkeypatch.setattr(parse_ocr, "internal_high_resolution_weak_region_tasks", weak_region_crops)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=10.0, height=10.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=page, program=object(), evidence=page_evidence()),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass("primary", OcrPassScope.PAGE, 1.0, (3,)),
            OcrPass(
                "weak-regions",
                OcrPassScope.WEAK_REGIONS,
                1.0,
                (6,),
                tiles=2,
                run_if_characters_below=5,
                region_first=False,
            ),
        ),
    )

    observations, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert observations.text == ("x", "recovered")
    diagnostics = report.passes
    assert diagnostics[1]["accepted_additions"] == 1
    assert diagnostics[1]["region_stage"] == "weak-region-crops"


def test_adaptive_rescue_uses_high_resolution_only_for_undersampled_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    rescue_raster = ocr.internal_Raster(RasterImage(bytes(400), 20, 20, 1), 144)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(
            primary_raster, (0.0, 0.0, 10.0, 10.0)
        ),
    )
    monkeypatch.setattr(
        parse_ocr, "compose_page", lambda *internal_args, **internal_kwargs: object()
    )
    render_budgets: list[int] = []

    def render(*internal_args, **internal_kwargs):
        render_budgets.append(internal_kwargs["max_pixels"])
        return rescue_raster

    monkeypatch.setattr(parse_ocr, "internal_rendered_page_raster", render)
    calls = 0

    def recognize(task: ocr.internal_OcrTask, **internal_kwargs: object) -> ocr.internal_Candidate:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ocr.internal_candidate(
                3,
                candidate_observations("a" * 40, 95.0),
                median_text_height=20.0,
            )
        return ocr.internal_candidate(
            3,
            ObservationBatch.from_columns(
                ("recovered",),
                ((5.0, 5.0, 9.0, 9.0),),
                source=ObservationSource.OCR,
                confidence=(95.0,),
            ),
        )

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=10.0, height=10.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=page, program=object(), evidence=page_evidence()),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass(
                "primary-page",
                OcrPassScope.PAGE,
                1.0,
                (3,),
                adaptive_scale=True,
                region_first=False,
                pixel_budget=ocr.PRIMARY_OCR_PIXELS,
            ),
        ),
    )

    result, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert "recovered" in result.text
    assert render_budgets
    assert set(render_budgets) == {ocr.MAX_OCR_PIXELS}
    diagnostic = report.passes[0]
    rescue_decision = internal_report_mapping(diagnostic["adaptive_rescue_decision"])
    rescue = internal_report_mapping(diagnostic["adaptive_rescue"])
    assert rescue_decision["run"] is True
    assert rescue["scope"] == "weak-regions"
    assert rescue["region_boxes"]


def test_adaptive_rescue_skips_high_resolution_for_large_primary_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_raster = ocr.internal_Raster(RasterImage(bytes([255]) * 100, 10, 10, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(
            primary_raster, (0.0, 0.0, 10.0, 10.0)
        ),
    )

    def unexpected_render(*internal_args: object, **internal_kwargs: object) -> None:
        raise AssertionError("unexpected rescue raster")

    monkeypatch.setattr(
        parse_ocr,
        "internal_rendered_page_raster",
        unexpected_render,
    )
    monkeypatch.setattr(
        parse_ocr,
        "internal_recognize",
        lambda task, **internal_kwargs: ocr.internal_candidate(
            task.mode,
            candidate_observations("heading", 96.0),
            median_text_height=40.0,
        ),
    )

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=10.0, height=10.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=page, program=object(), evidence=page_evidence()),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass(
                "primary-page",
                OcrPassScope.PAGE,
                1.0,
                (3,),
                adaptive_scale=True,
                region_first=False,
                pixel_budget=ocr.PRIMARY_OCR_PIXELS,
            ),
        ),
    )

    result, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert result.text == ("heading",)
    diagnostic = report.passes[0]
    assert diagnostic["adaptive_rescue"] is None
    rescue_decision = internal_report_mapping(diagnostic["adaptive_rescue_decision"])
    assert rescue_decision["reason"] == "primary-text-already-large"


def test_adaptive_rescue_defers_to_scheduled_fallback_below_character_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda internal_capture, **internal_kwargs: ocr.internal_RasterRegion(
            raster, (0.0, 0.0, 10.0, 10.0)
        ),
    )

    def unexpected_render(*internal_args: object, **internal_kwargs: object) -> None:
        raise AssertionError("orientation rescue should defer to the fallback")

    monkeypatch.setattr(parse_ocr, "internal_rendered_page_raster", unexpected_render)
    executed_modes: list[int] = []

    def recognize(task: ocr.internal_OcrTask) -> ocr.internal_Candidate:
        executed_modes.append(task.mode)
        text = "orientation preview" if task.mode == 12 else "complete fallback text"
        return ocr.internal_candidate(
            task.mode,
            candidate_observations(text, 95.0),
            median_text_height=18.0,
        )

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=10.0, height=10.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=page, evidence=page_evidence()),
    )
    plan = WorkPlan(
        PageRoute.HYBRID,
        ocr_passes=(
            OcrPass(
                "orientation-page",
                OcrPassScope.PAGE,
                1.0,
                (12,),
                adaptive_scale=True,
                minimum_characters_for_rescue=300,
                region_first=False,
                pixel_budget=ocr.PRIMARY_OCR_PIXELS,
            ),
            OcrPass(
                "orientation-page-fallback",
                OcrPassScope.PAGE,
                1.0,
                (11,),
                run_if_characters_below=300,
                region_first=False,
            ),
        ),
    )

    result, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert executed_modes == [12, 11]
    assert result.text == ("complete fallback text",)
    diagnostics = report.passes
    assert diagnostics[0]["adaptive_rescue"] is None
    assert diagnostics[1]["selected"] is True


def test_vector_preflight_skips_known_undersampled_primary_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_width = 1_000
    preview_height = 1_000
    preview_pixels = bytearray([255]) * (preview_width * preview_height)
    for band_y in (100, 250, 400, 550, 700):
        for y in range(band_y, band_y + 7):
            preview_pixels[y * preview_width : (y + 1) * preview_width] = bytes(preview_width)
    preview = ocr.internal_Raster(
        RasterImage(preview_pixels, preview_width, preview_height, 1),
        72,
    )
    high_resolution = ocr.internal_Raster(RasterImage(bytes(400), 20, 20, 1), 288)
    render_budgets: list[int] = []
    monkeypatch.setattr(
        parse_ocr, "compose_page", lambda *internal_args, **internal_kwargs: object()
    )
    monkeypatch.setattr(
        parse_ocr,
        "internal_dominant_image_region",
        lambda *internal_args, **internal_kwargs: None,
    )

    def render(*internal_args, **internal_kwargs):
        budget = internal_kwargs["max_pixels"]
        render_budgets.append(budget)
        return preview if budget == ocr.OCR_PREFLIGHT_PIXELS else high_resolution

    monkeypatch.setattr(parse_ocr, "internal_rendered_page_raster", render)
    monkeypatch.setattr(
        parse_ocr,
        "internal_recognize",
        lambda task, **internal_kwargs: ocr.internal_candidate(
            task.mode,
            candidate_observations("schematic labels recovered", 95.0),
        ),
    )

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=10.0, height=10.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            program=object(),
            evidence=replace(page_evidence(), vector_complexity=100_000),
        ),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass(
                "primary-page",
                OcrPassScope.PAGE,
                4.0,
                (3,),
                adaptive_scale=True,
                region_first=False,
                pixel_budget=ocr.PRIMARY_OCR_PIXELS,
            ),
        ),
    )

    _, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert render_budgets == [ocr.OCR_PREFLIGHT_PIXELS, ocr.MAX_OCR_PIXELS]
    diagnostic = report.passes[0]
    adaptive_preflight = internal_report_mapping(diagnostic["adaptive_preflight"])
    assert adaptive_preflight["source"] == "vector-render"


def test_candidate_regions_combine_images_vectors_and_sparse_labels() -> None:
    page = SimpleNamespace(width=1_000.0, height=1_000.0, extraction_cache={})
    native = ObservationBatch.from_columns(
        ("native",),
        ((20.0, 20.0, 180.0, 60.0),),
        source=ObservationSource.NATIVE,
    )
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            observations=native,
            drawings=(
                SimpleNamespace(kind="fill", rect=(700.0, 700.0, 820.0, 820.0)),
                SimpleNamespace(kind="stroke", rect=(400.0, 400.0, 500.0, 500.0)),
            ),
            grid_lines=(),
            evidence=replace(
                page_evidence(),
                image_boxes=((600.0, 100.0, 900.0, 300.0),),
            ),
        ),
    )

    regions = ocr.internal_candidate_ocr_regions(capture)
    reasons = {reason for region in regions for reason in region.reasons}

    assert "image" in reasons
    assert "uncovered-vector" in reasons
    assert "vector-density" in reasons or "sparse-label" in reasons
    assert regions[0].score >= regions[-1].score
    assert all(0.0 <= value <= 1_000.0 for region in regions for value in region.page_box)


def test_candidate_regions_add_fine_vector_label_cells_for_schematics() -> None:
    page = SimpleNamespace(width=1_000.0, height=800.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            observations=ObservationBatch.empty(),
            drawings=tuple(
                SimpleNamespace(kind="stroke", rect=(x, y, x + 12.0, y + 8.0))
                for x, y in ((80.0, 100.0), (420.0, 260.0), (760.0, 520.0))
            ),
            grid_lines=(),
            evidence=replace(page_evidence(), vector_complexity=180),
        ),
    )

    regions = ocr.internal_candidate_ocr_regions(capture)

    assert any("vector-label-neighborhood" in region.reasons for region in regions)


def test_stroked_vector_text_evidence_requires_distributed_compact_paths() -> None:
    drawings = tuple(
        SimpleNamespace(
            kind="stroke",
            path=object(),
            rect=(20.0 + column * 45.0, 20.0 + row * 50.0, 22.0 + column * 45.0, 23.0 + row * 50.0),
            stroke_color=(0.0, 0.0, 0.6),
            stroke_opacity=1.0,
            stroke_pattern=None,
            line_width=0.5,
            line_cap=1,
            line_join=1,
            dash_pattern=None,
            blend_mode=None,
            soft_mask_alpha=None,
        )
        for row in range(15)
        for column in range(20)
    )

    distributed = parse_capture.internal_stroked_vector_text_evidence(
        drawings,
        page_width=1_000.0,
        page_height=800.0,
    )
    clustered = parse_capture.internal_stroked_vector_text_evidence(
        tuple(
            SimpleNamespace(**(vars(drawing) | {"rect": (20.0, 20.0, 22.0, 23.0)}))
            for drawing in drawings
        ),
        page_width=1_000.0,
        page_height=800.0,
    )

    assert distributed.trusted
    assert distributed.candidate_paths == 300
    assert len(distributed.drawing_indexes) == 300
    assert not clustered.trusted


def test_stroked_vector_decoder_corrects_one_character_ocr_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = ocr.StrokedTextDecode(
        observations=(
            ocr.StrokedTextObservation(
                text="D7",
                bbox=(0.1, 0.1, 3.9, 1.9),
                first_drawing=0,
                last_drawing=1,
                confidence=96.0,
            ),
        ),
        eligible_seeds=3,
        aligned_seeds=3,
        accepted_seeds=2,
        initial_signatures=2,
        learned_signatures=3,
        approximate_signatures=0,
    )
    monkeypatch.setattr(parse_ocr, "decode_stroked_text_profile", lambda *args: decoded)
    page = SimpleNamespace(extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            drawings=(object(),),
            evidence=replace(
                page_evidence(),
                stroked_vector_text=StrokedVectorTextEvidence(
                    trusted=True,
                    drawing_indexes=(0,),
                ),
            ),
        ),
    )
    observations = ObservationBatch.from_columns(
        ("07",),
        ((0.0, 0.0, 4.0, 2.0),),
        source=ObservationSource.OCR,
        confidence=(95.0,),
    )

    trace = ocr.internal_RecognitionTrace.create()
    recovered = ocr.internal_recover_stroked_vector_text(capture, observations, trace)

    assert recovered.text == ("D7",)
    assert recovered.source.tolist() == [int(ObservationSource.STRUCTURE)]
    assert trace.report().stroked_vector_decode["corrections"] == 1


def test_stroked_vector_substitution_repairs_only_anchored_low_confidence_edits() -> None:
    assert ocr.internal_stroked_vector_substitution(
        "cn",
        "C11",
        confidence=80.0,
        overlap=1.0,
    )
    assert not ocr.internal_stroked_vector_substitution(
        "cn",
        "C11",
        confidence=95.0,
        overlap=1.0,
    )
    assert not ocr.internal_stroked_vector_substitution(
        "cn",
        "C11",
        confidence=80.0,
        overlap=0.70,
    )
    assert not ocr.internal_stroked_vector_substitution(
        "R1",
        "D7",
        confidence=80.0,
        overlap=1.0,
    )


def test_stroked_vector_pack_uses_independent_horizontal_and_vertical_padding() -> None:
    runs = (
        ocr.StrokedTextRun((0.0, 0.0, 10.0, 2.0), (0, 1), 2),
        ocr.StrokedTextRun((20.0, 0.0, 30.0, 2.0), (2, 3), 2),
    )

    cells, height = ocr.internal_pack_stroked_text_runs(
        runs,
        width=100.0,
        horizontal_padding=4.0,
        vertical_padding=2.0,
    )

    assert tuple(cell.packed_box for cell in cells) == (
        (8.0, 4.0, 18.0, 6.0),
        (26.0, 4.0, 36.0, 6.0),
    )
    assert height == 10.0


def test_packed_stroked_vector_candidate_maps_each_cell_back_to_page() -> None:
    image = RasterImage(bytes(40 * 20), 40, 20, 1)
    packed = ocr.internal_PackedStrokedTextRaster(
        raster=ocr.internal_Raster(image, 432),
        packed_box=(0.0, 0.0, 40.0, 20.0),
        cells=(
            ocr.internal_StrokedTextCell(
                source_box=(100.0, 200.0, 110.0, 205.0),
                packed_box=(10.0, 10.0, 20.0, 15.0),
                drawing_indexes=(7, 8),
            ),
        ),
    )
    observations = ObservationBatch.from_columns(
        ("R7", "outside"),
        ((11.0, 11.0, 19.0, 14.0), (30.0, 1.0, 35.0, 4.0)),
        source=ObservationSource.OCR,
        confidence=(97.0, 99.0),
    )
    symbols = ObservationBatch.from_columns(
        ("R", "7"),
        ((11.0, 11.0, 14.0, 14.0), (15.0, 11.0, 19.0, 14.0)),
        source=ObservationSource.OCR,
        confidence=(98.0, 96.0),
    )

    remapped, unmatched = ocr.internal_remap_stroked_vector_candidate(
        ocr.internal_candidate(11, observations, symbols=symbols),
        packed,
    )

    assert unmatched == 1
    assert remapped.observations.text == ("R7",)
    assert remapped.observations.bbox.tolist() == [[101.0, 201.0, 109.0, 204.0]]
    assert remapped.observations.sequence.tolist() == [7]
    assert remapped.observations.line_break_before.tolist() == [True]
    assert remapped.symbols.text == ("R", "7")
    assert remapped.symbols.bbox.tolist() == [
        [101.0, 201.0, 104.0, 204.0],
        [105.0, 201.0, 109.0, 204.0],
    ]
    assert remapped.symbols.sequence.tolist() == [7, 7]


def test_stroked_vector_symbol_seeds_require_exact_cell_glyph_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = ocr.StrokedTextRun(
        bbox=(100.0, 200.0, 110.0, 205.0),
        drawing_indexes=(7, 8),
        glyph_count=2,
    )
    monkeypatch.setattr(
        parse_ocr,
        "internal_stroked_text_profile",
        lambda capture: ocr.StrokedTextProfile(seed_runs=(run,)),
    )
    monkeypatch.setattr(
        parse_pipeline,
        "internal_stroked_text_profile",
        lambda capture: ocr.StrokedTextProfile(seed_runs=(run,)),
    )
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=SimpleNamespace(extraction_cache={}),
            drawings=(),
            evidence=replace(
                page_evidence(),
                stroked_vector_text=StrokedVectorTextEvidence(
                    trusted=True,
                    drawing_indexes=(7, 8),
                ),
            ),
        ),
    )
    exact = ObservationBatch.from_columns(
        ("R", "7"),
        ((100.0, 200.0, 104.0, 205.0), (105.0, 200.0, 110.0, 205.0)),
        source=ObservationSource.OCR,
        confidence=(97.0, 91.0),
        sequence=(7, 7),
    )
    incomplete = exact.take((0,))

    seeds = ocr.internal_stroked_vector_symbol_seeds(capture, exact)

    assert seeds == (ocr.StrokedTextSeed("R7", run.bbox, 91.0, 7),)
    assert ocr.internal_stroked_vector_symbol_seeds(capture, incomplete) == ()


def test_packed_stroked_vector_decode_gate_requires_reusable_alphabet() -> None:
    weak = ocr.StrokedTextDecode(aligned_seeds=11, learned_signatures=30)
    strong = ocr.StrokedTextDecode(
        observations=tuple(
            ocr.StrokedTextObservation("R1", (0.0, 0.0, 1.0, 1.0), 0, 1) for _ in range(16)
        ),
        aligned_seeds=12,
        learned_signatures=16,
    )

    assert not ocr.internal_packed_stroked_vector_decode_gate(weak, 60)[0]
    assert ocr.internal_packed_stroked_vector_decode_gate(strong, 60)[0]


def test_weak_packed_stroked_vector_seed_uses_full_layer_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = RasterImage(bytes(100), 10, 10, 1)
    raster = ocr.internal_Raster(image, 432)
    packed = ocr.internal_PackedStrokedTextRaster(
        raster=raster,
        packed_box=(0.0, 0.0, 10.0, 10.0),
        cells=(
            ocr.internal_StrokedTextCell(
                source_box=(20.0, 20.0, 25.0, 23.0),
                packed_box=(2.0, 2.0, 7.0, 5.0),
                drawing_indexes=(0, 1),
            ),
        ),
    )
    monkeypatch.setattr(
        parse_ocr, "internal_stroked_vector_text_raster", lambda *args, **kwargs: packed
    )
    monkeypatch.setattr(
        parse_ocr,
        "internal_full_stroked_vector_text_raster",
        lambda *args, **kwargs: ocr.internal_RasterRegion(raster, (0.0, 0.0, 100.0, 100.0)),
    )
    monkeypatch.setattr(
        parse_ocr,
        "internal_remap_stroked_vector_candidate",
        lambda candidate, internal_packed: (candidate, 0),
    )
    monkeypatch.setattr(
        parse_ocr,
        "internal_decode_stroked_vector_text",
        lambda capture, observations, symbols=None: ocr.StrokedTextDecode(),
    )
    recognized: list[bool] = []

    def recognize(
        task: ocr.internal_OcrTask,
        **internal_kwargs: object,
    ) -> ocr.internal_Candidate:
        recognized.append(task.recognize_words)
        text = "packed" if task.recognize_words else "full fallback"
        return ocr.internal_candidate(task.mode, candidate_observations(text, 95.0))

    monkeypatch.setattr(parse_ocr, "internal_recognize", recognize)

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=100.0, height=100.0, extraction_cache={})
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            evidence=replace(
                page_evidence(),
                stroked_vector_text=StrokedVectorTextEvidence(
                    trusted=True,
                    drawing_indexes=(0, 1),
                    bbox=(20.0, 20.0, 25.0, 23.0),
                ),
            ),
        ),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(OcrPass("stroked", OcrPassScope.STROKED_VECTOR_TEXT, 6.0, (11,)),),
    )

    observations, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    assert recognized == [True, False]
    assert observations.text == ("full fallback",)
    diagnostic = report.passes[0]
    assert diagnostic["region_stage"] == "stroked-vector-text-fallback"
    assert diagnostic["task_count"] == 2
    assert diagnostic["raster_pixels"] == 200
    assert report.stroked_vector_packed["fallback_used"] is True


def test_document_stroked_alphabet_uses_richest_page_as_only_ocr_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = (((False, ((0, 0), (16, 16))),),)
    alphabet = ((signature, "A"),)
    pages = tuple(
        SimpleNamespace(
            page_number=index,
            extraction_cache={},
            internal_page_lock=RLock(),
        )
        for index in (1, 2)
    )
    captures = tuple(
        cast(
            CapturedPage,
            SimpleNamespace(
                page=page,
                drawings=(),
                evidence=replace(
                    page_evidence(),
                    stroked_vector_text=StrokedVectorTextEvidence(
                        trusted=True,
                        drawing_indexes=(0,),
                        bbox=(0.0, 0.0, 10.0, 4.0),
                        candidate_paths=candidate_paths,
                    ),
                ),
            ),
        )
        for page, candidate_paths in zip(pages, (50, 100), strict=True)
    )
    ocr_calls: list[int] = []
    plan_calls: list[int] = []

    class Extraction:
        def __init__(self, page: SimpleNamespace, capture: CapturedPage) -> None:
            self.page = page
            self.internal_recognition: RecognitionResult | None = None
            self.internal_recognized_at: float | None = None
            self.internal_capture = capture

        def capture(self) -> CapturedPage:
            return self.internal_capture

        def plan(self) -> WorkPlan:
            plan_calls.append(int(self.page.page_number))
            return WorkPlan(PageRoute.OCR, reason=PagePlanReason.STROKED_VECTOR_TEXT)

        def ocr(self, context: object) -> ObservationBatch:
            ocr_calls.append(int(self.page.page_number))
            observations = candidate_observations("seed", 99.0)
            self.internal_recognition = RecognitionResult(
                observations,
                RecognitionReport(stroked_vector_alphabet=alphabet),
            )
            return observations

    extractions = {
        id(page): Extraction(page, capture) for page, capture in zip(pages, captures, strict=True)
    }
    monkeypatch.setattr(parse_pipeline, "page_extraction", lambda page: extractions[id(page)])
    decoded = ocr.StrokedTextDecode(
        observations=tuple(
            ocr.StrokedTextObservation(
                "A",
                (float(index), 0.0, float(index + 1), 1.0),
                index,
                index,
            )
            for index in range(20)
        ),
        alphabet=alphabet,
        candidate_runs=20,
        decoded_candidate_runs=20,
        candidate_glyphs=40,
        decoded_candidate_glyphs=40,
    )
    monkeypatch.setattr(
        parse_pipeline,
        "decode_stroked_text_profile_with_alphabet",
        lambda profile, learned: decoded,
    )

    seeds, reused = parse_pipeline.internal_prepare_document_stroked_mappings(
        pages,
        captures,
        cast(TaskScope, object()),
    )

    assert (seeds, reused) == (1, 1)
    assert ocr_calls == [2]
    assert plan_calls == [1]
    seed_recognition = extractions[id(pages[1])].internal_recognition
    reused_recognition = extractions[id(pages[0])].internal_recognition
    assert seed_recognition is not None
    assert reused_recognition is not None
    assert seed_recognition.report.document_stroked_glyphs["role"] == "seed"
    assert reused_recognition.report.document_stroked_glyphs["role"] == "reuse"
    assert reused_recognition.report.passes[0]["raster_pixels"] == 0
    assert not parse_pipeline.internal_document_stroked_decode_is_sufficient(
        replace(decoded, decoded_candidate_runs=19)
    )


def test_document_stroked_alphabet_falls_back_to_page_ocr_when_coverage_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = (((False, ((0, 0), (16, 16))),),)
    alphabet = ((signature, "A"),)
    pages = tuple(
        SimpleNamespace(
            page_number=index,
            extraction_cache={},
            internal_page_lock=RLock(),
        )
        for index in (1, 2)
    )
    captures = tuple(
        cast(
            CapturedPage,
            SimpleNamespace(
                page=page,
                drawings=(),
                evidence=replace(
                    page_evidence(),
                    stroked_vector_text=StrokedVectorTextEvidence(
                        trusted=True,
                        drawing_indexes=(0,),
                        bbox=(0.0, 0.0, 10.0, 4.0),
                        candidate_paths=candidate_paths,
                    ),
                ),
            ),
        )
        for page, candidate_paths in zip(pages, (50, 100), strict=True)
    )
    ocr_calls: list[int] = []
    plan_calls: list[int] = []

    class Extraction:
        def __init__(self, page: SimpleNamespace, capture: CapturedPage) -> None:
            self.page = page
            self.internal_recognition: RecognitionResult | None = None
            self.internal_capture = capture

        def capture(self) -> CapturedPage:
            return self.internal_capture

        def plan(self) -> WorkPlan:
            plan_calls.append(int(self.page.page_number))
            return WorkPlan(PageRoute.OCR, reason=PagePlanReason.STROKED_VECTOR_TEXT)

        def ocr(self, context: object) -> ObservationBatch:
            ocr_calls.append(int(self.page.page_number))
            observations = candidate_observations("seed", 99.0)
            self.internal_recognition = RecognitionResult(
                observations,
                RecognitionReport(stroked_vector_alphabet=alphabet),
            )
            return observations

    extractions = {
        id(page): Extraction(page, capture) for page, capture in zip(pages, captures, strict=True)
    }
    monkeypatch.setattr(parse_pipeline, "page_extraction", lambda page: extractions[id(page)])
    monkeypatch.setattr(
        parse_pipeline,
        "decode_stroked_text_profile_with_alphabet",
        lambda profile, learned: ocr.StrokedTextDecode(
            observations=(ocr.StrokedTextObservation("A", (0.0, 0.0, 1.0, 1.0), 0, 0),),
            alphabet=alphabet,
            candidate_runs=20,
            decoded_candidate_runs=1,
            candidate_glyphs=40,
            decoded_candidate_glyphs=1,
        ),
    )

    seeds, reused = parse_pipeline.internal_prepare_document_stroked_mappings(
        pages,
        captures,
        cast(TaskScope, object()),
    )

    assert (seeds, reused) == (2, 0)
    assert ocr_calls == [2, 1]
    assert plan_calls == [1]
    roles: list[object] = []
    for page in pages:
        recognition = extractions[id(page)].internal_recognition
        assert recognition is not None
        roles.append(recognition.report.document_stroked_glyphs["role"])
    assert tuple(roles) == ("seed", "seed")


def test_document_stroked_alphabet_blacklists_conflicting_signatures() -> None:
    first: parse_pipeline.GlyphSignature = (((False, ((0, 0), (16, 16))),),)
    second: parse_pipeline.GlyphSignature = (((False, ((0, 16), (16, 0))),),)
    alphabet: dict[parse_pipeline.GlyphSignature, str] = {first: "A"}
    ambiguous: set[parse_pipeline.GlyphSignature] = set()

    parse_pipeline.internal_merge_document_stroked_alphabet(alphabet, ambiguous, ((second, "B"),))
    parse_pipeline.internal_merge_document_stroked_alphabet(alphabet, ambiguous, ((first, "C"),))
    parse_pipeline.internal_merge_document_stroked_alphabet(alphabet, ambiguous, ((first, "A"),))

    assert alphabet == {second: "B"}
    assert ambiguous == {first}


def test_stroked_text_profile_is_cached_per_capture_drawings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    profile = ocr.StrokedTextProfile()
    monkeypatch.setattr(
        parse_ocr,
        "profile_stroked_text",
        lambda drawings, indexes: calls.append(drawings) or profile,
    )
    page = SimpleNamespace(extraction_cache={})
    evidence = replace(
        page_evidence(),
        stroked_vector_text=StrokedVectorTextEvidence(
            trusted=True,
            drawing_indexes=(0,),
        ),
    )
    first_drawings = (object(),)
    second_drawings = (object(),)
    first = cast(
        CapturedPage,
        SimpleNamespace(page=page, drawings=first_drawings, evidence=evidence),
    )
    second = cast(
        CapturedPage,
        SimpleNamespace(page=page, drawings=second_drawings, evidence=evidence),
    )

    assert ocr.internal_stroked_text_profile(first) is profile
    assert ocr.internal_stroked_text_profile(first) is profile
    assert ocr.internal_stroked_text_profile(second) is profile
    assert calls == [first_drawings, second_drawings]


def test_uncovered_vector_area_ignores_page_sized_background() -> None:
    observations = ObservationBatch.from_columns(
        ("native",),
        ((0.0, 0.0, 1.0, 1.0),),
        source=ObservationSource.NATIVE,
    )
    drawings = (
        SimpleNamespace(kind="fillstroke", rect=(0.0, 0.0, 100.0, 100.0)),
        SimpleNamespace(kind="fill", rect=(20.0, 20.0, 30.0, 30.0)),
        *(SimpleNamespace(kind="stroke", rect=(0.0, 0.0, 1.0, 1.0)) for _ in range(178)),
    )

    assert parse_capture.internal_uncovered_vector_area(
        drawings,
        observations,
        page_area=10_000.0,
    ) == pytest.approx(100.0)


def test_ocr_region_coverage_is_directional_for_contained_images() -> None:
    target = (0.0, 0.0, 100.0, 100.0)
    contained_image = (0.0, 0.0, 100.0, 10.0)

    assert ocr.internal_ocr_region_overlap(target, contained_image) == 1.0
    assert ocr.internal_ocr_region_coverage(target, contained_image) == 0.1
    assert ocr.internal_ocr_region_coverage(contained_image, target) == 1.0


def test_candidate_region_does_not_use_an_image_that_covers_only_a_small_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = ocr.internal_RasterRegion(
        ocr.internal_Raster(RasterImage(bytes(1_000), 100, 10, 1), 72),
        (0.0, 0.0, 100.0, 10.0),
    )
    rendered_raster = ocr.internal_Raster(RasterImage(bytes(10_000), 100, 100, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_page_image_regions",
        lambda *internal_args, **internal_kwargs: (direct,),
    )
    rendered_crops: list[tuple[float, float, float, float]] = []

    def render(*internal_args, **internal_kwargs):
        rendered_crops.append(internal_kwargs["crop"])
        return rendered_raster

    monkeypatch.setattr(parse_ocr, "internal_rendered_page_raster", render)
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=SimpleNamespace(width=100.0, height=100.0)),
    )
    target = ocr.internal_OcrRegion((0.0, 0.0, 100.0, 100.0), 1.0, ("test",))
    ocr_pass = OcrPass("regions", OcrPassScope.PAGE, 1.0, (3,))

    tasks, pixels, _, boxes = ocr.internal_candidate_region_tasks(
        capture,
        (target,),
        ocr_pass,
        rendered=object(),
        compact_image=False,
    )

    assert rendered_crops == [target.page_box]
    assert boxes == (target.page_box,)
    assert tasks[0].page_box == target.page_box
    assert pixels == 10_000


def test_candidate_region_defers_layered_images_to_compositor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_box = (0.0, 0.0, 100.0, 100.0)
    direct_regions = (
        ocr.internal_RasterRegion(
            ocr.internal_Raster(RasterImage(bytes(2_500), 50, 50, 1), 72),
            page_box,
        ),
        ocr.internal_RasterRegion(
            ocr.internal_Raster(RasterImage(bytes(5_000), 100, 50, 1), 72),
            page_box,
        ),
    )
    rendered_raster = ocr.internal_Raster(RasterImage(bytes(10_000), 100, 100, 1), 72)
    monkeypatch.setattr(
        parse_ocr,
        "internal_page_image_regions",
        lambda *internal_args, **internal_kwargs: direct_regions,
    )
    rendered_crops: list[tuple[float, float, float, float]] = []

    def render(*internal_args, **internal_kwargs):
        rendered_crops.append(internal_kwargs["crop"])
        return rendered_raster

    monkeypatch.setattr(parse_ocr, "internal_rendered_page_raster", render)
    capture = cast(
        CapturedPage,
        SimpleNamespace(page=SimpleNamespace(width=100.0, height=100.0)),
    )
    target = ocr.internal_OcrRegion(page_box, 1.0, ("test",))
    ocr_pass = OcrPass("regions", OcrPassScope.PAGE, 1.0, (11,))

    tasks, pixels, _, boxes = ocr.internal_candidate_region_tasks(
        capture,
        (target,),
        ocr_pass,
        rendered=object(),
        compact_image=False,
    )

    assert rendered_crops == [page_box]
    assert boxes == (page_box,)
    assert tasks[0].image is rendered_raster.image
    assert tasks[0].recognize_words is True
    assert pixels == 10_000


def test_distributed_outline_text_requires_many_small_paths_across_both_axes() -> None:
    drawings = tuple(
        SimpleNamespace(
            kind="fill",
            rect=(5.0 + column * 4.5, 5.0 + row * 9.0, 6.0 + column * 4.5, 6.0 + row * 9.0),
        )
        for row in range(10)
        for column in range(20)
    )
    page = SimpleNamespace(width=100.0, height=100.0)

    distributed = cast(CapturedPage, SimpleNamespace(page=page, drawings=drawings))
    too_few = cast(CapturedPage, SimpleNamespace(page=page, drawings=drawings[:-1]))
    clustered = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            drawings=tuple(
                SimpleNamespace(kind="fill", rect=(5.0, 5.0, 6.0, 6.0)) for _ in range(200)
            ),
        ),
    )

    assert ocr.internal_has_distributed_outline_text(distributed)
    assert not ocr.internal_has_distributed_outline_text(too_few)
    assert not ocr.internal_has_distributed_outline_text(clustered)


def test_distributed_outline_text_uses_one_full_page_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = ocr.internal_Raster(RasterImage(bytes(100), 10, 10, 1), 72)
    proposed = ocr.internal_OcrRegion((0.0, 0.0, 100.0, 20.0), 1.0, ("image",))
    requested_batches: list[tuple[ocr.internal_OcrRegion, ...]] = []
    requested_budgets: list[int] = []
    monkeypatch.setattr(
        parse_ocr, "internal_candidate_ocr_regions", lambda internal_capture: (proposed,)
    )
    monkeypatch.setattr(parse_ocr, "internal_has_distributed_outline_text", lambda capture: True)

    def candidate_tasks(
        internal_capture: CapturedPage,
        regions: tuple[ocr.internal_OcrRegion, ...],
        ocr_pass: OcrPass,
        **internal_kwargs: object,
    ):
        requested_batches.append(regions)
        requested_budgets.append(ocr_pass.pixel_budget)
        task = ocr.internal_OcrTask(
            mode=ocr_pass.modes[0],
            image=raster.image,
            rectangle=(0, 0, raster.width, raster.height),
            page_box=regions[0].page_box,
            resolution=raster.resolution,
        )
        return (task,), raster.width * raster.height, None, (regions[0].page_box,)

    monkeypatch.setattr(parse_ocr, "internal_candidate_region_tasks", candidate_tasks)
    monkeypatch.setattr(
        parse_ocr,
        "internal_recognize",
        lambda task, **internal_kwargs: ocr.internal_candidate(
            task.mode,
            candidate_observations("full page outline text", 95.0),
        ),
    )

    class Context:
        def raise_if_cancelled(self) -> None:
            pass

        def map_ordered(self, function, values, **internal_kwargs):
            return map(function, values)

    page = SimpleNamespace(width=100.0, height=100.0, extraction_cache={})
    capture = cast(CapturedPage, SimpleNamespace(page=page, evidence=page_evidence()))
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass(
                "primary",
                OcrPassScope.PAGE,
                1.0,
                (3,),
                region_first=True,
                pixel_budget=ocr.PRIMARY_OCR_PIXELS,
            ),
        ),
    )

    observations, report = internal_recognize_with_report(
        capture,
        plan,
        cast(TaskScope, Context()),
    )

    page_box = (0.0, 0.0, 100.0, 100.0)
    assert observations.text == ("full page outline text",)
    assert requested_batches == [
        (ocr.internal_OcrRegion(page_box, float("inf"), ("distributed-outline-text",)),)
    ]
    assert requested_budgets == [ocr.PRIMARY_OCR_PIXELS]
    diagnostic = report.passes[0]
    assert diagnostic["region_stage"] == "distributed-outline-page"
    assert diagnostic["region_boxes"] == (page_box,)


def test_candidate_regions_cover_low_coverage_hybrid_schematics() -> None:
    page = SimpleNamespace(width=1_000.0, height=800.0, extraction_cache={})
    native = ObservationBatch.from_columns(
        tuple("native" for _ in range(8)),
        tuple((float(index * 10), 20.0, float(index * 10 + 5), 30.0) for index in range(8)),
        source=ObservationSource.NATIVE,
    )
    capture = cast(
        CapturedPage,
        SimpleNamespace(
            page=page,
            observations=native,
            drawings=(SimpleNamespace(kind="stroke", rect=(300.0, 300.0, 500.0, 500.0)),),
            grid_lines=(),
            evidence=replace(page_evidence(), vector_complexity=180),
        ),
    )

    regions = ocr.internal_candidate_ocr_regions(capture)

    assert any("vector-label-neighborhood" in region.reasons for region in regions)


def test_recognition_timeout_scales_with_raster_size() -> None:
    image = RasterImage(bytes(4), 2, 2, 1)
    page_box = (0.0, 0.0, 612.0, 792.0)

    def task_for(width: int, height: int):
        return parse_ocr.internal_OcrTask(
            mode=3,
            image=image,
            rectangle=(0, 0, width, height),
            page_box=page_box,
            resolution=300,
        )

    small = parse_ocr.internal_recognition_timeout(task_for(1_000, 1_000))
    large = parse_ocr.internal_recognition_timeout(task_for(4_000, 4_000))

    assert small == parse_ocr.OCR_TIMEOUT_MILLISECONDS
    assert large > small
    assert large <= parse_ocr.OCR_TIMEOUT_MAX_MILLISECONDS


def test_timeout_recovery_task_reduces_the_raster_and_keeps_page_mapping() -> None:
    width, height = 4_000, 3_000
    task = parse_ocr.internal_OcrTask(
        mode=3,
        image=RasterImage(bytes(width * height), width, height, 1),
        rectangle=(0, 0, width, height),
        page_box=(0.0, 0.0, 600.0, 450.0),
        resolution=400,
    )

    reduced = parse_ocr.internal_timeout_recovery_task(task)

    assert reduced is not None
    pixels = reduced.rectangle[2] * reduced.rectangle[3]
    assert pixels <= parse_ocr.OCR_TIMEOUT_RETRY_PIXELS
    assert reduced.rectangle == (0, 0, reduced.image.width, reduced.image.height)
    # The reduced raster still stands for the same area of the page.
    assert reduced.page_box == pytest.approx(task.page_box)
    assert reduced.resolution < task.resolution


def test_timeout_recovery_task_skips_rasters_already_small_enough() -> None:
    task = parse_ocr.internal_OcrTask(
        mode=3,
        image=RasterImage(bytes(900 * 900), 900, 900, 1),
        rectangle=(0, 0, 900, 900),
        page_box=(0.0, 0.0, 600.0, 600.0),
        resolution=300,
    )

    assert parse_ocr.internal_timeout_recovery_task(task) is None


def test_recover_timed_out_tasks_only_reruns_empty_timeouts() -> None:
    width, height = 4_000, 3_000
    timed_out = parse_ocr.internal_OcrTask(
        mode=3,
        image=RasterImage(bytes(width * height), width, height, 1),
        rectangle=(0, 0, width, height),
        page_box=(0.0, 0.0, 600.0, 450.0),
        resolution=400,
    )
    healthy = replace(timed_out, mode=6)
    batch = ocr.ObservationBatch.from_columns(
        ["recovered"],
        [(0.0, 0.0, 10.0, 10.0)],
        source=ocr.ObservationSource.OCR,
        confidence=[90.0],
        sequence=range(1),
    )
    candidates = (
        ocr.internal_candidate(3, ocr.ObservationBatch.empty(), recognition_status="timeout"),
        ocr.internal_candidate(6, batch, recognition_status="ok"),
    )
    reruns: list[int] = []

    def recognize(tasks):
        reruns.append(len(tasks))
        return tuple(
            ocr.internal_candidate(task.mode, batch, recognition_status="ok") for task in tasks
        )

    recovered = parse_ocr.internal_recover_timed_out_tasks(
        (timed_out, healthy), candidates, recognize
    )

    assert reruns == [1]
    assert recovered[0].recognition_status == "timeout-recovered"
    assert len(recovered[0].observations) == 1
    assert recovered[1] is candidates[1]


def test_direct_scan_allowed_for_a_full_page_image_without_native_text() -> None:
    def capture_for(**evidence):
        defaults = {
            "visible_native_characters": 0,
            "image_count": 1,
            "full_page_image": True,
        }
        return SimpleNamespace(evidence=SimpleNamespace(**(defaults | evidence)))

    plan = WorkPlan(PageRoute.OCR, reason=PagePlanReason.NATIVE_TEXT_UNAVAILABLE)

    assert parse_ocr.internal_direct_scan_allowed(capture_for(), plan)
    # Content the dominant image cannot account for keeps the compositor render.
    assert not parse_ocr.internal_direct_scan_allowed(capture_for(full_page_image=False), plan)
    assert not parse_ocr.internal_direct_scan_allowed(
        capture_for(visible_native_characters=5), plan
    )
    assert not parse_ocr.internal_direct_scan_allowed(
        capture_for(),
        WorkPlan(
            PageRoute.OCR,
            reason=PagePlanReason.NATIVE_TEXT_CORRUPT,
            allow_direct_image_ocr=False,
        ),
    )
    # Pages with native text or no images keep their existing behaviour.
    assert parse_ocr.internal_direct_scan_allowed(capture_for(visible_native_characters=50), plan)
    assert parse_ocr.internal_direct_scan_allowed(capture_for(image_count=0), plan)


def test_decoded_image_enlarges_low_resolution_scans_toward_the_ocr_target(monkeypatch) -> None:
    image = SimpleNamespace(
        image_source=SimpleNamespace(decode=lambda: None),
        raw_data=b"encoded",
        dictionary={"Width": 200, "Height": 100},
    )
    monkeypatch.setattr(
        parse_ocr,
        "decode_pdf_image",
        lambda internal_raw, internal_dictionary: SimpleNamespace(
            data=bytes(200 * 100),
            width=200,
            height=100,
            channels=1,
        ),
    )
    # 200x100 samples over a 200x100 point area is 72 DPI.
    display_area = 200.0 * 100.0

    decoded = parse_ocr.internal_decoded_image_raster(image, display_area)
    unscaled = parse_ocr.internal_decoded_image_raster(image, display_area, upscale=False)

    assert decoded is not None
    assert unscaled is not None
    assert decoded.resolution == pytest.approx(parse_ocr.DIRECT_OCR_TARGET_RESOLUTION, abs=8)
    assert decoded.width > unscaled.width
    assert (unscaled.width, unscaled.height) == (200, 100)
