# SPDX-License-Identifier: AGPL-3.0-only


import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.streams import ContentStreamExecutor, ContentStreamFrame, StreamKey
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.types import PdfName, PdfReference, Rectangle


def make_state() -> ContentInterpreter:
    return ContentInterpreter(ObjectResolver(b"", {}), None, None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("indirect_array", [False, True])
@pytest.mark.parametrize("indirect_components", [False, True])
@pytest.mark.parametrize("transparency", [False, True])
def test_frame_keeps_resolved_local_bbox_and_original_operand(
    indirect_array: bool, indirect_components: bool, transparency: bool
) -> None:
    state = make_state()
    resolver = state.resolver
    values = [5, 7, 1, 2]
    array = []
    for index, value in enumerate(values, 10):
        resolver.objects[key_for(index, 0)] = value  # ty: ignore[unresolved-attribute]
        array.append(PdfReference(index, 0) if indirect_components else value)
    original_array = array.copy()
    resolver.objects[key_for(20, 0)] = array  # ty: ignore[unresolved-attribute]
    operand = PdfReference(20, 0) if indirect_array else array
    dictionary: PdfDict = {"Subtype": PdfName.of("Form"), "BBox": operand}
    if transparency:
        dictionary["Group"] = {"S": PdfName.of("Transparency")}
    frame = state.append_form_xobject(PdfStream(dictionary=dictionary), 0)
    assert frame is not None
    assert frame.form_bbox == (1.0, 2.0, 5.0, 7.0)
    assert frame.form_bbox_operand is operand
    assert frame.clip_bbox == (1.0, 2.0, 5.0, 7.0)
    assert array == original_array


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ([1, 2, 5, 7], (1, 2, 5, 7)),
        ([5, 2, 1, 7], (1, 2, 5, 7)),
        ([1, 7, 5, 2], (1, 2, 5, 7)),
        ([5, 7, 1, 2], (1, 2, 5, 7)),
        ([3, 2, 3, 7], (3, 2, 3, 7)),
        ([1, 4, 5, 4], (1, 4, 5, 4)),
        ([3, 4, 3, 4], (3, 4, 3, 4)),
    ],
)
def test_frame_normalizes_corner_order_without_discarding_empty_bounds(
    bounds: list[int], expected: Rectangle
) -> None:
    frame = make_state().append_form_xobject(PdfStream(dictionary={"BBox": bounds}), 0)
    assert frame is not None
    assert frame.form_bbox == expected
    assert frame.clip_bbox == expected
    assert frame.form_bbox_operand is bounds


@pytest.mark.parametrize(
    ("form_matrix", "parent_matrix", "expected_ctm", "expected_envelope"),
    [
        (None, IDENTITY_MATRIX, IDENTITY_MATRIX, (1, 2, 5, 7)),
        (
            [2, 0, 0, 3, 10, 20],
            IDENTITY_MATRIX,
            Matrix(2, 0, 0, 3, 10, 20),
            (12, 26, 20, 41),
        ),
        (
            [0, 1, -1, 0, 10, 20],
            IDENTITY_MATRIX,
            Matrix(0, 1, -1, 0, 10, 20),
            (3, 21, 8, 25),
        ),
        (
            [1, 0.5, 2, 1, 10, 20],
            IDENTITY_MATRIX,
            Matrix(1, 0.5, 2, 1, 10, 20),
            (15, 22.5, 29, 29.5),
        ),
        (
            [1, 0, 0, 1, 5, 6],
            Matrix(2, 0, 0, 3, 7, 11),
            Matrix(2, 0, 0, 3, 17, 29),
            (19, 35, 27, 50),
        ),
        (
            [1, 0, 0, 0, 3, 4],
            IDENTITY_MATRIX,
            Matrix(1, 0, 0, 0, 3, 4),
            (4, 4, 8, 4),
        ),
        (
            [-1, 0, 0, -2, 0, 0],
            IDENTITY_MATRIX,
            Matrix(-1, 0, 0, -2, 0, 0),
            (-5, -14, -1, -4),
        ),
    ],
)
def test_frame_retains_local_bounds_separately_from_transformed_envelope(
    form_matrix: list[float] | None,
    parent_matrix: Matrix,
    expected_ctm: Matrix,
    expected_envelope: Rectangle,
) -> None:
    state = make_state()
    state.graphics.ctm = parent_matrix
    dictionary: PdfDict = {"BBox": [1, 2, 5, 7]}
    if form_matrix is not None:
        dictionary["Matrix"] = form_matrix
    frame = state.append_form_xobject(PdfStream(dictionary=dictionary), 0)
    assert frame is not None
    assert frame.form_bbox == (1, 2, 5, 7)
    assert frame.ctm == expected_ctm
    assert frame.clip_bbox == expected_envelope


@pytest.mark.parametrize("dictionary", [{}, {"BBox": None}])
def test_missing_form_bounds_retain_strict_required_bbox_failure(dictionary: PdfDict) -> None:
    with pytest.raises(PdfParseError, match="requires a BBox"):
        make_state().append_form_xobject(PdfStream(dictionary=dictionary), 0)


@pytest.mark.parametrize(
    "bounds",
    [
        [],
        [0, 1, 2],
        [0, 1, 2, 3, 4],
        "0 0 1 1",
        [True, 0, 1, 1],
        [0, 0, "1", 1],
        [0, 0, float("nan"), 1],
        [0, 0, float("inf"), 1],
    ],
)
def test_invalid_form_bounds_retain_strict_box_failure(bounds: object) -> None:
    with pytest.raises(ValueError, match="invalid box value"):
        make_state().append_form_xobject(PdfStream(dictionary={"BBox": bounds}), 0)


def test_local_bbox_transport_preserves_reader_bbox_hook_and_queue_signature() -> None:
    class CustomInterpreter(ContentInterpreter):
        def resolve_form_bbox(self, value: object) -> Rectangle | None:
            assert value == "reader operand"
            return (5, 7, 1, 2)

    class ExistingExecutor(ContentStreamExecutor):
        def queue(
            self,
            stream: PdfStream,
            resources: PdfDict,
            ctm: Matrix,
            depth: int,
            *,
            clip_bbox: Rectangle | None = None,
            form_bbox_operand: object = None,
            group_alpha: float | None = None,
            stream_key: StreamKey | None = None,
        ) -> ContentStreamFrame | None:
            assert form_bbox_operand == "reader operand"
            return super().queue(
                stream,
                resources,
                ctm,
                depth,
                clip_bbox=clip_bbox,
                form_bbox_operand=form_bbox_operand,
                group_alpha=group_alpha,
                stream_key=stream_key,
            )

    state = CustomInterpreter(ObjectResolver(b"", {}), None, None)  # ty: ignore[invalid-argument-type]
    state.stream_executor = ExistingExecutor(state)
    frame = state.append_form_xobject(PdfStream(dictionary={"BBox": "reader operand"}), 0)
    assert frame is not None
    assert frame.form_bbox == (1, 2, 5, 7)
    assert frame.form_bbox_operand == "reader operand"
    assert frame.clip_bbox == (1, 2, 5, 7)


def test_reader_missing_bbox_hook_keeps_none_transport() -> None:
    class CustomInterpreter(ContentInterpreter):
        def resolve_form_bbox(self, value: object) -> Rectangle | None:
            return None

    state = CustomInterpreter(ObjectResolver(b"", {}), None, None)  # ty: ignore[invalid-argument-type]
    frame = state.append_form_xobject(PdfStream(dictionary={}), 0)
    assert frame is not None
    assert frame.form_bbox is frame.clip_bbox is frame.form_bbox_operand is None


def test_existing_frame_and_queue_callers_default_to_no_local_bbox() -> None:
    state = make_state()
    stream = PdfStream()
    frame = ContentStreamFrame(stream, {}, IDENTITY_MATRIX, 1, (1, 2, 5, 7), 0.5)
    assert frame.form_bbox is None
    queued = state.stream_executor.queue(stream, {}, IDENTITY_MATRIX, 1, clip_bbox=(1, 2, 5, 7))
    assert queued is not None
    assert queued.form_bbox is None
