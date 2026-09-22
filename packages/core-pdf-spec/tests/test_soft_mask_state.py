# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any, cast

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import PdfPath
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import CachedPdfObject, PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_11_transparency.soft_masks import SoftMask, parse_soft_mask
from core_pdf_spec.types import PdfName, PdfReference


class Sink:
    def __init__(self) -> None:
        self.paints: list[tuple[SoftMask | None, float, float, str | None, bool, bool]] = []
        self.frames: list[tuple[str, ContentStreamFrame, SoftMask | None]] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def paint_path(
        self, state: ContentInterpreter, path: PdfPath, kind: str, fill_rule: str
    ) -> None:
        graphics = state.graphics
        self.paints.append(
            (
                graphics.soft_mask,
                graphics.fill_opacity,
                graphics.stroke_opacity,
                graphics.blend_mode,
                graphics.alpha_is_shape,
                graphics.text_knockout,
            )
        )

    def enter_stream(self, state: ContentInterpreter, frame: ContentStreamFrame) -> None:
        self.frames.append(("enter", frame, state.graphics.soft_mask))

    def exit_stream(self, state: ContentInterpreter, frame: ContentStreamFrame) -> None:
        self.frames.append(("exit", frame, state.graphics.soft_mask))


def make_state() -> tuple[ContentInterpreter, Sink]:
    sink = Sink()
    return ContentInterpreter(ObjectResolver(b"", {}), cast(Any, sink), cast(Any, None)), sink


def make_group(**entries: object) -> PdfStream:
    return PdfStream(
        dictionary={
            "Subtype": PdfName.of("Form"),
            "BBox": [0, 0, 2, 3],
            "Group": {"S": PdfName.of("Transparency")},
            **entries,
        },
        raw_data=b"0 0 1 1 re f",
    )


def make_mask(group: PdfStream | None = None, **entries: object) -> PdfDict:
    return cast(PdfDict, {"S": PdfName.of("Alpha"), "G": group or make_group(), **entries})


def test_alpha_descriptor_preserves_group_identity_without_decoding_or_resource_traversal() -> None:
    state, _ = make_state()
    group = make_group()

    def reject_decode(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("mask group decoded during dictionary parsing")

    group.decoder = reject_decode
    mask = make_mask(group, BC=PdfReference(99, 0))
    extgstate: PdfDict = {"SMask": mask, "ca": 0.4}
    resources: PdfDict = {"ExtGState": {"M": extgstate}}
    group.dictionary["Resources"] = resources
    state.resources = resources
    state.op_gs((PdfName.of("M"),), 0)
    descriptor = state.graphics.soft_mask
    assert descriptor is not None
    assert descriptor.group is group
    assert group.dictionary["Resources"] is resources
    assert extgstate["SMask"] is mask
    assert descriptor.subtype == "Alpha"
    assert descriptor.transfer is descriptor.backdrop_color is descriptor.color_space is None
    assert state.graphics.fill_opacity == 0.4
    with pytest.raises(AttributeError):
        cast(Any, descriptor).ctm = Matrix(2, 0, 0, 2, 0, 0)


@pytest.mark.parametrize("indirect", [False, True])
def test_soft_mask_freezes_gs_CTM_and_omission_differs_from_None(indirect: bool) -> None:
    state, _ = make_state()
    state.graphics.ctm = Matrix(2, 0, 0, 3, 5, 7)
    group = make_group(Matrix=[1, 0, 0, 1, 11, 13])
    resolver = cast(ObjectResolver, state.resolver)
    resolver.objects[key_for(1, 0)] = group
    mask = make_mask(G=PdfReference(1, 0) if indirect else group)
    resolver.objects[key_for(2, 0)] = cast(CachedPdfObject, mask)
    state.apply_extgstate({"SMask": PdfReference(2, 0) if indirect else mask})
    descriptor = state.graphics.soft_mask
    assert descriptor is not None
    assert descriptor.group is group
    assert descriptor.ctm == Matrix(2, 0, 0, 3, 5, 7)
    state.op_cm((1, 0, 0, 1, 17, 19), 0)
    assert descriptor.ctm != state.graphics.ctm
    state.apply_extgstate({})
    state.apply_extgstate({"SMask": None})
    resolver.objects[key_for(3, 0)] = None
    state.apply_extgstate({"SMask": PdfReference(3, 0)})
    assert state.graphics.soft_mask is descriptor
    state.op_q((), 0)
    state.apply_extgstate({"SMask": PdfName.of("None")})
    assert state.graphics.soft_mask is None
    state.op_Q((), 0)
    assert state.graphics.soft_mask is descriptor


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("knockout", [False, True])
def test_prepared_transparency_frame_resets_mask_but_retains_other_inherited_state(
    isolated: bool, knockout: bool
) -> None:
    state, sink = make_state()
    state.apply_extgstate(
        {"SMask": make_mask(), "ca": 0.3, "CA": 0.6, "BM": "Multiply", "AIS": True, "TK": False}
    )
    saved = state.capture_stream_state()
    group = make_group(
        Matrix=[1, 0, 0, 1, 4, 5],
        Group={"S": PdfName.of("Transparency"), "I": isolated, "K": knockout},
    )
    frame = state.append_form_xobject(group, 0)
    assert frame is not None
    state.stream_executor.consume_frame(frame)
    assert sink.paints == [(None, 1.0, 1.0, None, True, False)]
    assert frame.form_bbox == (0, 0, 2, 3)
    assert frame.clip_bbox == (4, 5, 6, 8)
    assert frame.group_isolated is isolated
    assert frame.group_knockout is knockout
    assert sink.frames == [
        ("enter", frame, saved.graphics_state.soft_mask),
        ("exit", frame, saved.graphics_state.soft_mask),
    ]
    assert state.capture_stream_state() == saved


@pytest.mark.parametrize("failure", [False, True])
def test_ordinary_Form_inherits_mask_and_prepared_frame_restores_after_failure(
    failure: bool,
) -> None:
    state, sink = make_state()
    state.apply_extgstate({"SMask": make_mask()})
    state.op_BT((), 0)
    saved = state.capture_stream_state()
    form = PdfStream(
        dictionary={"Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1]},
        raw_data=b"0 0 1 1 re f BT " + (b"unknown" if failure else b"ET"),
    )
    frame = state.append_form_xobject(form, 0)
    assert frame is not None
    if failure:
        with pytest.raises(PdfParseError, match="unknown content operator"):
            state.stream_executor.consume_frame(frame)
    else:
        state.stream_executor.consume_frame(frame)
    assert sink.paints[0][0] is saved.graphics_state.soft_mask
    assert state.capture_stream_state() == saved
    assert frame.old_state is None
    assert not state.stream_executor.active_streams


def test_consume_frame_rejects_an_already_entered_frame_without_unwinding_it() -> None:
    state, _ = make_state()
    frame = ContentStreamFrame(PdfStream(), {}, IDENTITY_MATRIX, 0, None)
    state.stream_executor.enter(frame)
    try:
        with pytest.raises(PdfParseError, match="already entered"):
            state.stream_executor.consume_frame(frame)
        assert frame.old_state is not None
    finally:
        state.stream_executor.exit(frame)


def test_prepared_frame_decode_failure_leaves_parent_mask_and_scope_intact() -> None:
    state, sink = make_state()
    state.apply_extgstate({"SMask": make_mask()})
    state.op_BT((), 0)
    saved = state.capture_stream_state()
    group = make_group()

    def reject_decode(*args: Any, **kwargs: Any) -> bytes:
        raise PdfParseError("failed group decode")

    group.decoder = reject_decode
    frame = state.append_form_xobject(group, 0)
    assert frame is not None
    with pytest.raises(PdfParseError, match="failed group decode"):
        state.stream_executor.consume_frame(frame)
    assert state.capture_stream_state() == saved
    assert frame.old_state is None
    assert not sink.frames
    assert not state.stream_executor.active_streams


@pytest.mark.parametrize("transfer", [None, PdfName.of("Identity")])
def test_identity_transfer_defaults(transfer: object) -> None:
    state, _ = make_state()
    mask = parse_soft_mask(make_mask(TR=transfer), state.resolver, ctm=IDENTITY_MATRIX)
    assert mask is not None
    assert mask.transfer is None


def test_resolved_transfer_clips_its_single_output_and_defines_nonzero_outside_alpha() -> None:
    state, _ = make_state()
    resolver = cast(ObjectResolver, state.resolver)
    resolver.objects[key_for(1, 0)] = [PdfReference(2, 0), 1]
    resolver.objects[key_for(2, 0)] = 0
    transfer = {"FunctionType": 2, "Domain": PdfReference(1, 0), "C0": [1.5], "C1": [-0.5], "N": 1}
    mask = parse_soft_mask(make_mask(TR=transfer), resolver, ctm=IDENTITY_MATRIX)
    assert mask is not None
    assert mask.transfer is not None
    assert mask.transfer(0) == (1,)
    assert mask.transfer(0.5) == (0.5,)
    assert mask.transfer(1) == (0,)


@pytest.mark.parametrize("sampled", [False, True])
def test_sampled_and_stitching_scalar_transfer_functions(sampled: bool) -> None:
    state, _ = make_state()
    function = (
        PdfStream(
            dictionary={
                "FunctionType": 0,
                "Domain": [0, 1],
                "Range": [0, 1],
                "Size": [2],
                "BitsPerSample": 8,
            },
            raw_data=b"\x00\xff",
        )
        if sampled
        else {
            "FunctionType": 3,
            "Domain": [0, 1],
            "Functions": [{"FunctionType": 2, "Domain": [0, 1], "N": 1}],
            "Bounds": [],
            "Encode": [0, 1],
        }
    )
    mask = parse_soft_mask(make_mask(TR=function), state.resolver, ctm=IDENTITY_MATRIX)
    assert mask is not None
    assert mask.transfer is not None
    assert mask.transfer(0.25) == (0.25,)


@pytest.mark.parametrize(
    "function",
    [
        PdfName.of("Other"),
        [],
        {"FunctionType": 2, "Domain": [0, 1, 0, 1], "N": 1},
        {"FunctionType": 2, "Domain": [0, 1], "Range": [0, 1, 0, 1], "N": 1},
        {"FunctionType": 2, "Domain": [0, 1], "C0": [0, 0], "C1": [1, 1], "N": 1},
        {"FunctionType": 4, "Domain": [0, 1]},
    ],
)
def test_transfer_requires_a_single_input_and_output(function: object) -> None:
    state, _ = make_state()
    with pytest.raises(ValueError):
        parse_soft_mask(make_mask(TR=function), state.resolver, ctm=IDENTITY_MATRIX)


def test_transfer_rejects_mismatched_unselected_stitching_child_and_cycles() -> None:
    state, _ = make_state()
    scalar = {"FunctionType": 2, "Domain": [0, 1], "N": 1}
    vector = {**scalar, "C0": [0, 0], "C1": [1, 1]}
    stitching = {
        "FunctionType": 3,
        "Domain": [0, 1],
        "Bounds": [0.8],
        "Encode": [0, 1, 0, 1],
        "Functions": [scalar, vector],
    }
    with pytest.raises(ValueError, match="one output"):
        parse_soft_mask(make_mask(TR=stitching), state.resolver, ctm=IDENTITY_MATRIX)
    stitching["Functions"] = [stitching]
    with pytest.raises(ValueError, match="cyclic"):
        parse_soft_mask(make_mask(TR=stitching), state.resolver, ctm=IDENTITY_MATRIX)


def test_transfer_compiler_can_be_injected_without_changing_group_identity() -> None:
    state, _ = make_state()
    function = PdfStream(
        dictionary={"FunctionType": 4, "Domain": [0, 1], "Range": [0, 1]},
        raw_data=b"{ 1 exch sub }",
    )
    inputs: list[object] = []

    def compile_function(value: object) -> Any:
        inputs.append(value)
        return lambda value: (1 - value,)

    group = make_group()
    mask = parse_soft_mask(
        make_mask(group, TR=function),
        state.resolver,
        ctm=IDENTITY_MATRIX,
        compile_function=compile_function,
    )
    assert mask is not None
    assert mask.group is group
    assert mask.transfer is not None
    assert len(inputs) == 1
    assert isinstance(inputs[0], PdfStream)
    assert inputs[0].raw_data is function.raw_data
    assert mask.transfer(0.25) == (0.75,)


@pytest.mark.parametrize(
    ("space", "expected"),
    [("DeviceGray", (0,)), ("DeviceRGB", (0, 0, 0)), ("DeviceCMYK", (0, 0, 0, 1))],
)
def test_luminosity_requires_a_group_space_and_defaults_to_its_initial_color(
    space: str, expected: tuple[float, ...]
) -> None:
    state, _ = make_state()
    group = make_group(Group={"S": PdfName.of("Transparency"), "CS": PdfName.of(space)})
    mask = parse_soft_mask(
        make_mask(group, S=PdfName.of("Luminosity")), state.resolver, ctm=IDENTITY_MATRIX
    )
    assert mask is not None
    assert mask.subtype == "Luminosity"
    assert mask.backdrop_color == expected
    assert mask.color_space is not None
    assert mask.color_space.kind == space


def test_luminosity_resolves_only_selected_indirect_space_and_backdrop_values() -> None:
    state, _ = make_state()
    resolver = cast(ObjectResolver, state.resolver)
    resolver.objects.update(
        {
            key_for(1, 0): {"S": PdfName.of("Transparency"), "CS": PdfReference(2, 0)},
            key_for(2, 0): [PdfName.of("CalGray"), PdfReference(3, 0)],
            key_for(3, 0): {"WhitePoint": PdfReference(4, 0), "Unused": PdfReference(99, 0)},
            key_for(4, 0): [PdfReference(5, 0), 1, 1],
            key_for(5, 0): 1,
            key_for(6, 0): [PdfReference(7, 0)],
            key_for(7, 0): 0.4,
        }
    )
    mask = parse_soft_mask(
        make_mask(
            make_group(Group=PdfReference(1, 0)),
            S=PdfName.of("Luminosity"),
            BC=PdfReference(6, 0),
        ),
        resolver,
        ctm=IDENTITY_MATRIX,
    )
    assert mask is not None
    assert mask.backdrop_color == (0.4,)
    assert mask.color_space is not None
    assert mask.color_space.kind == "CalGray"
    assert key_for(99, 0) not in resolver.objects


@pytest.mark.parametrize("backdrop", [[0.1, 0.2], [0.1, 0.2, True], [0.1, 0.2, float("nan")]])
def test_luminosity_rejects_invalid_backdrop_components(backdrop: object) -> None:
    state, _ = make_state()
    group = make_group(Group={"S": PdfName.of("Transparency"), "CS": PdfName.of("DeviceRGB")})
    with pytest.raises(ValueError):
        parse_soft_mask(
            make_mask(group, S=PdfName.of("Luminosity"), BC=backdrop),
            state.resolver,
            ctm=IDENTITY_MATRIX,
        )


@pytest.mark.parametrize(
    "space", [None, PdfName.of("Pattern"), [PdfName.of("Lab"), {"WhitePoint": [1, 1, 1]}]]
)
def test_luminosity_rejects_absent_or_invalid_blending_spaces(space: object) -> None:
    state, _ = make_state()
    group = make_group(Group={"S": PdfName.of("Transparency"), "CS": space})
    with pytest.raises(ValueError, match="blending color space"):
        parse_soft_mask(
            make_mask(group, S=PdfName.of("Luminosity")), state.resolver, ctm=IDENTITY_MATRIX
        )


@pytest.mark.parametrize(
    "value",
    [
        False,
        1,
        PdfName.of("Unknown"),
        {},
        make_mask(Type=PdfName.of("XObject")),
        make_mask(S=PdfName.of("Unknown")),
        make_mask(G=PdfStream()),
        make_mask(make_group(Type=PdfName.of("Page"))),
        make_mask(make_group(Group={})),
        make_mask(make_group(BBox=[0, 0, 1])),
        make_mask(make_group(Matrix=[1, 0, 0, 1, True, 0])),
        make_mask(make_group(Group={"S": PdfName.of("Transparency"), "I": 1})),
    ],
)
def test_invalid_soft_masks_reject_without_replacing_existing_state(value: object) -> None:
    state, _ = make_state()
    state.apply_extgstate({"SMask": make_mask()})
    previous = state.graphics.soft_mask
    with pytest.raises(ValueError):
        state.apply_extgstate({"ca": 0.2, "SMask": value})
    assert state.graphics.soft_mask is previous
    assert state.graphics.fill_opacity == 0.2


def test_cyclic_group_reference_rejects_without_recursing() -> None:
    state, _ = make_state()
    resolver = cast(ObjectResolver, state.resolver)
    resolver.objects[key_for(1, 0)] = PdfReference(2, 0)
    resolver.objects[key_for(2, 0)] = PdfReference(1, 0)
    resolver.objects[key_for(3, 0)] = cast(CachedPdfObject, make_mask(G=PdfReference(1, 0)))
    with pytest.raises(ValueError, match="cyclic"):
        state.apply_extgstate({"SMask": PdfReference(3, 0)})
