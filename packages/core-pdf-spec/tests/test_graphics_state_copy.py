"""q saves a copy of every graphics state parameter (ISO 32000-2 8.4.2)."""

from copy import copy

from core_pdf_spec.s_07_content.model import GraphicsState


def test_copy_carries_every_field_into_a_new_state() -> None:
    state = GraphicsState()
    markers = {name: object() for name in GraphicsState.__slots__}
    for name, marker in markers.items():
        setattr(state, name, marker)
    copied = copy(state)
    assert copied is not state
    assert type(copied) is GraphicsState
    assert GraphicsState.__fields__ == GraphicsState.__slots__
    for name, marker in markers.items():
        assert getattr(copied, name) is marker


def test_changing_the_copy_leaves_the_original() -> None:
    state = GraphicsState()
    copied = copy(state)
    copied.line_width = 7.0
    assert state.line_width != 7.0
