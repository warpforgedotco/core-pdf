from core_pdf.impl.render_display import PATH_PAINT_LAYOUT
from core_pdf.impl.render_model import PathPaintItem, PathPaintKind


def test_slot_built_paint_item_matches_the_constructor() -> None:
    values = (
        PathPaintKind.FILL,
        7,
        (0.0, 1.0, 2.0, 3.0),
        None,
        (0.5,),
        0.25,
        None,
        None,
        1.0,
        0,
        0,
        None,
        "nonzero",
        None,
        None,
        False,
        None,
        None,
        True,
        None,
        None,
    )
    built = PATH_PAINT_LAYOUT.build(values)
    constructed = PathPaintItem(*values)  # type: ignore[call-arg]
    for name in PathPaintItem.__fields__:
        assert getattr(built, name) == getattr(constructed, name), name
