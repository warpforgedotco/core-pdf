import pytest

from core_pdf.impl.render_blend import BlendOp, blend_op, declared_blend, kernel_blend_code


@pytest.mark.parametrize(
    ("name", "lowered", "folded"),
    [
        (None, None, None),
        ("Normal", BlendOp.NORMAL, BlendOp.NORMAL),
        ("nOrMaL", BlendOp.NORMAL, BlendOp.NORMAL),
        ("Multiply", BlendOp.MULTIPLY, BlendOp.MULTIPLY),
        ("SCREEN", BlendOp.SCREEN, BlendOp.SCREEN),
        ("ColorDodge", BlendOp.COLOR_DODGE, BlendOp.COLOR_DODGE),
        ("colorburn", BlendOp.COLOR_BURN, BlendOp.COLOR_BURN),
        ("Difference", BlendOp.UNSUPPORTED, BlendOp.UNSUPPORTED),
        ("ſcreen", BlendOp.UNSUPPORTED, BlendOp.SCREEN),
        ("Multiplyß", BlendOp.UNSUPPORTED, BlendOp.UNSUPPORTED),
    ],
)
def test_lowered_and_casefolded_names_keep_their_own_matches(name, lowered, folded):
    assert blend_op(name) is lowered
    assert blend_op(name, casefold=True) is folded


def test_only_the_exact_default_name_is_dropped():
    assert declared_blend(None) is None
    assert declared_blend("Normal") is None
    assert declared_blend("normal") == "normal"
    assert declared_blend("Multiply") == "Multiply"


def test_kernel_codes_treat_unsupported_modes_as_normal():
    assert [kernel_blend_code(op) for op in (None, *BlendOp)] == [0, 0, 1, 2, 3, 4, 0]
