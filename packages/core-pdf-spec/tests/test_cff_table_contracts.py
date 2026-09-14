"""CFF table readers keep their strict parsing extension contract on malformed input."""

import pytest

from core_pdf_spec.s_09_fonts.font_program import CFFFont


def table_font(data=b"", *, cid=True, count=3):
    font = CFFFont.__new__(CFFFont)
    font.data = data
    font.charstrings = [b"\x0e"] * count
    font.top_dict = {}
    font.is_cid_keyed = cid
    return font


@pytest.mark.parametrize(
    ("reader", "operator"), [("read_fd_select", (12, 37)), ("read_font_dicts", (12, 36))]
)
@pytest.mark.parametrize("values", [None, [], [0.5], [-1], [float("inf")], [float("nan")], [0, 1]])
def test_cid_table_offsets_require_one_finite_nonnegative_integer(reader, operator, values):
    font = table_font(b"\0" * 10)
    if values is not None:
        font.top_dict[operator] = values
    with pytest.raises(ValueError):
        getattr(font, reader)()


@pytest.mark.parametrize("data", [b"\0\0\1\1", b"\3\0\2\0\0\0\0\1\1\0\3"])
def test_fdselect_explicit_and_range_forms_select_same_font_dicts(data):
    font = table_font(b"prefix" + data)
    font.top_dict = {(12, 37): [6]}
    assert font.read_fd_select() == (0, 1, 1)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\0\0",
        b"\2\0\0\0",
        b"\3",
        b"\3\0\0",
        b"\3\0\1\0",
        b"\3\0\1\0\1\0\0\3",
        b"\3\0\1\0\0\0\0\2",
        b"\3\0\2\0\0\0\0\0\1\0\3",
    ],
)
def test_fdselect_rejects_truncation_invalid_format_bounds_and_nonincreasing_ranges(data):
    font = table_font(b"prefix" + data)
    font.top_dict = {(12, 37): [6]}
    with pytest.raises(ValueError):
        font.read_fd_select()


def test_fdarray_parses_each_dictionary_and_noncid_fonts_need_no_fd_tables():
    # INDEX with two one-byte dictionaries: operator 0 and operator 1.
    font = table_font(b"prefix\0\2\1\1\2\3\0\1")
    font.top_dict = {(12, 36): [6]}
    assert font.read_font_dicts() == ({0: []}, {1: []})
    font.is_cid_keyed = False
    assert font.read_font_dicts() == ()
    assert font.read_fd_select() == (0, 0, 0)


@pytest.mark.parametrize(
    "private",
    [
        [],
        [1],
        [1, 2, 3],
        [0.5, 0],
        [float("inf"), 0],
        [0, float("inf")],
        [-1, 0],
        [0, -1],
        [100, 0],
    ],
)
def test_private_dictionary_bounds_reject_malformed_values(private):
    with pytest.raises(ValueError):
        table_font().read_private_subrs({18: private})


@pytest.mark.parametrize(
    "encoded", [b"\x13", b"\x8b\x8c\x13", b"\x8a\x13", b"\x1e\x0a\x5f\x13", b"\x1e\x1b\x99\x9f\x13"]
)
def test_subroutine_offset_requires_one_finite_nonnegative_integer(encoded):
    with pytest.raises(ValueError):
        table_font(encoded).read_private_subrs({18: [len(encoded), 0]})


def test_private_subroutine_offset_is_relative_to_private_dictionary():
    # Private DICT says Subrs is at relative offset 2. The INDEX has one return program.
    font = table_font(b"prefix\x8d\x13\0\1\1\1\2\x0b")
    assert font.read_private_subrs({18: [2, 6]}) == [b"\x0b"]
    assert font.read_private_subrs({}) == []
    assert font.read_private_subrs({18: [0, 6]}) == []


@pytest.mark.parametrize("data", [None, b"", b"\2\0\4\1", b"\1\0\5\1", b"\1\0\4\1\0\0\0\0\0\0\0\0"])
def test_cff_construction_rejects_missing_program_and_missing_top_dictionary(data):
    with pytest.raises(ValueError):
        CFFFont(data)


@pytest.mark.parametrize("values", [None, [], [0, 1]])
def test_required_top_dictionary_offset_has_no_implicit_default(values):
    font = table_font()
    if values is not None:
        font.top_dict[17] = values
    with pytest.raises(ValueError, match="dictionary offset"):
        font.dict_offset(17)
    if values is None:
        assert font.dict_offset(17, default=0) == 0


def test_unterminated_escaped_dict_operator_is_rejected():
    with pytest.raises(ValueError, match="dict operator"):
        table_font().parse_dict(b"\x0c")


@pytest.mark.parametrize("glyph_id", [-1, 3])
def test_font_projection_rejects_out_of_range_glyph_id(glyph_id):
    font = table_font()
    font.fd_select = (0, 0, 0)
    font.local_subrs = ((b"\x0b",),)
    with pytest.raises(ValueError, match="glyph id"):
        font.local_subrs_for_glyph(glyph_id)
    with pytest.raises(ValueError, match="glyph id"):
        font.font_matrix(glyph_id)


@pytest.mark.parametrize("fd", [-1, 1])
def test_font_projection_rejects_missing_font_dictionary(fd):
    font = table_font()
    font.fd_select = (fd, fd, fd)
    font.local_subrs = ((b"\x0b",),)
    font.font_dicts = ({},)
    with pytest.raises(ValueError, match="dictionary index"):
        font.local_subrs_for_glyph(0)
    with pytest.raises(ValueError, match="dictionary index"):
        font.font_matrix(0)
