import pytest

from tests.src.core_pdf.render_dict_snapshot import (
    FIXTURES,
    RENDER_DICT_FIXTURES,
    recorded,
    render_dict_digests,
)

RECORDED = recorded()


@pytest.mark.parametrize("fixture", RENDER_DICT_FIXTURES)
def test_rendered_page_dict_matches_snapshot(fixture: str) -> None:
    path = FIXTURES / fixture
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("fixture corpus not initialized")
    assert render_dict_digests(fixture) == RECORDED[fixture]
