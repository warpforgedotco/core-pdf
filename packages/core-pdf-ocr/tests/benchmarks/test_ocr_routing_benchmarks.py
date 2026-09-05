from collections.abc import Iterator
from typing import Any

import pytest

from core_pdf_ocr.impl.extract.capture import capture_page
from core_pdf_ocr.impl.extract.observations import plan_page
from tests.helpers.benchmark_pages import DENSE_PDF, opened_page


@pytest.fixture(scope="module")
def dense() -> Iterator[Any]:
    with opened_page(DENSE_PDF) as page:
        yield capture_page(page)


def test_plan_page_benchmark(benchmark, dense: Any) -> None:
    """The routing decision. Cheap, but it gates every page."""
    plan = benchmark(plan_page, dense)

    assert not plan.ocr_passes
