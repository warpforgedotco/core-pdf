# SPDX-License-Identifier: AGPL-3.0-only
"""ISO 32000-2, Table 31 and 8.3.2.3: the page's default user-space unit."""

import pytest

from core_pdf_spec.s_07_document.page import (
    PAGE_INHERITED_KEYS,
    iter_page_nodes,
    page_user_unit,
)
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.types import PdfName


@pytest.mark.parametrize("value", [None, 1, 1.0, 0.25, 2.5, 75001, 1e20])
def test_positive_units_and_default(value: int | float | None) -> None:
    # 75000 is an Acrobat limit, not a limit imposed by Table 31.
    assert page_user_unit(value) == (1.0 if value is None else value)


@pytest.mark.parametrize(
    "value", [0, -1, True, False, "2", [], {}, float("nan"), float("inf"), -float("inf"), 10**400]
)
def test_invalid_user_unit_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="UserUnit"):
        page_user_unit(value)


def test_user_unit_is_not_inherited() -> None:
    leaf: PdfDict = {"Type": PdfName(b"Page")}
    root: PdfDict = {"Type": PdfName(b"Pages"), "Kids": [leaf], "UserUnit": 3}
    node = next(iter_page_nodes(root, lambda value: value))
    assert "UserUnit" not in PAGE_INHERITED_KEYS
    assert "UserUnit" not in node.inherited_values
    assert page_user_unit(node.dictionary.get("UserUnit")) == 1.0
