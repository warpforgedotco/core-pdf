# SPDX-License-Identifier: AGPL-3.0-only
"""Resource type validation must not eagerly resolve unused entries."""

import pytest

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.resources import resolve_resource_dict
from core_pdf.impl.types import PdfReference
from tests.helpers.resolvers import IdentityResolver


def test_resource_resolution_preserves_selected_and_unused_references() -> None:
    entries = {"Selected": PdfReference(7, 0), "Unused": PdfReference(8, 0)}
    assert resolve_resource_dict(entries, IdentityResolver()) is entries
    assert resolve_resource_dict(None, IdentityResolver()) is None


@pytest.mark.parametrize("value", [False, 0, "Font", [], PdfReference(7, 0)])
def test_resource_resolution_rejects_non_dictionary_values(value: object) -> None:
    with pytest.raises(PdfParseError, match="resource value must be a dictionary"):
        resolve_resource_dict(value, IdentityResolver())


def test_resource_type_policy_is_only_called_for_invalid_resources() -> None:
    calls: list[object] = []

    def ignore(value: object) -> None:
        calls.append(value)

    resolver = IdentityResolver()
    assert resolve_resource_dict({}, resolver, on_invalid=ignore) == {}
    assert resolve_resource_dict(None, resolver, on_invalid=ignore) is None
    assert calls == []
    assert resolve_resource_dict(42, resolver, on_invalid=ignore) is None
    assert calls == [42]
