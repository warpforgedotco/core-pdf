"""Guard the compat package's re-export table against drifting from its facades.

`core_pdf.api.compat` re-exports a curated set of facade symbols so callers can
write `from core_pdf.api.compat import LTChar`. The table is written by hand to
keep the lazy `__getattr__` loading in `compat/__init__.py` intact, which means
nothing stops a facade from gaining a public name the table never learns about.
That is exactly what happened to `LTFigure` and `LTTextContainer`: both were in
`compat.pdfminer.__all__` and reachable as `compat.pdfminer.LTFigure`, but not
as `compat.LTFigure`.
"""

from __future__ import annotations

import pytest

from core_pdf.api import compat


def test_every_pdfminer_public_name_is_reexported() -> None:
    """pdfminer is re-exported wholesale, so its `__all__` is the contract."""
    from core_pdf.api.compat import pdfminer

    exported = {
        name
        for name, (module_name, _) in compat.internal_EXPORTS.items()
        if module_name == "pdfminer"
    }
    assert set(pdfminer.__all__) - exported == set()


@pytest.mark.parametrize("name", sorted(compat.internal_EXPORTS))
def test_exported_name_resolves(name: str) -> None:
    """Every table entry names a symbol its facade actually defines."""
    module_name, attribute = compat.internal_EXPORTS[name]
    module = __import__(f"core_pdf.api.compat.{module_name}", fromlist=[attribute])
    assert hasattr(module, attribute)
    assert getattr(compat, name) is getattr(module, attribute)


def test_all_covers_the_table_and_the_facade_modules() -> None:
    assert set(compat.__all__) == set(compat.internal_EXPORTS) | set(compat.internal_MODULES)
