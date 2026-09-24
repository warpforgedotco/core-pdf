import tomllib
from pathlib import Path

import core_pdf.impl

ROOT_CONTRACT = "impl root modules never import an impl subpackage"
REPOSITORY = Path(__file__).resolve().parents[3]


def contract(name):
    config = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
    return next(c for c in config["tool"]["importlinter"]["contracts"] if c["name"] == name)


def test_every_impl_root_module_is_bound_by_the_root_contract():
    impl = Path(core_pdf.impl.__file__).parent
    modules = {
        f"core_pdf.impl.{path.stem}" for path in impl.glob("*.py") if path.stem != "__init__"
    }
    assert modules == set(contract(ROOT_CONTRACT)["source_modules"])


def test_the_root_contract_forbids_every_impl_subpackage():
    impl = Path(core_pdf.impl.__file__).parent
    subpackages = {f"core_pdf.impl.{path.parent.name}" for path in impl.glob("*/__init__.py")}
    assert subpackages == set(contract(ROOT_CONTRACT)["forbidden_modules"])
