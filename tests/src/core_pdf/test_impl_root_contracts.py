"""impl is flat, so its import contracts list modules; these keep the lists complete.

A feature's modules share its prefix. Shared code lives in the unprefixed
modules, which the root contract binds; every feature module is forbidden to
them. A contract that names one module of a feature names all of them, except
the lists that deliberately name part of one: two contracts forbid only
part of document composition, and serialization binds output_serialize alone.
"""

import tomllib
from pathlib import Path

import core_pdf.impl

ROOT_CONTRACT = "Shared impl modules never import a feature module"
FEATURES = (
    "capture",
    "document",
    "recovery",
    "extract",
    "fonts",
    "graphics",
    "layout",
    "output",
    "render",
)
PARTIAL_LISTS = {
    ("Capture stays below document composition and output processing", "forbidden_modules"),
    ("Object recovery stays below content capture and document composition", "forbidden_modules"),
    ("Serialization stays below processing and the specification", "source_modules"),
}
REPOSITORY = Path(__file__).resolve().parents[3]


def contracts():
    config = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
    return config["tool"]["importlinter"]["contracts"]


def impl_modules():
    impl = Path(core_pdf.impl.__file__).parent
    return {f"core_pdf.impl.{path.stem}" for path in impl.glob("*.py") if path.stem != "__init__"}


def feature_of(module):
    stem = module.removeprefix("core_pdf.impl.")
    return next((feature for feature in FEATURES if stem.startswith(f"{feature}_")), None)


def feature_modules():
    groups = {feature: set() for feature in FEATURES}
    for module in impl_modules():
        if (feature := feature_of(module)) is not None:
            groups[feature].add(module)
    return groups


def test_impl_has_no_subpackages():
    impl = Path(core_pdf.impl.__file__).parent
    assert [path.parent.name for path in impl.glob("*/__init__.py")] == ["data"]


def test_the_root_contract_binds_every_shared_module_and_forbids_every_feature_module():
    root = next(c for c in contracts() if c["name"] == ROOT_CONTRACT)
    shared = {module for module in impl_modules() if feature_of(module) is None}
    assert set(root["source_modules"]) == shared
    assert set(root["forbidden_modules"]) == set().union(*feature_modules().values())


def test_contracts_name_whole_features():
    groups = feature_modules()
    for contract in contracts():
        for field in ("source_modules", "forbidden_modules"):
            if (contract["name"], field) in PARTIAL_LISTS:
                continue
            listed = set(contract.get(field, ()))
            for feature, modules in groups.items():
                if listed & modules:
                    assert modules <= listed, (contract["name"], field, feature)
