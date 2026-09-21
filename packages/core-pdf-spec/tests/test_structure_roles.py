# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable, Mapping

import pytest

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_14_structure.roles import (
    MATHML_NAMESPACE,
    PDF_1_7_NAMESPACE,
    PDF_2_0_NAMESPACE,
    StructureType,
    is_standard_structure_type,
    resolve_structure_role,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfReference, PdfString


def namespace(name: str, mapping: object = None) -> dict[str, object]:
    return {"Type": PdfName.of("Namespace"), "NS": PdfString(name.encode()), "RoleMapNS": mapping}


def resolver(objects: Mapping[int, object]) -> Callable[[object], object]:
    return lambda value: (
        objects.get(value.object_number) if isinstance(value, PdfReference) else value
    )


def test_undefined_namespace_maps_transitively_then_defaults_to_pdf17() -> None:
    result = resolve_structure_role(
        PdfName.of("Foo"),
        role_map={"Foo": PdfName.of("Bar"), "Bar": PdfName.of("P")},
        context=SemanticContext(PdfVersion(2, 0)),
    )
    assert (result.name, result.namespace, result.status) == ("P", PDF_1_7_NAMESPACE, "standard")
    assert result.path == tuple(StructureType(name, None) for name in ("Foo", "Bar", "P"))


@pytest.mark.parametrize("value", [None, PdfReference(1), PdfReference(99)])
def test_optional_namespace_resolving_to_null_is_absent(value: object) -> None:
    result = resolve_structure_role(
        PdfName.of("Custom"),
        namespace=value,
        role_map={"Custom": PdfName.of("P")},
        resolve=resolver({1: None}),
    )
    assert result.path == (StructureType("Custom", None), StructureType("P", None))
    assert (result.name, result.namespace) == ("P", PDF_1_7_NAMESPACE)


@pytest.mark.parametrize("objects", [{2: None}, {}])
def test_required_role_target_namespace_resolving_to_null_is_rejected(
    objects: dict[int, object],
) -> None:
    with pytest.raises(ValueError, match="namespace dictionary"):
        resolve_structure_role(
            PdfName.of("Custom"),
            namespace=namespace("urn:custom", {"Custom": [PdfName.of("P"), PdfReference(2)]}),
            resolve=resolver(objects),
        )


def test_required_namespace_name_resolving_to_null_is_rejected() -> None:
    with pytest.raises(ValueError, match="namespace NS"):
        resolve_structure_role(
            PdfName.of("Custom"), namespace={"NS": PdfReference(1)}, resolve=resolver({1: None})
        )


@pytest.mark.parametrize(("minor", "expected"), [(3, "P"), (4, "P"), (5, "H1"), (7, "H1")])
def test_standard_types_are_remapped_from_pdf15(minor: int, expected: str) -> None:
    result = resolve_structure_role(
        PdfName.of("P"),
        role_map={"P": PdfName.of("Custom"), "Custom": PdfName.of("H1")},
        context=SemanticContext(PdfVersion(1, minor)),
    )
    assert result.name == expected


def test_no_context_retains_modern_mapping_and_stops_at_recognized_target() -> None:
    result = resolve_structure_role(PdfName.of("P"), role_map={"P": PdfName.of("H1"), "H1": 7})
    assert (result.name, result.status) == ("H1", "standard")


def test_recognized_return_to_start_stops_before_cycle_detection() -> None:
    result = resolve_structure_role(
        PdfName.of("P"), role_map={"P": PdfName.of("Custom"), "Custom": PdfName.of("P")}
    )
    assert (result.name, result.status) == ("P", "standard")
    assert len(result.path) == 3


def test_indirect_namespace_and_role_map_dictionaries_are_resolved_on_demand() -> None:
    objects = {
        1: namespace("urn:a", PdfReference(2)),
        2: {"Custom": PdfName.of("P")},
    }
    result = resolve_structure_role(
        PdfName.of("Custom"), namespace=PdfReference(1), resolve=resolver(objects)
    )
    assert (result.name, result.namespace) == ("P", PDF_1_7_NAMESPACE)


def test_cross_namespace_chains_keep_same_named_types_distinct() -> None:
    objects = {
        1: namespace("urn:one", {"Custom": [PdfName.of("Custom"), PdfReference(2)]}),
        2: namespace("urn:two", {"Custom": [PdfName.of("Title"), PdfReference(3)]}),
        3: namespace(PDF_2_0_NAMESPACE),
    }
    result = resolve_structure_role(
        PdfName.of("Custom"), namespace=PdfReference(1), resolve=resolver(objects)
    )
    assert (result.name, result.namespace, result.status) == (
        "Title",
        PDF_2_0_NAMESPACE,
        "standard",
    )
    assert [item.namespace for item in result.path] == ["urn:one", "urn:two", PDF_2_0_NAMESPACE]


def test_explicit_default_namespace_does_not_use_the_global_role_map() -> None:
    mapping = {"P": PdfName.of("H1")}
    implicit = resolve_structure_role(PdfName.of("P"), role_map=mapping)
    explicit = resolve_structure_role(
        PdfName.of("P"), namespace=namespace(PDF_1_7_NAMESPACE), role_map=mapping
    )
    assert implicit.name == "H1"
    assert explicit.name == "P"
    assert implicit.namespace == explicit.namespace == PDF_1_7_NAMESPACE
    assert explicit.path[0].namespace == PDF_1_7_NAMESPACE
    assert implicit.path[0].namespace is None


def test_single_name_target_enters_default_namespace_without_root_remapping() -> None:
    result = resolve_structure_role(
        PdfName.of("Custom"),
        namespace=namespace("urn:one", {"Custom": PdfName.of("Unknown")}),
        role_map={"Unknown": PdfName.of("P")},
    )
    assert (result.name, result.namespace, result.status) == (
        "Unknown",
        PDF_1_7_NAMESPACE,
        "unmapped",
    )


@pytest.mark.parametrize("name", ["Note", "Title", "H7", "P", "H01", "H0", "H١"])
def test_namespace_identity_selects_the_standard_vocabulary(name: str) -> None:
    assert is_standard_structure_type(name, PDF_1_7_NAMESPACE) == (name in {"Note", "P"})
    assert is_standard_structure_type(name, PDF_2_0_NAMESPACE) == (name in {"Title", "H7", "P"})
    assert not is_standard_structure_type(name, "urn:custom")


def test_pdf20_header_does_not_assign_pdf20_namespace_to_a_new_type() -> None:
    result = resolve_structure_role(PdfName.of("Title"), context=SemanticContext(PdfVersion(2, 0)))
    assert (result.namespace, result.status) == (PDF_1_7_NAMESPACE, "unmapped")


def test_direct_mathml_preserves_domain_identity_without_claiming_vocabulary_validation() -> None:
    result = resolve_structure_role(PdfName.of("math"), namespace=namespace(MATHML_NAMESPACE))
    assert (result.name, result.namespace, result.status) == ("math", MATHML_NAMESPACE, "domain")


def test_root_role_map_cycle_is_a_finite_result_not_a_parse_error() -> None:
    result = resolve_structure_role(
        PdfName.of("A"), role_map={"A": PdfName.of("B"), "B": PdfName.of("A")}
    )
    assert (result.name, result.status) == ("A", "cycle")
    assert [item.name for item in result.path] == ["A", "B", "A"]


def test_cross_namespace_cycle_preserves_its_path() -> None:
    objects = {
        1: namespace("urn:one", {"A": [PdfName.of("B"), PdfReference(2)]}),
        2: namespace("urn:two", {"B": [PdfName.of("A"), PdfReference(1)]}),
    }
    result = resolve_structure_role(
        PdfName.of("A"), namespace=PdfReference(1), resolve=resolver(objects)
    )
    assert result.status == "cycle"
    assert result.path[0] == result.path[-1]


@pytest.mark.parametrize(
    "target", [7, PdfString(b"P"), [], [PdfName.of("P")], [PdfName.of("P"), {}]]
)
def test_invalid_namespace_targets_fail_strictly(target: object) -> None:
    with pytest.raises(ValueError, match="target"):
        resolve_structure_role(
            PdfName.of("Custom"), namespace=namespace("urn:a", {"Custom": target})
        )


@pytest.mark.parametrize(
    "value", [7, {}, {"NS": PdfName.of("urn:a")}, {"NS": PdfString(b"urn:a"), "Type": 1}]
)
def test_invalid_namespace_dictionaries_fail_strictly(value: object) -> None:
    with pytest.raises(ValueError, match="namespace"):
        resolve_structure_role(PdfName.of("P"), namespace=value)


@pytest.mark.parametrize("value", [PdfString(b"Custom"), "Custom", 42, None])
def test_structure_type_must_be_a_name_without_a_reader_extension(value: object) -> None:
    with pytest.raises(ValueError, match="PDF name"):
        resolve_structure_role(value)


def test_root_map_cannot_contain_cross_namespace_arrays() -> None:
    with pytest.raises(ValueError, match="target"):
        resolve_structure_role(
            PdfName.of("Custom"), role_map={"Custom": [PdfName.of("P"), PdfReference(1)]}
        )


def test_namespace_identity_is_the_name_not_reference_number() -> None:
    objects = {2: namespace("urn:a")}
    with pytest.raises(ValueError, match="another namespace"):
        resolve_structure_role(
            PdfName.of("A"),
            namespace=namespace("urn:a", {"A": [PdfName.of("B"), PdfReference(2)]}),
            resolve=resolver(objects),
        )


def test_unknown_role_keeps_exact_namespace_string() -> None:
    result = resolve_structure_role(PdfName.of("P"), namespace=namespace(PDF_1_7_NAMESPACE + "/"))
    assert result.namespace == PDF_1_7_NAMESPACE + "/"
    assert result.status == "unmapped"


@pytest.mark.parametrize("version", [None, PdfVersion(9, 0)])
def test_unknown_context_fails_without_guessing_the_mapping_rules(
    version: PdfVersion | None,
) -> None:
    with pytest.raises(PdfUnsupportedError):
        resolve_structure_role(PdfName.of("P"), context=SemanticContext(version))


def test_long_role_map_chain_is_iterative() -> None:
    mapping = {f"Custom{n}": PdfName.of(f"Custom{n + 1}") for n in range(1200)}
    mapping["Custom1200"] = PdfName.of("P")
    result = resolve_structure_role(PdfName.of("Custom0"), role_map=mapping)
    assert result.name == "P"
    assert len(result.path) == 1202
