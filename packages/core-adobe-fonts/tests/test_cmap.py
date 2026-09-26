# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest

from core_adobe_fonts.cmap.decoder import CMapDecoder
from core_adobe_fonts.cmap.resources import resolve_cmap_decoder, resolve_cmap_resource
from core_adobe_fonts.cmap.tokenizer import (
    CMapBlock,
    CMapProgram,
    CMapToken,
    cmap_tokens,
    decode_cmap_hex_token,
    iter_cmap_tokens,
)

CODESPACE = b"1 begincodespacerange <00> <ff> endcodespacerange\n"


@pytest.mark.parametrize(
    "mapping",
    [
        b"2 begincidchar <01> 7 <02> bad endcidchar",
        b"2 begincidchar <01> 7 <02> 65536 endcidchar",
        b"2 begincidchar <01> 7 <02> endcidchar",
        b"1 begincidrange <01> <03> 65535 endcidrange",
        b"1 begincidrange <03> <01> 7 endcidrange",
        b"1 begincidchar {<01>} 7 endcidchar",
        b"1 begincidchar <0100> 7 endcidchar",
        b"1 begincidchar <01> 7",
    ],
)
def test_cid_cmap_rejects_malformed_mappings(mapping: bytes) -> None:
    with pytest.raises(ValueError):
        CMapDecoder(CODESPACE + mapping)


@pytest.mark.parametrize(
    "suffix",
    [b"<unterminated", b"(unterminated", b"[<00>", b"{dup", b"begincmap"],
)
def test_cmap_rejects_incomplete_tokens_and_scope(suffix: bytes) -> None:
    with pytest.raises(ValueError, match="unterminated"):
        CMapProgram.parse(CODESPACE + suffix)


@pytest.mark.parametrize(
    ("include_arrays", "include_words", "block_values", "direct_values"),
    [
        (False, False, [b"<01>", b"(A)"], [b"<01>", b"(A)", b"<02>", b"(B)"]),
        (
            False,
            True,
            [b"<01>", b"(A)", b"word", b"<<", b">>"],
            [b"<01>", b"(A)", b"[", b"<02>", b"(B)", b"]", b"word", b"<<", b">>"],
        ),
        (
            True,
            False,
            [b"<01>", b"(A)", b"[<02> (B)]"],
            [b"<01>", b"(A)", b"[<02> (B)]"],
        ),
        (
            True,
            True,
            [b"<01>", b"(A)", b"[<02> (B)]", b"word", b"<<", b">>"],
            [b"<01>", b"(A)", b"[<02> (B)]", b"word", b"<<", b">>"],
        ),
    ],
)
def test_cmap_token_selection_preserves_array_grouping(
    include_arrays: bool,
    include_words: bool,
    block_values: list[bytes],
    direct_values: list[bytes],
) -> None:
    data = b"<01> (A) [<02> (B)] word << >> {<03>} % <04> ignored\n"
    block = CMapBlock(data, tuple(iter_cmap_tokens(data, group_arrays=True)))
    assert (
        block.token_values(include_arrays=include_arrays, include_words=include_words)
        == block_values
    )
    assert (
        cmap_tokens(data, include_arrays=include_arrays, include_words=include_words)
        == direct_values
    )
    with pytest.raises(ValueError, match="unterminated"):
        cmap_tokens(data + b"<broken", include_arrays=include_arrays, include_words=include_words)


def test_cmap_preserves_spec_defined_identity_and_invalid_code_consumption() -> None:
    cmap = CMapDecoder(b"/Identity-H usecmap")
    assert cmap.decode_entries(b"\x00A\x00") == [(b"\x00A", 65), (b"\x00", 0)]
    assert decode_cmap_hex_token(b"<4>") == b"@"


def test_adobe_resources_load_with_parent_inheritance() -> None:
    cmap = resolve_cmap_decoder("UniJIS-UTF16-V")
    assert cmap is not None
    assert cmap.wmode == 1
    assert cmap.decode_entries(b"\x00A")[0][1] > 0


def test_resource_loader_preserves_cycle_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    from core_adobe_fonts.cmap import resources as cmap_resources

    requests: list[str] = []

    def resource(name: str) -> bytes:
        requests.append(name)
        return b"/Loop usecmap"

    monkeypatch.setattr(cmap_resources, "resolve_cmap_resource", resource)
    with pytest.raises(ValueError, match="cyclic"):
        cmap_resources.resolve_cmap_decoder("Loop")
    assert requests == ["Loop", "Loop"]


def test_resource_loader_keeps_inheritance_and_local_writing_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core_adobe_fonts.cmap import resources as cmap_resources

    resources = {
        "Grandparent": CODESPACE + b"/WMode 1 def 1 begincidchar <01> 7 endcidchar",
        "Parent": b"/Grandparent usecmap 1 begincidchar <02> 8 endcidchar",
        "Child": b"/Parent usecmap /WMode 0 def 1 begincidchar <01> 9 endcidchar",
    }
    monkeypatch.setattr(cmap_resources, "resolve_cmap_resource", resources.get)
    cmap = cmap_resources.resolve_cmap_decoder("Child")
    assert cmap is not None
    assert cmap.wmode == 0
    assert cmap.decode_entries(b"\x01\x02") == [(b"\x01", 9), (b"\x02", 8)]


def test_cmap_resource_names_are_already_decoded() -> None:
    assert resolve_cmap_decoder("Identity-H") is not None
    assert resolve_cmap_decoder("/Identity-H") is None
    assert resolve_cmap_resource("UniJIS-UTF16-H") is not None
    assert resolve_cmap_resource("/UniJIS-UTF16-H") is None


class LenientProgram(CMapProgram):
    __slots__ = ()

    def block_count(self, begin_index: int) -> int:  # noqa: ARG002
        return 0

    def validate_block(
        self, begin_keyword: bytes, block_tokens: list[CMapToken], declared_count: int
    ) -> None:
        pass

    def reject_unterminated_block(self) -> None:
        pass


class LenientDecoder(CMapDecoder):
    __slots__ = ()

    max_inheritance_depth = 1

    @staticmethod
    def parse_program(data: bytes) -> CMapProgram:
        return LenientProgram.parse(data)

    def reject_mapping(self, reason: str, cause: BaseException | None = None) -> None:
        pass


def test_mapping_hooks_skip_what_the_strict_reader_rejects() -> None:
    mapping = (
        b"9 begincidchar <01> 7 <02> bad 03 4 <04> 65536 <05> 8 <06> endcidchar\n"
        b"3 begincidrange <10> <12> 65535 <20> <22> 1 <30> endcidrange"
    )
    with pytest.raises(ValueError):
        CMapDecoder(CODESPACE + mapping)
    cmap = LenientDecoder(CODESPACE + mapping)
    assert cmap.cid_mappings == {b"\x01": 7, b"\x05": 8}
    assert [(item.start, item.end, item.first_cid) for item in cmap.cid_ranges] == [
        (b"\x20", b"\x22", 1)
    ]


def test_program_hooks_relax_counts_and_terminators() -> None:
    data = CODESPACE + b"begincidchar <01> 7 endcidchar 1 begincidchar <02> 8"
    with pytest.raises(ValueError, match="block count"):
        list(CMapProgram.parse(data).blocks(b"begincidchar", b"endcidchar"))
    blocks = list(LenientProgram.parse(data).blocks(b"begincidchar", b"endcidchar"))
    assert [block.token_values(include_words=True) for block in blocks] == [[b"<01>", b"7"]]


def test_inheritance_depth_limit_is_a_subclass_policy() -> None:
    assert CMapDecoder(CODESPACE, inheritance_depth=50).code_space_ranges
    assert LenientDecoder(CODESPACE, inheritance_depth=1).code_space_ranges
    with pytest.raises(ValueError, match="nesting too deep"):
        LenientDecoder(CODESPACE, inheritance_depth=2)
