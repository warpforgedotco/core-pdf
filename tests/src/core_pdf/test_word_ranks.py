"""Word-rank lookup and resource recovery using small, deterministic indexes."""

import gzip
import io
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from core_pdf.impl._impl.layout import text_rules as rules


def internal_index_bytes(entries: dict[str, int]) -> bytes:
    records = bytearray()
    offsets = []
    for word, rank in sorted(entries.items()):
        offsets.append(len(records))
        records.extend(word.encode("utf-8") + b"\0" + rules.UINT32.pack(rank))
    offsets.append(len(records))
    return (
        rules.WORD_RANK_HEADER.pack(rules.WORD_RANK_MAGIC, len(entries))
        + b"".join(rules.UINT32.pack(offset) for offset in offsets)
        + records
    )


@pytest.fixture(autouse=True)
def internal_clear_rank_caches():
    rules.english_word_ranks.cache_clear()
    rules.word_rank.cache_clear()
    yield
    rules.english_word_ranks.cache_clear()
    rules.word_rank.cache_clear()


@pytest.mark.parametrize("on_disk", [False, True])
def test_index_mapping_lookup_iteration_and_missing_keys(tmp_path, on_disk: bool) -> None:
    entries = {"apple": 7, "pear": 2, "éclair": 11}
    data = internal_index_bytes(entries)
    path = tmp_path / "ranks.bin"
    path.write_bytes(data)
    index = rules.WordRankIndex(str(path) if on_disk else data)
    assert len(index) == 3
    assert list(index) == ["apple", "pear", "éclair"]
    assert dict(index) == entries
    for missing in ("", "aardvark", "orange", "zebra", "食"):
        assert index.lookup(missing) is None
        with pytest.raises(KeyError) as error:
            index[missing]
        assert error.value.args == (missing,)


@pytest.mark.parametrize("on_disk", [False, True])
def test_empty_index_is_a_valid_empty_mapping(tmp_path, on_disk: bool) -> None:
    data = internal_index_bytes({})
    path = tmp_path / "empty.bin"
    path.write_bytes(data)
    index = rules.WordRankIndex(str(path) if on_disk else data)
    assert not index
    assert list(index) == []
    assert index.lookup("word") is None


@pytest.mark.parametrize("on_disk", [False, True])
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"short", "truncated"),
        (b"", "truncated"),
        (rules.WORD_RANK_HEADER.pack(b"INVALID!", 0) + b"\0" * 4, "unsupported"),
        (rules.WORD_RANK_HEADER.pack(rules.WORD_RANK_MAGIC, 20), "unsupported"),
        (rules.WORD_RANK_HEADER.pack(rules.WORD_RANK_MAGIC, 0) + b"\0" * 5, "offsets"),
    ],
)
def test_malformed_indexes_fail_at_open(tmp_path, on_disk: bool, data: bytes, message: str) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        rules.WordRankIndex(str(path) if on_disk else data)


def test_file_without_mappable_descriptor_uses_bytes(monkeypatch) -> None:
    data = internal_index_bytes({"hello": 3})
    monkeypatch.setattr(rules, "open", lambda *args: io.BytesIO(data), raising=False)
    assert dict(rules.WordRankIndex("unmappable")) == {"hello": 3}


@pytest.mark.parametrize("binary", [False, True])
def test_word_rank_normalizes_case_and_rejects_nonwords(monkeypatch, binary: bool) -> None:
    entries = {"hello": 3, "strasse": 5}
    ranks = rules.WordRankIndex(internal_index_bytes(entries)) if binary else entries
    monkeypatch.setattr(rules, "english_word_ranks", lambda: ranks)
    assert rules.word_rank("HELLO") == 3
    assert rules.word_rank("Straße") == 5
    assert rules.word_rank("missing") is None
    for word in ("", "hello!", "123", "two words"):
        assert rules.word_rank(word) is None


def test_source_lists_filter_invalid_entries_and_preserve_count_precedence(tmp_path, monkeypatch):
    (tmp_path / rules.NORVIG_COUNTS).write_bytes(
        gzip.compress(b"Apple 12\ninvalid\ntoo many fields\n123 5\nPear nope\nPEAR 8\n")
    )
    (tmp_path / rules.WORDNINJA_WORDS).write_bytes(gzip.compress(b"apple\n\n123\nBANANA\npear\n"))
    monkeypatch.setattr(rules, "files", lambda package: tmp_path)
    assert rules.english_word_frequencies() == {
        "apple": rules.WordFrequency(12, 1),
        "pear": rules.WordFrequency(8, 6),
        "banana": rules.WordFrequency(0, 4),
    }


def test_gzip_resource_can_be_materialized_when_direct_read_fails(tmp_path, monkeypatch):
    import importlib.resources

    path = tmp_path / "words.gz"
    path.write_bytes(gzip.compress("hello\n世界\n".encode()))

    def unavailable():
        raise OSError("resource requires materialization")

    resource = SimpleNamespace(read_bytes=unavailable)
    monkeypatch.setattr(
        rules, "files", lambda package: SimpleNamespace(joinpath=lambda _: resource)
    )
    monkeypatch.setattr(importlib.resources, "as_file", lambda _: nullcontext(path))
    assert [line.strip() for line in rules.internal_gzipped_wordlist_lines("words.gz")] == [
        "hello",
        "世界",
    ]


@pytest.mark.parametrize("materialize", [False, True])
def test_rank_resource_without_filesystem_path(tmp_path, monkeypatch, materialize: bool):
    import importlib.resources

    data = internal_index_bytes({"hello": 3})
    path = tmp_path / "ranks.bin"
    path.write_bytes(data)

    def read_bytes():
        if materialize:
            raise OSError("resource requires materialization")
        return data

    resource = SimpleNamespace(read_bytes=read_bytes)
    monkeypatch.setattr(
        rules, "files", lambda package: SimpleNamespace(joinpath=lambda _: resource)
    )
    monkeypatch.setattr(importlib.resources, "as_file", lambda _: nullcontext(path))
    assert dict(rules.english_word_ranks()) == {"hello": 3}


def test_invalid_packaged_index_recovers_from_compressed_sources(tmp_path, monkeypatch):
    (tmp_path / rules.WORD_RANK_INDEX).write_bytes(b"broken")
    (tmp_path / rules.NORVIG_COUNTS).write_bytes(gzip.compress(b"hello 12\n"))
    (tmp_path / rules.WORDNINJA_WORDS).write_bytes(gzip.compress(b"world\n"))
    monkeypatch.setattr(rules, "files", lambda package: tmp_path)
    assert rules.english_word_ranks() == {"hello": 1, "world": 1}


@pytest.mark.parametrize("first_valid", [False, True])
def test_compiled_distribution_tries_usable_index_candidates(tmp_path, monkeypatch, first_valid):
    import sys

    layout = tmp_path / "layout"
    local = layout / "data" / "wordlists"
    bundled = tmp_path / "core_pdf" / "impl" / "_impl" / "layout" / "data" / "wordlists"
    local.mkdir(parents=True)
    bundled.mkdir(parents=True)
    (local / rules.WORD_RANK_INDEX).write_bytes(
        internal_index_bytes({"local": 1}) if first_valid else b"broken"
    )
    (bundled / rules.WORD_RANK_INDEX).write_bytes(internal_index_bytes({"bundled": 2}))
    monkeypatch.setattr(rules, "__compiled__", True, raising=False)
    monkeypatch.setattr(rules, "__file__", str(layout / "text_rules.py"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert dict(rules.english_word_ranks()) == ({"local": 1} if first_valid else {"bundled": 2})


def test_compiled_distribution_without_loose_indexes_uses_package_resource(tmp_path, monkeypatch):
    import sys

    packaged = tmp_path / "packaged"
    packaged.mkdir()
    (packaged / rules.WORD_RANK_INDEX).write_bytes(internal_index_bytes({"resource": 4}))
    monkeypatch.setattr(rules, "__compiled__", True, raising=False)
    monkeypatch.setattr(rules, "__file__", str(tmp_path / "layout" / "text_rules.py"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(rules, "files", lambda package: packaged)
    assert dict(rules.english_word_ranks()) == {"resource": 4}
