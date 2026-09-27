from importlib import resources

from core_adobe_fonts.cmap.resources import (
    RESOURCE_PACKAGE,
    cmap_directories,
    resolve_cmap_resource,
)


def walked(name: str) -> bytes | None:
    root = resources.files(RESOURCE_PACKAGE).joinpath("cmaps")
    if not root.is_dir():
        return None
    deprecated = None
    candidates = [(root, None)]
    while candidates:
        current, parent_name = candidates.pop()
        if parent_name == "CMap":
            child = current.joinpath(name)
            if child.is_file():
                if "/deprecated/" not in str(child):
                    return child.read_bytes()
                deprecated = child
            continue
        candidates.extend((child, child.name) for child in current.iterdir() if child.is_dir())
    return deprecated.read_bytes() if deprecated is not None else None


def test_every_packaged_name_resolves_as_the_walk_resolved_it() -> None:
    names = {entry.name for directory in cmap_directories() for entry in directory.iterdir()}
    assert names
    for name in sorted(names | {"NoSuchCMap"}):
        assert resolve_cmap_resource(name) == walked(name), name


def test_the_directories_are_listed_once() -> None:
    assert cmap_directories() is cmap_directories()


def test_unsafe_names_are_refused() -> None:
    for name in ("", ".", "..", "a/b", "a\\\\b"):
        assert resolve_cmap_resource(name) is None
