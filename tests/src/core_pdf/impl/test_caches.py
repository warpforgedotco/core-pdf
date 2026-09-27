from core_pdf.impl.caches import BoundedDict, IdentityCache
from core_pdf.impl.types import MISSING


def test_a_bounded_dict_clears_everything_once_full() -> None:
    cache: BoundedDict[str, int] = BoundedDict(2)
    assert cache.put("a", 1) == 1
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache == {"c": 3}


def test_an_identity_cache_keys_by_object_and_extra_parts() -> None:
    first: dict[str, int] = {}
    second: dict[str, int] = {}
    cache: IdentityCache[str] = IdentityCache()
    cache.put(first, "first", 1)
    assert cache.get(first, 1) == "first"
    assert cache.get(first, 2) is None
    assert cache.get(second, 1) is None
    assert cache.get(second, 1, default=MISSING) is MISSING


def test_pins_are_part_of_the_identity() -> None:
    value: dict[str, int] = {}
    scope: dict[str, int] = {}
    cache: IdentityCache[int] = IdentityCache()
    cache.put(value, 1, pins=(scope,))
    assert cache.get(value, pins=(scope,)) == 1
    assert cache.get(value, pins=({},)) is None
    assert cache.get(value) is None


def test_none_values_are_distinguished_from_misses() -> None:
    value = object()
    cache: IdentityCache[None] = IdentityCache()
    cache.put(value, None)
    assert cache.get(value, default=MISSING) is None


def test_the_limit_is_read_when_storing() -> None:
    cache: IdentityCache[int] = IdentityCache(3)
    objects = [object() for _ in range(4)]
    for index, value in enumerate(objects[:3]):
        cache.put(value, index)
    assert len(cache) == 3
    cache.limit = 10
    cache.put(objects[3], 3)
    assert len(cache) == 4
    cache.limit = 4
    cache.put(object(), 4)
    assert len(cache) == 1


def test_prebuilt_keys_match_the_variadic_form() -> None:
    value: dict[str, int] = {}
    scope: dict[str, int] = {}
    cache: IdentityCache[int] = IdentityCache()
    cache.put_key(value, (id(value), id(scope), 2), 1, (scope,))
    assert cache.get(value, 2, pins=(scope,)) == 1
    cache.put(value, 3, 4)
    assert cache.get_key(value, (id(value), 4)) == 3
    assert cache.get_key(object(), (id(value), 4), MISSING) is MISSING


def test_discard_removes_only_the_matching_entry() -> None:
    value = object()
    other = object()
    cache: IdentityCache[int] = IdentityCache()
    cache.put(value, 1)
    cache.put(other, 2)
    cache.discard(value)
    assert list(cache) == [id(other)]
    cache.clear()
    assert not cache
