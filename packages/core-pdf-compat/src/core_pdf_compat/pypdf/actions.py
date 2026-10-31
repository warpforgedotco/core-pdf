# Follows pypdf's action handling, Copyright (c) 2006-2008, Mathieu Fenniak,
# under its BSD-style license.
"""pypdf's page actions: entries of a page's additional-actions dictionary."""

from __future__ import annotations

import logging
from abc import ABC
from enum import StrEnum, unique
from typing import Any

from .errors import ParseError
from .generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NullObject,
    TextStringObject,
    is_null_or_none,
)

logger = logging.getLogger(__name__)

_NOT_A_DICTIONARY = (
    "The type in a page object's additional-actions key must be a DictionaryObject: received type "
)


@unique
class PageTrigger(StrEnum):
    OPEN = "open"
    CLOSE = "close"

    @property
    def _name_object(self) -> NameObject:
        return NameObject("/O" if self is PageTrigger.OPEN else "/C")


def _chain_tail(head: DictionaryObject) -> DictionaryObject:
    """The last action reached through /Next, stopping at a cycle (ISO 32000-2 §12.6.2)."""
    current = head
    visited: set[int] = set()
    while not is_null_or_none(following := current.get("/Next")):
        if not isinstance(following, (ArrayObject, DictionaryObject)):
            raise TypeError(
                "An action dictionary’s Next entry must be an Action dictionary "
                f"or an array of Action dictionaries: received type {type(following).__name__}."
            )
        if id(following) in visited:
            logger.warning("Detected cycle in the action tree for %s", current)
            break
        visited.add(id(following))
        current = following[-1] if isinstance(following, ArrayObject) else following
    return current


class Action(DictionaryObject, ABC):
    def __init__(self) -> None:
        super().__init__()
        self[NameObject("/Type")] = NameObject("/Action")
        self[NameObject("/Next")] = NullObject()

    @classmethod
    def _create_new(cls, page: Any, trigger: PageTrigger, action: Action) -> None:
        name = trigger._name_object
        if "/AA" not in page:
            page[NameObject("/AA")] = DictionaryObject({name: action})
            return
        if isinstance(page["/AA"], NullObject):
            page[NameObject("/AA")] = DictionaryObject()
        actions = page["/AA"]
        if not isinstance(actions.get_object(), DictionaryObject):
            received = type(actions).__name__
            if getattr(page.pdf, "strict", False):
                raise ParseError(f"{_NOT_A_DICTIONARY}{received}.")
            logger.warning("%s%s.", _NOT_A_DICTIONARY, received)
            return
        head = actions.get(name)
        if is_null_or_none(head):
            actions.update({name: action})
            return
        if not isinstance(head, DictionaryObject):
            raise ParseError(f"{_NOT_A_DICTIONARY}{type(head).__name__}.")
        _chain_tail(head)[NameObject("/Next")] = action

    @classmethod
    def _delete(cls, page: Any, trigger: PageTrigger) -> None:
        if "/AA" not in page:
            return
        actions = page["/AA"]
        name = trigger._name_object
        if name not in actions:
            return
        del actions[name]
        if not actions:
            del page["/AA"]


class JavaScript(Action):
    def __init__(self, java_script: str) -> None:
        super().__init__()
        self[NameObject("/S")] = NameObject("/JavaScript")
        self[NameObject("/JS")] = TextStringObject(java_script)
