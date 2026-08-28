# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled PDF dictionary lookup kernels."""

from __future__ import annotations


def lookup_dict_key_default(value: object, key: str, default: object = None) -> object:
    if not isinstance(value, dict):
        return default
    return value.get(key, default)


def lookup_dict_key(value: object, key: str) -> object:
    return lookup_dict_key_default(value, key, None)


__all__ = (
    "lookup_dict_key",
    "lookup_dict_key_default",
)
