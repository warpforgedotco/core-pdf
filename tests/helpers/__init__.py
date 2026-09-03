# SPDX-License-Identifier: AGPL-3.0-only
"""Shared test construction helpers.

Importable as ``tests.helpers.*`` because ``pyproject.toml`` puts the repository
root on ``pythonpath``. Keep production-shaped builders here (real dataclasses,
real in-memory PDFs) so tests do not stand in for those objects with
``SimpleNamespace`` and break when production stops using ``getattr``.
"""
