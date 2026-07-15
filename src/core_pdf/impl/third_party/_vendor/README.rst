Vendored Dependencies
=====================

This directory follows pip's vendoring model:

* vendored libraries are released PyPI packages;
* versions are pinned in ``vendor.txt``;
* vendored library code should not be edited manually;
* local changes, when unavoidable, belong in ``tools/vendoring/patches``.

Run ``uv run --group vendor vendoring sync -v`` to refresh the vendored tree.

Vendored packages
-----------------

* ``fonttools``: used for faithful TrueType/OpenType parsing behavior.
