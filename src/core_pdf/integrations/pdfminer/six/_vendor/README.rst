Vendored Dependencies
=====================

This directory follows pip's vendoring model:

* vendored libraries are released PyPI packages;
* versions are pinned in ``vendor.txt``;
* upstream license files are retained alongside their packages;
* vendored library code must not be edited manually;
* imports are rewritten into the vendored namespace during synchronization.

Vendored packages
-----------------

* ``pdfminer.six``: PDF parsing and layout compatibility implementation.
