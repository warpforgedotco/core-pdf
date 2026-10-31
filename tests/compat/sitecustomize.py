"""Install the facade aliases in subprocesses, which upstream CLI tests spawn."""

import os

if os.environ.get("CORE_PDF_COMPAT_ALIASES"):
    import compat_alias  # noqa: F401
