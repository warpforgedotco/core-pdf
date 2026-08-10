# SPDX-License-Identifier: AGPL-3.0-only
# nuitka-project: --lto=yes
# nuitka-project: --remove-output
# nuitka-project: --include-package=core_pdf
# nuitka-project: --include-package-data=core_pdf
# nuitka-project: --nofollow-import-to=core_pdf._vendor.fontTools.misc.testTools
from __future__ import annotations

from core_pdf.cli import main

if __name__ == "__main__":
    main()
