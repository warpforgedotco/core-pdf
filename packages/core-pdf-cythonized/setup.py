# SPDX-License-Identifier: AGPL-3.0-only
"""Cython build hook.

Kept as a setup.py because the extension list has to be produced by
cythonize() at build time; everything else about the distribution is declared
in pyproject.toml.
"""

from pathlib import Path

from setuptools import setup

# The kernels must reproduce CPython's float arithmetic bit for bit, and
# CPython does not fuse. Left to itself the compiler contracts expressions
# like b*b - 4*a*c into an FMA, which is more accurate and therefore wrong
# here: it moves a root by one ULP and the differential tests fail. Turning
# contraction off is what makes 'identical output' true rather than close.
STRICT_FLOAT_ARGS = ["-ffp-contract=off"]

SOURCES = sorted(str(path) for path in Path("src").rglob("*.pyx"))


def extensions() -> list:
    from Cython.Build import cythonize
    from setuptools import Extension

    modules = [
        Extension(
            source.removeprefix("src/").removesuffix(".pyx").replace("/", "."),
            [source],
            extra_compile_args=STRICT_FLOAT_ARGS,
        )
        for source in SOURCES
    ]
    return cythonize(
        modules,
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        quiet=True,
    )


setup(ext_modules=extensions())
