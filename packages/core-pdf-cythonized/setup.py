# SPDX-License-Identifier: AGPL-3.0-only
"""Cython build hook.

Kept as a setup.py because the extension list has to be produced by
cythonize() at build time, and because the float-contraction and optimization
flags have to be chosen per compiler; everything else about the distribution
is declared in pyproject.toml.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext

# The kernels must reproduce CPython's float arithmetic bit for bit, and
# CPython does not fuse. Left to itself a compiler contracts expressions like
# b*b - 4*a*c into an FMA, which is more accurate and therefore wrong here: it
# moves a root by one ULP and the golden vectors fail.
#
# These are per-compiler because there is no portable spelling. Passing the
# GCC/Clang flag to MSVC would be worse than passing nothing: MSVC warns and
# carries on, so the build would succeed with the wrong semantics and only the
# golden vectors would notice -- on a platform this repo does not otherwise
# test.
CONTRACTION_FLAGS = {
    "unix": ["-ffp-contract=off"],
    "mingw32": ["-ffp-contract=off"],
    # MSVC has no direct equivalent. /fp:precise is the default and does not
    # contract on x64, but it is stated explicitly so an inherited setting
    # cannot quietly enable it, and so ARM64 (which otherwise may contract) is
    # covered too.
    "msvc": ["/fp:precise"],
}

# Stated rather than inherited. setuptools takes the optimization level from
# the interpreter's CFLAGS, and a CFLAGS in the environment replaces that
# rather than adding to it -- so a shell that exports CFLAGS for include
# paths, as a Homebrew setup commonly does, silently built every kernel at
# -O0. Nothing failed: the golden vectors pass at any level. The kernels just
# ran up to ten times slower, and a benchmark taken on such a build measured
# the wrong thing. These come after CFLAGS on the command line, so they win.
#
# -O3 is what CPython itself is built with. None of these levels licenses
# float reassociation; that takes -ffast-math, which nothing here passes.
OPTIMIZATION_FLAGS = {
    "unix": ["-O3"],
    "mingw32": ["-O3"],
    "msvc": ["/O2"],
}

SOURCES = sorted(str(path) for path in Path("src").rglob("*.pyx"))


class BuildExtWithStrictFloats(build_ext):
    """Apply the per-compiler flags once the compiler is actually known."""

    def build_extensions(self) -> None:
        compiler_type = self.compiler.compiler_type
        flags = CONTRACTION_FLAGS.get(compiler_type)
        if flags is None:
            raise RuntimeError(
                f"unknown compiler {compiler_type!r}: refusing to build "
                "without a float-contraction flag, because the kernels would compile "
                "with semantics the golden vectors do not describe"
            )
        flags = [*OPTIMIZATION_FLAGS[compiler_type], *flags]
        for extension in self.extensions:
            extension.extra_compile_args = [*extension.extra_compile_args, *flags]
        super().build_extensions()


def extensions() -> list:
    from Cython.Build import cythonize

    return cythonize(
        SOURCES,
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        quiet=True,
    )


setup(ext_modules=extensions(), cmdclass={"build_ext": BuildExtWithStrictFloats})
