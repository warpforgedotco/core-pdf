# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext

CONTRACTION_FLAGS = {
    "unix": ["-ffp-contract=off"],
    "mingw32": ["-ffp-contract=off"],
    "msvc": ["/fp:precise"],
}

OPTIMIZATION_FLAGS = {
    "unix": ["-O3"],
    "mingw32": ["-O3"],
    "msvc": ["/O2"],
}

SOURCES = sorted(str(path) for path in Path("src").rglob("*.pyx"))


class BuildExtWithStrictFloats(build_ext):
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
