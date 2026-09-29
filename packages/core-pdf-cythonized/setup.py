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

# Pure Python mode sources: every module but a package's __init__ is a kernel.
# Remaining .pyx files (if any) compile alongside them.
SOURCES = sorted(
    str(path)
    for pattern in ("*.py", "*.pyx")
    for path in Path("src").rglob(pattern)
    if path.name != "__init__.py"
)


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
    import numpy
    from Cython.Build import cythonize
    from setuptools import Extension

    return cythonize(
        [
            Extension(
                ".".join(Path(source).relative_to("src").with_suffix("").parts),
                [source],
                include_dirs=[numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")],
            )
            for source in SOURCES
        ],
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        quiet=True,
    )


setup(ext_modules=extensions(), cmdclass={"build_ext": BuildExtWithStrictFloats})
