"""Verify built workspace wheels in an isolated installation outside the checkout."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile

MODEL_WHEEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)

# Model installation belongs to the installer. A facade import or classification
# must not try to repair its environment by downloading a dependency at runtime.
OFFLINE_RUNTIME = """
import sys

def reject_runtime_download(event, arguments):
    if event in {"socket.connect", "subprocess.Popen", "os.system"}:
        raise AssertionError(f"runtime dependency download attempted: {event}")

sys.addaudithook(reject_runtime_download)
"""

SPEC_SMOKE = """
import importlib
import importlib.util
import pkgutil
import zlib
from importlib import resources
from types import SimpleNamespace

import core_pdf_spec
import numpy
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.model import GraphicsState
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_filters.pipeline import decode_stream_data
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.color import color_space_paints
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace, parse_device_n_attributes
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.s_08_graphics.pdf_function import compile_pdf_function
from core_pdf_spec.s_09_fonts.cmap_resources import resolve_cmap_decoder
from core_pdf_spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf_spec.s_11_transparency.groups import composite_knockout_element, remove_group_backdrop
from core_pdf_spec.s_11_transparency.soft_masks import SoftMask, parse_soft_mask
from core_pdf_spec.standards import PdfVersion
from core_pdf_spec.types import PdfName

for module in pkgutil.walk_packages(core_pdf_spec.__path__, core_pdf_spec.__name__ + "."):
    importlib.import_module(module.name)
assert importlib.util.find_spec("core_pdf") is None
assert importlib.util.find_spec("core_pdf_ocr") is None
assert importlib.util.find_spec("core_pdf_validate") is None
assert PdfVersion.parse("2.0").recognized
assert importlib.util.find_spec("fontTools") is None
assert importlib.util.find_spec("imagecodecs") is None
lexer = PdfLexer(b"<< /Type /Page >>")
try:
    assert str(lexer.parse_dictionary_or_stream()["Type"]) == "Page"
finally:
    lexer.close()
assert decode_stream_data(zlib.compress(b"PDF"), {"Filter": "FlateDecode"}) == b"PDF"
assert resolve_cmap_decoder("UniJIS-UTF16-V") is not None
assert glyph_name_to_unicode("uni00410042") == "AB"
attributes = parse_device_n_attributes(
    {
        "Subtype": "NChannel",
        "Process": {
            "ColorSpace": "DeviceCMYK",
            "Components": ["Cyan", "Magenta", "Yellow", "Black"],
        },
    },
    ("Black", "Magenta"),
)
assert attributes.process is not None
assert attributes.process.color_space.kind == "DeviceCMYK"
assert attributes.process.component_indices == (None, 1, None, 0)
assert not color_space_paints(ColorSpace("DeviceN", ((0, 1),) * 2, colorants=("None", "None")))
assert color_space_paints(ColorSpace("DeviceN", ((0, 1),) * 2, colorants=("None", "Ink")))
color, alpha = remove_group_backdrop(
    numpy.array([[0.75]]), numpy.array([1.0]),
    numpy.array([[1.0]]), numpy.array([1.0]), numpy.array([0.5]),
)
assert numpy.allclose(color, [[0.5]]) and numpy.allclose(alpha, [0.5])
color, alpha, group_alpha = composite_knockout_element(
    numpy.array([[0.0, 0.0, 1.0]]), numpy.array([1.0]),
    backdrop_components=numpy.array([[1.0, 1.0, 1.0]]), backdrop_alpha=numpy.array([1.0]),
    element_components=numpy.array([[1.0, 1.0, 1.0]]), element_alpha=numpy.array([1.0]),
    shape=numpy.array([1.0]), group_alpha=numpy.array([1.0]),
    element_group_alpha=numpy.array([0.0]),
)
assert numpy.allclose(color, [[1.0, 1.0, 1.0]])
assert numpy.allclose(alpha, [1.0]) and numpy.allclose(group_alpha, [0.0])
frame = ContentStreamFrame(PdfStream(), {}, IDENTITY_MATRIX, 1, None)
assert not frame.group_knockout
frame = ContentStreamFrame(PdfStream(), {}, IDENTITY_MATRIX, 1, None, group_knockout=True)
assert frame.group_knockout
assert GraphicsState().text_knockout
calculator = PdfStream(
    dictionary={
        "FunctionType": 4, "Domain": [0, 1], "Range": [0, 1],
        "Filter": PdfName.of("FlateDecode"),
    },
    raw_data=zlib.compress(b"{ dup .5 lt { 1 exch sub } { dup mul } ifelse }"),
    spec={"Filter": PdfName.of("FlateDecode")},
)
evaluate = compile_pdf_function(calculator)
assert evaluate(0.25) == (0.75,)
assert evaluate(0.75) == (0.5625,)
assert evaluate(-1.0) == (1.0,) and evaluate(2.0) == (1.0,)
assert evaluate(0.25) == (0.75,)
resolver = ObjectResolver(b"", {})
try:
    state = ContentInterpreter(resolver, SimpleNamespace(text_boundary=lambda *args: None), None)
    state.apply_extgstate({"TK": False})
    state.op_BT((), 0)
    state.apply_extgstate({"TK": "ignored invalid value", "ca": 0.4})
    state.op_ET((), 0)
    assert not state.graphics.text_knockout and state.graphics.fill_opacity == 0.4
    state.apply_extgstate({"TK": True})
    assert state.graphics.text_knockout
    group = PdfStream(
        dictionary={
            "Subtype": PdfName.of("Form"), "BBox": [0, 0, 1, 1],
            "Group": {"S": PdfName.of("Transparency")},
        },
        raw_data=b"0 0 1 1 re f",
    )
    mask_dictionary = {
        "S": PdfName.of("Alpha"), "G": group,
        "TR": calculator,
    }
    mask = parse_soft_mask(mask_dictionary, resolver, ctm=IDENTITY_MATRIX)
    assert isinstance(mask, SoftMask) and mask.group is group
    assert mask.transfer is not None and mask.transfer(0.0) == (1.0,)
    assert mask.transfer(0.75) == (0.5625,)
    state.apply_extgstate({"SMask": mask_dictionary})
    installed_mask = state.graphics.soft_mask
    assert isinstance(installed_mask, SoftMask)
    paints = []
    ignore = lambda *args: None
    state.sink = SimpleNamespace(
        text_boundary=ignore, save_graphics=ignore, restore_graphics=ignore,
        enter_stream=ignore, exit_stream=ignore,
        paint_path=lambda state, *args: paints.append(
            (state.graphics.soft_mask, state.graphics.fill_opacity)
        ),
    )
    frame = state.append_form_xobject(group, 0)
    assert frame is not None and frame.form_bbox == (0, 0, 1, 1)
    state.stream_executor.consume_frame(frame)
    assert paints == [(None, 1.0)] and state.graphics.soft_mask is installed_mask
    state.apply_extgstate({"SMask": PdfName.of("None")})
    assert state.graphics.soft_mask is None
finally:
    resolver.close()
assert resources.files("core_pdf_spec").joinpath("py.typed").is_file()
data = resources.files("core_pdf_spec._vendor.font_data")
assert data.joinpath("LICENSE.txt").is_file()
assert data.joinpath("LICENSE.external.txt").is_file()
print("Standalone spec wheel: parsing, NChannel, K/TK/SMask, Type 4, codecs and resources passed")
"""

CORE_SMOKE = """
import importlib.util
import sys
from pathlib import Path
import core_pdf
import core_pdf_spec.exceptions as errors
from core_pdf import PdfDocument
from core_pdf.api.compat.pypdf import PdfReader
assert importlib.util.find_spec("spacy") is None
assert importlib.util.find_spec("en_core_web_sm") is None
assert core_pdf.PdfError is errors.PdfError
assert core_pdf.PdfParseError is errors.PdfParseError
with PdfDocument(Path(sys.argv[1])) as document:
    assert document.page_count() > 0
    assert document.standards.effective_version is not None
assert len(PdfReader(sys.argv[1]).pages) > 0
assert not any(name == "core_pdf_ocr" or name.startswith("core_pdf_ocr.") for name in sys.modules)
assert not any(name.startswith("core_pdf_validate") for name in sys.modules)
print("Core wheel: native and pypdf APIs work without NLP dependencies")
"""

UNSTRUCTURED_FAILURE_SMOKE = (
    OFFLINE_RUNTIME
    + """
import importlib
import importlib.abc
import importlib.util
import types

mode, failure, expected_spacy, model_url = sys.argv[1:]
assert (importlib.util.find_spec("spacy") is not None) == (expected_spacy == "present")
assert importlib.util.find_spec("en_core_web_sm") is None
expected_cause = None
if failure == "load-oserror":
    expected_cause = OSError("simulated unreadable English model")
    model = types.ModuleType("en_core_web_sm")
    def broken_load():
        raise expected_cause
    model.load = broken_load
    sys.modules[model.__name__] = model
elif failure == "import-error":
    expected_cause = ImportError("simulated missing transitive NLP dependency")
    class BrokenModelFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "en_core_web_sm":
                raise expected_cause
            return None
    sys.meta_path.insert(0, BrokenModelFinder())
else:
    assert failure == "missing-model"

try:
    if mode == "direct":
        importlib.import_module("core_pdf.api.compat.unstructured")
    elif mode == "parent":
        from core_pdf.api.compat import partition_pdf
    elif mode == "elements":
        from core_pdf.api.compat.unstructured import Element
    elif mode == "child-elements":
        importlib.import_module("core_pdf.api.compat.unstructured._elements")
    elif mode == "retry-elements":
        try:
            importlib.import_module("core_pdf.api.compat.unstructured")
        except ImportError:
            pass
        else:
            raise AssertionError("initial facade import must fail without the model")
        importlib.import_module("core_pdf.api.compat.unstructured._elements")
    else:
        raise AssertionError(f"unexpected import mode: {mode}")
except ImportError as error:
    assert "pip install" in str(error), str(error)
    assert "core-pdf[unstructured]" in str(error), str(error)
    assert model_url in str(error), str(error)
    if expected_cause is None:
        assert isinstance(error.__cause__, ModuleNotFoundError), repr(error.__cause__)
        assert error.__cause__.name == "en_core_web_sm", repr(error.__cause__)
    else:
        assert error.__cause__ is expected_cause, repr(error.__cause__)
else:
    raise AssertionError(f"{mode} import succeeded with {failure}")
print(f"Unstructured wheel: {mode} import explains {failure}, spaCy {expected_spacy}")
"""
)

UNSTRUCTURED_SMOKE = (
    OFFLINE_RUNTIME
    + """
import importlib.util
import en_core_web_sm

assert importlib.util.find_spec("unstructured") is None
loads = []
original_load = en_core_web_sm.load
def counted_load(*args, **kwargs):
    loads.append(None)
    return original_load(*args, **kwargs)
en_core_web_sm.load = counted_load

from core_pdf.api.compat import partition_pdf
from core_pdf.api.compat import unstructured as facade
from core_pdf.api.compat.unstructured import _classification as classification

assert len(loads) == 1, "the model must load during facade import"
pipeline = classification.internal_NLP
assert pipeline.meta["version"] == "3.8.0"
assert {"tagger", "parser"} <= set(pipeline.pipe_names)
assert partition_pdf is facade.partition_pdf
assert facade.Element("Example").text == "Example"
for text, category in (
    ("Document Overview", facade.Title),
    (
        "The researchers measured the samples and found that the treatment improved recovery.",
        facade.NarrativeText,
    ),
):
    assert classification.internal_element_class(text, (0, 40, 80, 60), 100) is category
    first = classification.internal_nlp_features(text)
    previous_hits = classification.internal_nlp_features.cache_info().hits
    assert classification.internal_nlp_features(text) is first
    assert classification.internal_nlp_features.cache_info().hits == previous_hits + 1
assert len(loads) == 1
print("Unstructured extra: real model semantics, eager initialization and caches passed")
"""
)

OCR_SMOKE = """
import core_pdf
import core_pdf_ocr
assert issubclass(core_pdf_ocr.PdfDocument, core_pdf.PdfDocument)
assert issubclass(core_pdf_ocr.PdfPage, core_pdf.PdfPage)
print("OCR wheel: document/page integration passed")
"""

VALIDATION_SMOKE = """
import sys
from importlib import resources
from core_pdf_validate import validate, VeraPdfBackend

assert not any(name == "core_pdf" or name.startswith("core_pdf.") for name in sys.modules)
report = validate(
    b"not a readable PDF",
    profiles=("pdfa-1b", "pdfx-4"),
    backend=VeraPdfBackend(executable="/nonexistent/core-pdf-validator"),
)
assert report.results[0].execution_status == "engine_unavailable"
assert report.results[1].execution_status == "unsupported_profile"
assert all(result.conformance == "not_checked" for result in report.results)
assert not any(name == "core_pdf" or name.startswith("core_pdf.") for name in sys.modules)
assert resources.files("core_pdf_validate").joinpath("py.typed").is_file()
print("Validation wheel: optional imports, explicit targets, and unavailable engine passed")
"""


def run(*arguments: str, cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def check_unstructured_metadata(wheel: Path) -> str:
    """Check the published extra and return its spaCy installation requirement."""
    with ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_names) == 1, metadata_names
        metadata = message_from_bytes(archive.read(metadata_names[0]))
    assert "unstructured" in metadata.get_all("Provides-Extra", [])
    nlp_requirements: dict[str, str] = {}
    expected_versions = {"spacy": {">=3.8.15", "<3.9.0"}, "en-core-web-sm": {"==3.8.0"}}
    for requirement in metadata.get_all("Requires-Dist", []):
        assert "@" not in requirement, f"published dependency uses a direct URL: {requirement}"
        dependency, separator, marker = requirement.partition(";")
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(.*)", dependency.strip())
        assert match is not None, requirement
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name not in expected_versions:
            continue
        assert name not in nlp_requirements, f"duplicate NLP requirement: {requirement}"
        assert separator, f"NLP dependency must be optional: {requirement}"
        assert marker.strip() in {
            'extra == "unstructured"',
            "extra == 'unstructured'",
        }, f"NLP dependency must be conditional only on the unstructured extra: {requirement}"
        versions = set(match[2].replace(" ", "").strip("()").split(","))
        assert versions == expected_versions[name], requirement
        nlp_requirements[name] = dependency.strip()
    assert nlp_requirements.keys() == expected_versions.keys(), nlp_requirements
    print("Core wheel metadata: named, versioned optional NLP dependencies passed", flush=True)
    return nlp_requirements["spacy"]


def check_unstructured_failures(python: Path, work: Path, *, spacy_installed: bool) -> None:
    failures = (
        ("missing-model", "load-oserror", "import-error") if spacy_installed else ("missing-model",)
    )
    for failure in failures:
        for mode in ("direct", "parent", "elements", "child-elements", "retry-elements"):
            run(
                str(python),
                "-I",
                "-c",
                UNSTRUCTURED_FAILURE_SMOKE,
                mode,
                failure,
                "present" if spacy_installed else "absent",
                MODEL_WHEEL_URL,
                cwd=work,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the four built wheels")
    parser.add_argument("--fixture", type=Path, required=True, help="A valid one-page PDF")
    args = parser.parse_args()
    wheel_dir = args.directory.resolve()
    wheels: dict[str, Path] = {}
    for name in ("core_pdf_spec", "core_pdf", "core_pdf_ocr", "core_pdf_validate"):
        matches = sorted(wheel_dir.glob(f"{name}-[0-9]*.whl"))
        if len(matches) != 1:
            parser.error(f"expected one {name} wheel in {wheel_dir}; found {len(matches)}")
        wheels[name] = matches[0]
    with ZipFile(wheels["core_pdf_spec"]) as archive:
        assert not any(
            "__pycache__" in name or name.endswith((".pyc", ".so")) for name in archive.namelist()
        ), "the spec wheel contains cached or compiled source artifacts"
    spacy_requirement = check_unstructured_metadata(wheels["core_pdf"])
    with tempfile.TemporaryDirectory(prefix="core-pdf-spec-wheel-") as temporary:
        work = Path(temporary)
        environment = work / "environment"
        run("uv", "venv", "--python", sys.executable, str(environment), cwd=work)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        install = ("uv", "pip", "install", "--python", str(python), "--find-links", str(wheel_dir))
        run(*install, str(wheels["core_pdf_spec"]), cwd=work)
        run(str(python), "-I", "-c", SPEC_SMOKE, cwd=work)
        run(*install, str(wheels["core_pdf"]), cwd=work)
        run(str(python), "-I", "-c", CORE_SMOKE, str(args.fixture.resolve()), cwd=work)
        run(str(python), "-I", "-m", "core_pdf", "--help", cwd=work)
        check_unstructured_failures(python, work, spacy_installed=False)
        run(*install, str(wheels["core_pdf_ocr"]), cwd=work)
        run(str(python), "-I", "-c", OCR_SMOKE, cwd=work)
        run(*install, str(wheels["core_pdf_validate"]), cwd=work)
        run(str(python), "-I", "-c", VALIDATION_SMOKE, cwd=work)
        run(*install, spacy_requirement, cwd=work)
        check_unstructured_failures(python, work, spacy_installed=True)
        run(*install, f"{wheels['core_pdf']}[unstructured]", MODEL_WHEEL_URL, cwd=work)
        run(str(python), "-I", "-c", UNSTRUCTURED_SMOKE, cwd=work)


if __name__ == "__main__":
    main()
