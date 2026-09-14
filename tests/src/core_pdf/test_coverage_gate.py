"""Coverage gates reject partial collection and independent metric regressions."""

import hashlib
from copy import deepcopy
from io import BytesIO

import pytest

from scripts.check_coverage import check_coverage
from scripts.install_ocr_ci import install_model


@pytest.fixture
def report():
    counts = {
        "covered_lines": 83,
        "num_statements": 100,
        "covered_branches": 72,
        "num_branches": 100,
    }
    return {
        "meta": {"branch_coverage": True},
        "totals": counts,
        "files": {"src/core_pdf/example.py": {"summary": counts.copy()}},
    }


@pytest.fixture
def baseline():
    return {"statements": {"covered": 82, "total": 100}, "branches": {"covered": 71, "total": 100}}


def test_complete_report_passes_without_rounding_up_floor(report, baseline):
    baseline["statements"] = {"covered": 830001, "total": 1000000}
    with pytest.raises(ValueError, match="statements coverage regressed"):
        check_coverage(report, baseline, set(report["files"]))
    baseline["statements"]["covered"] = 830000
    assert len(check_coverage(report, baseline, set(report["files"]))) == 2


@pytest.mark.parametrize("key", ["covered_lines", "covered_branches"])
def test_either_metric_regressing_fails(report, baseline, key):
    report["totals"][key] = report["files"]["src/core_pdf/example.py"]["summary"][key] = 0
    with pytest.raises(ValueError, match="coverage regressed"):
        check_coverage(report, baseline, set(report["files"]))


@pytest.mark.parametrize("change", ["missing", "extra", "branches", "totals"])
def test_partial_or_inconsistent_reports_fail(report, baseline, change):
    expected = set(report["files"])
    if change == "missing":
        expected.add("packages/core-pdf-ocr/src/core_pdf_ocr/unused.py")
    elif change == "extra":
        report["files"]["outside.py"] = deepcopy(next(iter(report["files"].values())))
    elif change == "branches":
        report["meta"]["branch_coverage"] = False
    else:
        report["totals"]["covered_lines"] += 1
    with pytest.raises(ValueError):
        check_coverage(report, baseline, expected)


@pytest.mark.parametrize("invalid", ["engine", "model", None])
def test_ocr_installer_checks_pins_before_writing(monkeypatch, tmp_path, invalid):
    data = b"fixture model"
    pins = {
        "version": "tesseract fixture",
        "model_url": "https://example.invalid/model",
        "model_sha256": hashlib.sha256(data).hexdigest(),
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: BytesIO(data))
    if invalid == "model":
        pins["model_sha256"] = "bad"
    version = "wrong" if invalid == "engine" else pins["version"] + "\nlibraries"
    destination = tmp_path / "tessdata"
    if invalid:
        with pytest.raises(ValueError, match="mismatch"):
            install_model(destination, version, pins)
        assert not destination.exists()
    else:
        install_model(destination, version, pins)
        assert (destination / "eng.traineddata").read_bytes() == data
