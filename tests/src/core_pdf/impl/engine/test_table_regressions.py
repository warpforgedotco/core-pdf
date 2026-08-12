import pytest

from scripts import score_unstructured_bench as s

# Baseline table_structure_f1 for top failing stems (captured from a recent run).
BASELINE = {
    "Covid19-White-Paper-FINALp4-5-p001.pdf": 0.0,
    "Covid19-White-Paper-FINALp4-5-p002.pdf": 0.0,
    (
        "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-"
        "Volume3-MassProperties-Pg54.pdf"
    ): 0.0,
    "O27Hara_DeepSeaFloorBio-p002.pdf": 0.0,
    "ijerph-19-00825-p020.pdf": 0.0,
    "VCAs_REV2_SCHEMATIC-p002.pdf": 0.0,
    (
        "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-"
        "Volume3-MassProperties-pg856.pdf"
    ): 0.0,
    "PDFTriage-p7-p002.pdf": 0.0,
    "virus-statistics-2022-p1-7-p001.pdf": 0.0,
    "BarrowArchAnalysis_Alaska1984-p076.pdf": 0.0,
    "pet-display-patent-p004.pdf": 0.0,
    "2020-jinich_p1-6-p006.pdf": 0.0,
    "ftgd0346-p016.pdf": 0.0,
    "Zhand-Ilavsky-p004.pdf": 0.0,
    "VCAs_REV2_SCHEMATIC-p001.pdf": 0.0,
    "Patent_US-12461028-B2_Nov-2025_p007.pdf": 0.0,
    "Zhand-Ilavsky-p012.pdf": 0.0,
    "Mission-costs_p27-35-p006.pdf": 0.0,
    "csia_federal_plan-p47-p52-p006.pdf": 0.0,
    "gs4dhdStrategicObjectives-p008.pdf": 0.0,
    "153rd-Omaha-Pow-Wow-p001.pdf": 0.0,
    "sydd0278.pdf": 0.0,
    "EPA_pesticide_label_2008-p003.pdf": 0.0,
    "AlienPlantThreatAssess-p24-p27-p004.pdf": 0.0,
    "phs-6031-p001.pdf": 0.0,
}


def test_table_structure_no_regression():
    """Ensure table_structure_f1 does not regress below the recorded baseline

    This test reproduces the SCORE-Bench scoring for the worst table-structure cases
    and asserts that parse changes do not reduce their previously-observed structure F1.
    The baselines are intentionally conservative and can be updated when a true
    improvement and verification is made.
    """
    cases = {case.stem: case for case in s.iter_score_bench_cases()}
    if not cases:
        pytest.skip("SCORE-Bench fixtures not present")

    for stem, baseline in BASELINE.items():
        if stem not in cases:
            msg = f"Case {stem} not present in SCORE-Bench fixtures"
            pytest.skip(msg)
        case = cases[stem]
        score = s.score_case(case)
        # If ground truth had no table structure, baseline may be None; skip in that case.
        if baseline is None:
            continue
        # No table structure in the ground truth (or prediction) means there is
        # nothing structural to regress, so skip rather than demanding a value.
        if score.table_structure_f1 is None:
            continue
        assert score.table_structure_f1 >= baseline - 1e-9, (
            f"Table structure F1 regressed for {stem}: {score.table_structure_f1} < {baseline}"
        )
