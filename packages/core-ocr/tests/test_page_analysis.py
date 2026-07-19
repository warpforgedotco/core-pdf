from core_ocr.impl.page_analysis import assess_native_text


def test_native_text_assessment_marks_corrupt_unicode_layer_suspect() -> None:
    assessment = assess_native_text(
        "DESCRIPTION ⠾ LAUNCH ⠬ TABLE "
        "MASS PROPERTIES CSM LM SLA EARTH LAUNCH TRANS DOCK "
        "WEIGHT INERTIAS COORDINATES AMENDMENT TABLE CONTINUED "
        "MASS VALUE"
    )

    assert assessment.status == "suspect"
    assert assessment.reason == "uninterpretable_unicode"
    assert assessment.uninterpretable_count == 2


def test_native_text_assessment_keeps_normal_technical_text_trusted() -> None:
    assessment = assess_native_text("Temperature −12 °C ± 2")

    assert assessment.status == "trusted"
    assert assessment.reason == "no_decode_artifacts"
