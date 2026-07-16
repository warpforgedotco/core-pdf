from core_pdf.impl.engine.extraction.common import observation_resolver, page_geometry
from core_pdf.impl.engine.extraction.ocr.postprocess import prune_weak_ocr_artifact_line_text


def resolved_table_fusion_line(text: str) -> observation_resolver.ResolvedTextLine:
    observation = page_geometry.PageObservation(
        kind="table_ocr_line",
        source="table_fusion_text",
        text=text,
    )
    return observation_resolver.ResolvedTextLine(
        text,
        observation,
        contributing_observations=(observation,),
    )


def resolved_full_page_line(
    text: str,
    *,
    confidence: float | None = None,
) -> observation_resolver.ResolvedTextLine:
    observation = page_geometry.PageObservation(
        kind="ocr_textline",
        source="full_page_simple",
        text=text,
        confidence=confidence,
    )
    return observation_resolver.ResolvedTextLine(
        text,
        observation,
        contributing_observations=(observation,),
    )


def test_prunes_geometryless_table_fusion_toc_leader_tail_keep_page_number() -> None:
    line = resolved_table_fusion_line(
        "CHAPTER RIVE: (6 THEFBI'S CONDUCT NOVEMBER OF THE PRELIMINARY 1995 .............224++-222"
    )

    assert (
        prune_weak_ocr_artifact_line_text(line)
        == "CHAPTER RIVE: (6 THEFBI'S CONDUCT NOVEMBER OF THE PRELIMINARY 1995 222"
    )


def test_prunes_geometryless_table_fusion_toc_leader_tail_without_page_number() -> None:
    line = resolved_table_fusion_line(
        "CHAPTER SIK: (U) THBPREDICATE, ........005sscsescsseeeessseesseeeenee D3"
    )

    assert prune_weak_ocr_artifact_line_text(line) == "CHAPTER SIK: (U) THBPREDICATE,"


def test_prunes_geometryless_table_fusion_toc_leader_tail_with_clean_final_page() -> None:
    line = resolved_table_fusion_line(
        "CHAPTER NINE: (U) THE SEARCH OF WEN HO LEB'S COMPUTER ......-444..396"
    )

    assert (
        prune_weak_ocr_artifact_line_text(line)
        == "CHAPTER NINE: (U) THE SEARCH OF WEN HO LEB'S COMPUTER 396"
    )


def test_does_not_prune_geometry_backed_table_fusion_toc_text() -> None:
    observation = page_geometry.PageObservation(
        kind="table_ocr_line",
        source="table_fusion_text",
        text="CHAPTER ONE ........ 1",
        bbox=(0.0, 0.0, 10.0, 10.0),
    )
    line = observation_resolver.ResolvedTextLine(
        observation.text,
        observation,
        contributing_observations=(observation,),
    )

    assert prune_weak_ocr_artifact_line_text(line) == "CHAPTER ONE ........ 1"


def test_does_not_prune_non_table_fusion_toc_text() -> None:
    line = resolved_full_page_line("CHAPTER ONE ........ 1")

    assert prune_weak_ocr_artifact_line_text(line) == "CHAPTER ONE ........ 1"


def test_prunes_weak_full_page_toc_leader_tail_keep_page_number() -> None:
    line = resolved_full_page_line(
        "LEE AND SYLVIA LEE: DECEMBER 1998 TO MARCH 1999... 2... .000.00.++ 629",
        confidence=66.0,
    )

    assert (
        prune_weak_ocr_artifact_line_text(line)
        == "LEE AND SYLVIA LEE: DECEMBER 1998 TO MARCH 1999 629"
    )


def test_does_not_prune_strong_full_page_toc_text() -> None:
    line = resolved_full_page_line(
        "LEE AND SYLVIA LEE: DECEMBER 1998 TO MARCH 1999... 2... .000.00.++ 629",
        confidence=95.0,
    )

    assert (
        prune_weak_ocr_artifact_line_text(line)
        == "LEE AND SYLVIA LEE: DECEMBER 1998 TO MARCH 1999... 2... .000.00.++ 629"
    )
