from core_ocr.impl.rendering import ocr_render_dpi_candidates_for_page


class VectorPageProfile:
    recommended_strategy = "vector_or_table"


class VectorPage:
    extraction_cache: dict[str, object] = {}
    media_box = None

    def get_page_profile(self) -> VectorPageProfile:
        return VectorPageProfile()


def test_vector_or_table_pages_include_high_resolution_retry() -> None:
    assert ocr_render_dpi_candidates_for_page(VectorPage()) == (250, 300, 400)
