from core_ocr.impl.rendering import (
    dense_vector_text_table_page,
    ocr_render_dpi_candidates_for_page,
)


class ContentStream:
    def __init__(self, decoded_bytes: int) -> None:
        self.decoded_bytes = decoded_bytes


class VectorPageProfile:
    recommended_strategy = "vector_or_table"


class VectorPage:
    extraction_cache: dict[str, object] = {}
    media_box = None

    def get_page_profile(self) -> VectorPageProfile:
        return VectorPageProfile()


def test_vector_or_table_pages_include_high_resolution_retry() -> None:
    assert ocr_render_dpi_candidates_for_page(VectorPage()) == (250, 300, 400)


class DenseVectorTextTableProfile:
    recommended_strategy = "text_table"
    has_path_ops = True
    content_streams = (ContentStream(180_000),)


class DenseVectorTextTablePage:
    chars = ()
    extraction_cache: dict[str, object] = {}
    media_box = None

    def get_page_profile(self) -> DenseVectorTextTableProfile:
        return DenseVectorTextTableProfile()


def test_nearly_textless_dense_vector_tables_include_high_resolution_retry() -> None:
    assert ocr_render_dpi_candidates_for_page(DenseVectorTextTablePage()) == (300, 400, 475)
    assert dense_vector_text_table_page(DenseVectorTextTablePage())
