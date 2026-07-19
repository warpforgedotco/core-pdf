from core_pdf.impl.engine.extraction.tables.grid import intersections_to_cells


def test_intersections_to_cells_indexes_points_by_row_and_column() -> None:
    intersections = {
        (0.0, 10.0): {"v": [("left", 0.0, 0.0, 10.0)], "h": [("top", 0.0, 0.0, 10.0)]},
        (10.0, 10.0): {
            "v": [("right", 10.0, 0.0, 10.0)],
            "h": [("top", 0.0, 0.0, 10.0)],
        },
        (0.0, 0.0): {
            "v": [("left", 0.0, 0.0, 10.0)],
            "h": [("bottom", 0.0, 0.0, 10.0)],
        },
        (10.0, 0.0): {
            "v": [("right", 10.0, 0.0, 10.0)],
            "h": [("bottom", 0.0, 0.0, 10.0)],
        },
    }

    assert intersections_to_cells(intersections) == [(0.0, 10.0, 10.0, 0.0)]
