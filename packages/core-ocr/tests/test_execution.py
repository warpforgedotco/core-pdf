from core_ocr.impl.execution import (
    rotate_ocr_image_half_turn,
    rotate_ocr_image_right_angle,
)
from core_ocr.impl.types import OcrImage


def test_half_turn_uses_contiguous_rgba_rows() -> None:
    image = OcrImage(
        bytes(
            (
                1,
                0,
                0,
                255,
                2,
                0,
                0,
                255,
                3,
                0,
                0,
                255,
                4,
                0,
                0,
                255,
            )
        ),
        width=2,
        height=2,
        bytes_per_pixel=4,
        bytes_per_line=8,
        source="test",
    )

    rotated = rotate_ocr_image_half_turn(image)

    assert rotated.data == bytes(
        (
            4,
            0,
            0,
            255,
            3,
            0,
            0,
            255,
            2,
            0,
            0,
            255,
            1,
            0,
            0,
            255,
        )
    )


def test_half_turn_uses_contiguous_bitmap_rows() -> None:
    image = OcrImage(
        bytes((0b00000001, 0b00000010, 0b00000011, 0b00000100)),
        width=2,
        height=2,
        bytes_per_pixel=1,
        bytes_per_line=2,
        source="test",
    )

    rotated = rotate_ocr_image_half_turn(image)

    assert rotated.data == bytes((0b00000100, 0b00000011, 0b00000010, 0b00000001))


def test_right_angle_rotation_uses_strided_rgb_rows() -> None:
    image = OcrImage(
        bytes(
            (
                1,
                2,
                3,
                4,
                5,
                6,
                99,
                7,
                8,
                9,
                10,
                11,
                12,
                98,
            )
        ),
        width=2,
        height=2,
        bytes_per_pixel=3,
        bytes_per_line=7,
        source="test",
    )

    clockwise = rotate_ocr_image_right_angle(image, clockwise=True)
    counterclockwise = rotate_ocr_image_right_angle(image, clockwise=False)

    assert clockwise.data == bytes((7, 8, 9, 1, 2, 3, 10, 11, 12, 4, 5, 6))
    assert counterclockwise.data == bytes((4, 5, 6, 10, 11, 12, 1, 2, 3, 7, 8, 9))
