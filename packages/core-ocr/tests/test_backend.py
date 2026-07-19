from core_ocr.impl.backend import rgba_image_to_bmp
from core_ocr.impl.types import OcrImage


def test_rgba_image_to_bmp_writes_bottom_up_bgr_pixels() -> None:
    image = OcrImage(
        bytes(
            (
                10,
                20,
                30,
                255,
                40,
                50,
                60,
                255,
                70,
                80,
                90,
                255,
                100,
                110,
                120,
                255,
            )
        ),
        width=2,
        height=2,
        bytes_per_pixel=4,
        bytes_per_line=8,
    )

    bmp = rgba_image_to_bmp(image)

    assert bmp is not None
    assert bmp[:2] == b"BM"
    assert bmp[54:] == bytes(
        (
            90,
            80,
            70,
            120,
            110,
            100,
            0,
            0,
            30,
            20,
            10,
            60,
            50,
            40,
            0,
            0,
        )
    )


def test_rgb_image_to_bmp_writes_bottom_up_bgr_pixels() -> None:
    image = OcrImage(
        bytes(
            (
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
            )
        ),
        width=2,
        height=2,
        bytes_per_pixel=3,
        bytes_per_line=6,
    )

    bmp = rgba_image_to_bmp(image)

    assert bmp is not None
    assert bmp[54:] == bytes(
        (
            90,
            80,
            70,
            120,
            110,
            100,
            0,
            0,
            30,
            20,
            10,
            60,
            50,
            40,
            0,
            0,
        )
    )
