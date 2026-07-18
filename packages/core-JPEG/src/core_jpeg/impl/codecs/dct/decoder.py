# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
from typing import Any, TypeAlias

from core_jpeg.impl.codecs.dct import markers as dct_markers
from core_jpeg.impl.codecs.dct import scan as dct_scan
from core_jpeg.impl.codecs.dct.bitstream import JpegBitReader
from core_jpeg.impl.codecs.dct.huffman import HuffmanTable
from core_jpeg.impl.codecs.dct.idct import idct_2d as transform_idct_2d
from core_jpeg.impl.errors import JpegError, JpegUnsupportedError

PASS1_BITS = 2
JpegComponent: TypeAlias = dict[str, Any]
ScanMeta: TypeAlias = tuple[JpegComponent, HuffmanTable, HuffmanTable, int, int]


class JPEGDecoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.quant_tables: dict[int, list[int]] = {}
        self.huffman_tables: dict[tuple[int, int], HuffmanTable] = {}
        self.components: list[JpegComponent] = []
        self.scans: list[dict[str, Any]] = []
        self.width = 0
        self.height = 0
        self.max_h = 0
        self.max_v = 0
        self.mcu_width = 0
        self.mcu_height = 0
        self.scan_data: bytes = b""
        self.restart_interval = 0
        self.mcu_counter = 0
        self.progressive = False
        self.adobe_transform: int | None = None
        self.idct_temp: list[float] = [0.0] * 64
        self.block: list[int] = [0] * 64

    def idct_2d(self, block: list[int]) -> list[int]:
        return transform_idct_2d(block, self.idct_temp)

    def read(self, n: int) -> bytes:
        return dct_markers.read(self, n)

    def read_marker(self) -> int:
        return dct_markers.read_marker(self)

    def parse(self) -> None:
        dct_markers.parse(self)

    def parse_dqt(self) -> None:
        dct_markers.parse_dqt(self)

    def parse_dht(self) -> None:
        dct_markers.parse_dht(self)

    def parse_sof0(self) -> None:
        dct_markers.parse_sof0(self)

    def parse_dri(self) -> None:
        dct_markers.parse_dri(self)

    def parse_app14(self) -> None:
        dct_markers.parse_app14(self)

    def parse_sos(self) -> None:
        dct_markers.parse_sos(self)

    def locate_eoi(self) -> None:
        dct_markers.locate_eoi(self)

    def skip_progressive_scan(self) -> None:
        dct_markers.skip_progressive_scan(self)

    def decode_mcu_block(
        self,
        reader: JpegBitReader,
        comp: JpegComponent,
        dc_tbl: HuffmanTable,
        ac_tbl: HuffmanTable,
        block: list[int],
    ) -> bool:
        return dct_scan.decode_mcu_block(reader, comp, dc_tbl, ac_tbl, block)

    def decode(self) -> bytes:
        return dct_scan.decode(self)

    def compose_rgb(self, comp_buf: dict[int, list[int]], comp_w: dict[int, int]) -> bytes:
        return dct_scan.compose_rgb(self, comp_buf, comp_w)

    def decode_progressive(self) -> bytes:
        return dct_scan.decode_progressive(self)

    def decode_progressive_dc_first_scan(
        self,
        reader: JpegBitReader,
        scan_meta: list[ScanMeta],
        coeff_buf: dict[int, list[int]],
        mcu_cols: int,
        mcu_rows: int,
        al: int,
    ) -> None:
        dct_scan.decode_progressive_dc_first_scan(
            self, reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, al
        )

    def decode_progressive_dc_refine_scan(
        self,
        reader: JpegBitReader,
        scan_meta: list[ScanMeta],
        coeff_buf: dict[int, list[int]],
        mcu_cols: int,
        mcu_rows: int,
        al: int,
    ) -> None:
        dct_scan.decode_progressive_dc_refine_scan(
            self, reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, al
        )

    def decode_progressive_ac_first_scan(
        self,
        reader: JpegBitReader,
        scan_meta: list[ScanMeta],
        coeff_buf: dict[int, list[int]],
        mcu_cols: int,
        mcu_rows: int,
        ss: int,
        se: int,
        al: int,
    ) -> None:
        dct_scan.decode_progressive_ac_first_scan(
            self, reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, ss, se, al
        )

    def decode_progressive_ac_refine_scan(
        self,
        reader: JpegBitReader,
        scan_meta: list[ScanMeta],
        coeff_buf: dict[int, list[int]],
        mcu_cols: int,
        mcu_rows: int,
        ss: int,
        se: int,
        al: int,
    ) -> None:
        dct_scan.decode_progressive_ac_refine_scan(
            self, reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, ss, se, al
        )

    def handle_restart(self, reader: JpegBitReader) -> bool:
        return dct_scan.handle_restart(self, reader)

    def validate_decode_state(self) -> None:
        if not self.components:
            raise JpegUnsupportedError("JPEGDecode missing frame header")
        if len(self.components) == 4 and self.adobe_transform not in {None, 0, 2}:
            raise JpegUnsupportedError("unsupported JPEG CMYK/YCCK color transform")
        for comp in self.components:
            qt_id = comp["qt"]
            if qt_id not in self.quant_tables:
                raise JpegUnsupportedError("JPEGDecode missing quantization table")
        if self.progressive:
            if not self.scans:
                raise JpegUnsupportedError("JPEGDecode missing scan data")
            for scan in self.scans:
                for item in scan["components"]:
                    if (0, item["dc_tbl"]) not in self.huffman_tables:
                        raise JpegUnsupportedError("JPEGDecode missing Huffman table")
                    if (1, item["ac_tbl"]) not in self.huffman_tables:
                        raise JpegUnsupportedError("JPEGDecode missing Huffman table")
            return
        if not self.scan_data:
            raise JpegUnsupportedError("JPEGDecode missing scan data")
        for comp in self.components:
            if "dc_tbl" not in comp or "ac_tbl" not in comp:
                raise JpegUnsupportedError("JPEGDecode missing scan component")
            if (0, comp["dc_tbl"]) not in self.huffman_tables:
                raise JpegUnsupportedError("JPEGDecode missing Huffman table")
            if (1, comp["ac_tbl"]) not in self.huffman_tables:
                raise JpegUnsupportedError("JPEGDecode missing Huffman table")

    @classmethod
    def from_data(cls, data: bytes) -> bytes:
        try:
            decoder = cls(data)
            decoder.parse()
            decoder.validate_decode_state()
            return decoder.decode()
        except (
            JpegError,
            struct.error,
            OSError,
            KeyError,
            IndexError,
            ValueError,
        ) as exc:
            raise JpegUnsupportedError("JPEGDecode failed") from exc


def decode_dct(data: bytes) -> bytes:
    return JPEGDecoder.from_data(data)
