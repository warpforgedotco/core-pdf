# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_jpeg.impl.codecs.jpx import markers as jpx_markers
from core_jpeg.impl.codecs.jpx import reconstruction as jpx_reconstruction
from core_jpeg.impl.codecs.jpx import tiles as jpx_tiles
from core_jpeg.impl.codecs.jpx.boxes import Jp2Parser
from core_jpeg.impl.codecs.jpx.packets import JpxPacketStreamConsumed
from core_jpeg.impl.codecs.jpx.params import (
    JPX_SUPPORTED_CODEBLOCK_STYLE,
    JpxCodingParams,
    JpxComponentCodingParams,
    JpxProgressionChange,
    coding_component_quant_guard_bits,
    coding_component_quant_steps,
    copy_component_coding_params,
    validate_jpx_coding_params,
)
from core_jpeg.impl.codecs.jpx.structures import (
    BitStream,
    JpxTilePart,
    JpxTilePartHeader,
    TileComponent,
)
from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError


class JpxImage:
    __slots__ = (
        "width",
        "height",
        "components",
        "expected_width",
        "expected_height",
        "x_origin",
        "y_origin",
        "x_end",
        "y_end",
        "tile_width",
        "tile_height",
        "tile_x_origin",
        "tile_y_origin",
        "tiles_cols",
        "tiles_rows",
        "levels",
        "codeblock_w",
        "codeblock_h",
        "codeblock_style",
        "prog_order",
        "num_layers",
        "capabilities",
        "packet_uses_sop",
        "packet_uses_eph",
        "multiple_component_transform",
        "precincts",
        "component_coding_params",
        "progression_changes",
        "roi_shift_by_component",
        "quant_guard_bits",
        "quant_guard_bits_by_component",
        "quant_style",
        "quant_style_by_component",
        "quant_steps",
        "tile_part_lengths",
        "tlm_next_index",
        "plm_next_index",
        "plm_packet_lengths",
        "plm_consume_index",
        "ppm_markers",
        "packed_packet_headers",
        "components_data",
        "tiles",
        "decoded_tile_data",
        "reversible",
    )

    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.components = 0
        self.expected_width: int | None = None
        self.expected_height: int | None = None
        self.x_origin = 0
        self.y_origin = 0
        self.x_end = 0
        self.y_end = 0
        self.tile_width = 0
        self.tile_height = 0
        self.tile_x_origin = 0
        self.tile_y_origin = 0
        self.tiles_cols = 0
        self.tiles_rows = 0
        self.levels = 0
        self.codeblock_w = 0
        self.codeblock_h = 0
        self.codeblock_style = 0
        self.prog_order = 0
        self.num_layers = 0
        self.capabilities = 0
        self.packet_uses_sop = False
        self.packet_uses_eph = False
        self.multiple_component_transform = 0
        self.precincts: list[Any] = []
        self.component_coding_params: list[JpxComponentCodingParams | None] = []
        self.progression_changes: list[JpxProgressionChange] = []
        self.roi_shift_by_component: list[int | None] = []
        self.quant_guard_bits = 0
        self.quant_guard_bits_by_component: list[int | None] = []
        self.quant_style = 0
        self.quant_style_by_component: list[int | None] = []
        self.quant_steps: list[list[tuple[int, int]]] = []
        self.tile_part_lengths: list[tuple[int, int]] = []
        self.tlm_next_index = 0
        self.plm_next_index = 0
        # One packet-length list per tile-part, in codestream order (ISO A.7.2).
        self.plm_packet_lengths: list[list[int]] = []
        self.plm_consume_index = 0
        self.ppm_markers: dict[int, bytes] = {}
        self.packed_packet_headers: bytes | None = None
        self.components_data: list[dict[Any, Any]] = []
        self.tiles: list[dict[Any, Any]] = []
        self.decoded_tile_data: list[tuple[int, bytes] | None] = []
        self.reversible = False

    def parse(self, data: bytes) -> bool:
        jp2 = Jp2Parser(data).parse()
        self.expected_width = jp2.width
        self.expected_height = jp2.height
        br = BitStream(jp2.codestream)
        marker = br.read_u16()
        if marker != 0xFF4F:
            return False
        if not self.parse_header(br):
            return False
        br.byte -= 2
        return self.parse_tile_parts(br, len(jp2.codestream))

    def tile_parts(self, data: bytes) -> list[JpxTilePart]:
        jp2 = Jp2Parser(data).parse()
        data = jp2.codestream
        self.expected_width = jp2.width
        self.expected_height = jp2.height
        br = BitStream(data)
        marker = br.read_u16()
        if marker != 0xFF4F:
            raise JpegUnsupportedError("JPXDecode missing SOC marker")
        if not self.parse_header(br):
            raise JpegUnsupportedError("JPXDecode failed to parse codestream header")
        br.byte -= 2
        parts: list[JpxTilePart] = []
        global_params = self.coding_params()
        tile_params: dict[int, JpxCodingParams] = {}
        tile_part_indices: dict[int, int] = {}
        tile_part_counts: dict[int, int] = {}
        while True:
            marker = br.read_u16()
            if marker == 0xFFD9:
                jpx_tiles.validate_tile_part_completion(
                    self,
                    tile_part_indices=tile_part_indices,
                    tile_part_counts=tile_part_counts,
                )
                return parts
            if marker != 0xFF90:
                raise JpegParseError("expected JPX SOT marker")
            header = self.read_tile_part_header(
                br,
                data_len=len(data),
                tile_params=tile_params,
                global_params=global_params,
            )
            self.validate_tile_part_header(
                header,
                tile_part_indices=tile_part_indices,
                tile_part_counts=tile_part_counts,
            )
            tile_params[header.tile_index] = header.coding_params
            parts.append(
                JpxTilePart(
                    tile_index=header.tile_index,
                    tile_part_index=header.tile_part_index,
                    tile_part_count=header.tile_part_count,
                    coding_params=header.coding_params,
                    payload=data[header.payload_start : header.payload_end],
                    packet_headers=header.packet_headers,
                    packet_lengths=header.packet_lengths,
                )
            )
            br.byte = header.payload_end

    def read_tile_part_header(
        self,
        br: BitStream,
        data_len: int,
        tile_params: dict[int, JpxCodingParams] | None = None,
        global_params: JpxCodingParams | None = None,
    ) -> JpxTilePartHeader:
        return jpx_markers.read_tile_part_header(
            self,
            br,
            data_len,
            tile_params=tile_params,
            global_params=global_params,
        )

    def parse_header(self, br: BitStream) -> bool:
        return jpx_markers.parse_header(self, br)

    def skip_marker_segment(self, br: BitStream) -> None:
        jpx_markers.skip_marker_segment(br)

    def parse_tlm(self, br: BitStream) -> None:
        jpx_markers.parse_tlm(self, br)

    def parse_plm(self, br: BitStream) -> None:
        jpx_markers.parse_plm(self, br)

    def parse_ppm(self, br: BitStream) -> None:
        jpx_markers.parse_ppm(self, br)

    def parse_ppt_marker(self, br: BitStream, markers: dict[int, bytes]) -> None:
        jpx_markers.parse_ppt_marker(br, markers)

    def parse_plt_marker(self, br: BitStream) -> list[int]:
        return jpx_markers.parse_plt_marker(br)

    def parse_crg(self, br: BitStream) -> None:
        jpx_markers.parse_crg(self, br)

    def ppm_packet_headers(self) -> bytes | None:
        return jpx_markers.ppm_packet_headers(self)

    def parse_siz(self, br: BitStream) -> None:
        jpx_markers.parse_siz(self, br)

    def parse_cod(self, br: BitStream) -> None:
        jpx_markers.parse_cod(self, br)

    def parse_cod_params(
        self,
        br: BitStream,
        base_params: JpxCodingParams,
    ) -> JpxCodingParams:
        return jpx_markers.parse_cod_params(self, br, base_params)

    def parse_coc(self, br: BitStream) -> None:
        jpx_markers.parse_coc(self, br)

    def parse_coc_params(
        self,
        br: BitStream,
        base_params: JpxCodingParams,
    ) -> JpxCodingParams:
        return jpx_markers.parse_coc_params(self, br, base_params)

    def parse_qcd(self, br: BitStream) -> None:
        jpx_markers.parse_qcd(self, br)

    def parse_qcd_params(
        self,
        br: BitStream,
        base_params: JpxCodingParams,
    ) -> JpxCodingParams:
        return jpx_markers.parse_qcd_params(self, br, base_params)

    def parse_qcc(self, br: BitStream) -> None:
        jpx_markers.parse_qcc(self, br)

    def parse_qcc_params(
        self,
        br: BitStream,
        base_params: JpxCodingParams,
    ) -> JpxCodingParams:
        return jpx_markers.parse_qcc_params(self, br, base_params)

    def parse_rgn(self, br: BitStream) -> None:
        jpx_markers.parse_rgn(self, br)

    def parse_rgn_params(
        self,
        br: BitStream,
        base_params: JpxCodingParams,
    ) -> JpxCodingParams:
        return jpx_markers.parse_rgn_params(self, br, base_params)

    def parse_poc(self, br: BitStream) -> None:
        jpx_markers.parse_poc(self, br)

    def parse_poc_params(
        self,
        br: BitStream,
        base_params: JpxCodingParams,
    ) -> JpxCodingParams:
        return jpx_markers.parse_poc_params(self, br, base_params)

    def parse_quantization(
        self,
        br: BitStream,
        length: int,
    ) -> tuple[int, int, list[tuple[int, int]]]:
        return jpx_markers.parse_quantization(br, length)

    def coding_params(self) -> JpxCodingParams:
        return JpxCodingParams(
            levels=self.levels,
            codeblock_w=self.codeblock_w,
            codeblock_h=self.codeblock_h,
            codeblock_style=self.codeblock_style,
            prog_order=self.prog_order,
            num_layers=self.num_layers,
            packet_uses_sop=self.packet_uses_sop,
            packet_uses_eph=self.packet_uses_eph,
            multiple_component_transform=self.multiple_component_transform,
            precincts=list(self.precincts),
            component_coding_params=[
                copy_component_coding_params(component_params)
                if component_params is not None
                else None
                for component_params in self.component_coding_params
            ],
            progression_changes=list(self.progression_changes),
            roi_shift_by_component=list(self.roi_shift_by_component),
            quant_guard_bits=self.quant_guard_bits,
            quant_guard_bits_by_component=list(self.quant_guard_bits_by_component),
            quant_style=self.quant_style,
            quant_style_by_component=list(self.quant_style_by_component),
            quant_steps=[list(steps) for steps in self.quant_steps],
            reversible=self.reversible,
        )

    def set_coding_params(self, params: JpxCodingParams) -> None:
        validate_jpx_coding_params(params)
        self.levels = params.levels
        self.codeblock_w = params.codeblock_w
        self.codeblock_h = params.codeblock_h
        self.codeblock_style = params.codeblock_style
        self.prog_order = params.prog_order
        self.num_layers = params.num_layers
        self.packet_uses_sop = params.packet_uses_sop
        self.packet_uses_eph = params.packet_uses_eph
        self.multiple_component_transform = params.multiple_component_transform
        self.precincts = list(params.precincts)
        self.component_coding_params = [
            copy_component_coding_params(component_params) if component_params is not None else None
            for component_params in params.component_coding_params
        ]
        self.progression_changes = list(params.progression_changes)
        self.roi_shift_by_component = list(params.roi_shift_by_component)
        self.quant_guard_bits = params.quant_guard_bits
        self.quant_guard_bits_by_component = list(params.quant_guard_bits_by_component)
        self.quant_style = params.quant_style
        self.quant_style_by_component = list(params.quant_style_by_component)
        self.quant_steps = [list(steps) for steps in params.quant_steps]
        self.reversible = params.reversible

    def component_quant_steps(self, component_index: int) -> list[tuple[int, int]]:
        return coding_component_quant_steps(self.coding_params(), component_index)

    def component_quant_guard_bits(self, component_index: int) -> int:
        return coding_component_quant_guard_bits(self.coding_params(), component_index)

    def parse_tile_parts(self, br: BitStream, data_len: int) -> bool:
        if self.codeblock_style & ~JPX_SUPPORTED_CODEBLOCK_STYLE:
            raise JpegUnsupportedError("JPX code-block style includes unsupported HT modes")
        parts: list[JpxTilePart] = []
        global_params = self.coding_params()
        tile_params: dict[int, JpxCodingParams] = {}
        tile_part_indices: dict[int, int] = {}
        tile_part_counts: dict[int, int] = {}
        while True:
            marker = br.read_u16()
            if marker == 0xFFD9:
                jpx_tiles.validate_tile_part_completion(
                    self,
                    tile_part_indices=tile_part_indices,
                    tile_part_counts=tile_part_counts,
                )
                self.decode_tile_parts(parts)
                return True
            if marker != 0xFF90:
                raise JpegParseError("expected JPX SOT marker")
            header = self.read_tile_part_header(
                br,
                data_len,
                tile_params=tile_params,
                global_params=global_params,
            )
            self.validate_tile_part_header(
                header,
                tile_part_indices=tile_part_indices,
                tile_part_counts=tile_part_counts,
            )
            validate_jpx_coding_params(header.coding_params)
            tile_params[header.tile_index] = header.coding_params
            payload = br.data[header.payload_start : header.payload_end]
            parts.append(
                JpxTilePart(
                    tile_index=header.tile_index,
                    tile_part_index=header.tile_part_index,
                    tile_part_count=header.tile_part_count,
                    coding_params=header.coding_params,
                    payload=payload,
                    packet_headers=header.packet_headers,
                    packet_lengths=header.packet_lengths,
                )
            )
            br.byte = header.payload_end

    def validate_tile_part_header(
        self,
        header: JpxTilePartHeader,
        *,
        tile_part_indices: dict[int, int],
        tile_part_counts: dict[int, int],
    ) -> None:
        jpx_tiles.validate_tile_part_header(
            self,
            header,
            tile_part_indices=tile_part_indices,
            tile_part_counts=tile_part_counts,
        )

    def decode_tile_parts(self, parts: list[JpxTilePart]) -> None:
        jpx_tiles.decode_tile_parts(self, parts)

    def decode_tile_parts_with_ppm(
        self,
        parts: list[JpxTilePart],
        packet_headers: bytes,
    ) -> None:
        jpx_tiles.decode_tile_parts_with_ppm(self, parts, packet_headers)

    def parallel_tile_worker_count(self, grouped: dict[int, list[JpxTilePart]]) -> int:
        return jpx_tiles.parallel_tile_worker_count(grouped)

    def worker_config(self) -> dict[str, Any]:
        return jpx_tiles.worker_config(self)

    def new_tile(
        self,
        tx: int,
        ty: int,
        coding_params: JpxCodingParams | None = None,
    ) -> dict[Any, Any]:
        return jpx_tiles.new_tile(self, tx, ty, coding_params)

    def configure_component_precincts(
        self,
        component: TileComponent,
        coding_params: JpxComponentCodingParams | JpxCodingParams | None = None,
    ) -> None:
        jpx_tiles.configure_component_precincts(self, component, coding_params)

    def initialize_tile_slots(self) -> None:
        jpx_tiles.initialize_tile_slots(self)

    def initialize_tiles(self) -> None:
        jpx_tiles.initialize_tiles(self)

    def ensure_tile(
        self,
        tile_index: int,
        coding_params: JpxCodingParams | None = None,
    ) -> dict[Any, Any]:
        return jpx_tiles.ensure_tile(self, tile_index, coding_params)

    def decode_tile_payload(
        self,
        tile_index: int,
        payload: bytes,
        coding_params: JpxCodingParams | None = None,
    ) -> int:
        return jpx_reconstruction.decode_tile_payload(
            self,
            tile_index,
            payload,
            coding_params,
        )

    def decode_tile_payload_stream(
        self,
        tile_index: int,
        payload: bytes,
        coding_params: JpxCodingParams | None = None,
        *,
        packet_headers: bytes | None = None,
        packet_header_offset: int = 0,
    ) -> JpxPacketStreamConsumed:
        return jpx_reconstruction.decode_tile_payload_stream(
            self,
            tile_index,
            payload,
            coding_params,
            packet_headers=packet_headers,
            packet_header_offset=packet_header_offset,
        )

    def image_x_end(self) -> int:
        return jpx_tiles.image_x_end(self)

    def image_y_end(self) -> int:
        return jpx_tiles.image_y_end(self)

    def tile_reference_bounds(self, tx: int, ty: int) -> tuple[int, int, int, int]:
        return jpx_tiles.tile_reference_bounds(self, tx, ty)

    def tile_dimensions(self, tx: int, ty: int) -> tuple[int, int]:
        return jpx_tiles.tile_dimensions(self, tx, ty)

    def component_tile_bounds(
        self,
        tx: int,
        ty: int,
        comp_index: int,
    ) -> tuple[int, int, int, int]:
        return jpx_tiles.component_tile_bounds(self, tx, ty, comp_index)

    def reconstruct_component(
        self,
        comp: TileComponent,
        comp_index: int,
        coding_params: JpxCodingParams | None = None,
    ) -> None:
        jpx_reconstruction.reconstruct_component(
            self,
            comp,
            comp_index,
            coding_params,
        )

    def decode_tile_components(
        self,
        tile: dict[Any, Any],
        coding_params: JpxCodingParams | None = None,
    ) -> bool:
        return jpx_reconstruction.decode_tile_components(self, tile, coding_params)

    def build_image(self, comp: TileComponent, reversible: bool) -> None:
        jpx_reconstruction.build_image(comp, reversible)

    def to_raw(self, component_mode: str = "default") -> bytes:
        return jpx_reconstruction.to_raw(self, component_mode)
