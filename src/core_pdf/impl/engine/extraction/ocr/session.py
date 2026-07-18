# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import ctypes
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeAlias, cast

from core_ocr.impl import deskew as ocr_deskew
from core_ocr.impl import execution as ocr_execution
from core_ocr.impl.backend import TesseractCtypesBackend
from core_ocr.impl.types import OcrDeskewInfo, OcrImage, OcrTextResult

OcrSessionCacheKey: TypeAlias = Hashable
OcrSessionCacheTuple: TypeAlias = tuple[OcrSessionCacheKey, ...]


@dataclass(frozen=True)
class PreparedOcrImage:
    pix: int
    owns_pix: bool
    source_key: OcrSessionCacheKey
    variant_key: OcrSessionCacheTuple
    deskew_info: OcrDeskewInfo | None = None


class OcrPageSession:
    def __init__(self) -> None:
        self._backend: TesseractCtypesBackend | None = None
        self._backend_loaded = False
        self._source_pix_cache: dict[OcrSessionCacheKey, int] = {}
        self._variant_pix_cache: dict[OcrSessionCacheTuple, PreparedOcrImage] = {}
        self._raw_buffer_cache: dict[int, tuple[object, object]] = {}

    def __enter__(self) -> OcrPageSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def backend(self) -> TesseractCtypesBackend | None:
        if not self._backend_loaded:
            self._backend = TesseractCtypesBackend.from_system()
            self._backend_loaded = True
        return self._backend

    def close(self) -> None:
        backend = self._backend
        leptonica = getattr(backend, "leptonica", None) if backend is not None else None
        if leptonica is None:
            return
        destroyed: set[int] = set()
        for prepared in self._variant_pix_cache.values():
            if not prepared.owns_pix or prepared.pix in destroyed:
                continue
            pix_handle = ctypes.c_void_p(prepared.pix)
            leptonica.pixDestroy(ctypes.byref(pix_handle))
            destroyed.add(prepared.pix)
        for pix in self._source_pix_cache.values():
            if pix in destroyed:
                continue
            pix_handle = ctypes.c_void_p(pix)
            leptonica.pixDestroy(ctypes.byref(pix_handle))
            destroyed.add(pix)
        self._variant_pix_cache.clear()
        self._source_pix_cache.clear()

    def deskew_diagnostics(self) -> tuple[dict[str, object], ...]:
        diagnostics: list[dict[str, object]] = []
        seen: set[OcrDeskewInfo] = set()
        for prepared in self._variant_pix_cache.values():
            info = prepared.deskew_info
            if info is None or info in seen:
                continue
            seen.add(info)
            diagnostics.append(ocr_deskew.deskew_diagnostic(info))
        return tuple(diagnostics)

    def image_to_text_result(
        self,
        image: OcrImage,
        *,
        psm: int,
        variables: ocr_execution.OcrVariables = None,
    ) -> OcrTextResult:
        backend = self.backend
        if backend is None:
            return OcrTextResult("", None)
        prepared = self.prepared_image(image)
        if prepared is None:
            return backend.image_to_text_result(
                image,
                psm=psm,
                resolution=image.resolution or ocr_execution.OCR_DEFAULT_DPI,
                variables=variables,
                buffer_cache=self._raw_buffer_cache,
            )
        return self._image_to_text_result_from_prepared(
            backend,
            prepared,
            image,
            psm=psm,
            variables=variables,
        )

    def image_to_text_results(
        self,
        image: OcrImage,
        *,
        psms: list[int],
        variables: ocr_execution.OcrVariables = None,
    ) -> list[OcrTextResult]:
        if not psms:
            return []
        backend = self.backend
        if backend is None:
            return [OcrTextResult("", None) for _ignored in psms]
        prepared = self.prepared_image(image)
        if prepared is None:
            return backend.image_to_text_results(
                image,
                psms=psms,
                resolution=image.resolution or ocr_execution.OCR_DEFAULT_DPI,
                variables=variables,
                buffer_cache=self._raw_buffer_cache,
            )
        results: list[OcrTextResult] = []
        for psm in psms:
            results.append(
                self._image_to_text_result_from_prepared(
                    backend,
                    prepared,
                    image,
                    psm=psm,
                    variables=variables,
                )
            )
        return results

    def image_region_to_text_result(
        self,
        image: OcrImage,
        rectangle: tuple[int, int, int, int],
        *,
        psm: int,
        variables: ocr_execution.OcrVariables = None,
    ) -> OcrTextResult:
        backend = self.backend
        if backend is None:
            return OcrTextResult("", None)
        prepared = self.prepared_image(image)
        if prepared is None:
            return backend.image_region_to_text_result(
                image,
                ocr_execution.rectangle_for_backend_image(image, rectangle),
                psm=psm,
                resolution=image.resolution or ocr_execution.OCR_DEFAULT_DPI,
                variables=variables,
                buffer_cache=self._raw_buffer_cache,
            )
        return self._image_to_text_result_from_prepared(
            backend,
            prepared,
            image,
            psm=psm,
            variables=variables,
            # ``_image_to_text_result_from_prepared`` maps source coordinates
            # to the prepared image. Passing an already mapped rectangle here
            # scales it a second time and can move it completely out of bounds.
            rectangle=rectangle,
        )

    def image_regions_to_text_results(
        self,
        image: OcrImage,
        requests: list[ocr_execution.RectangleOcrRequest],
    ) -> list[OcrTextResult]:
        if not requests:
            return []
        if any(request.rotate_vertical for request in requests):
            return ocr_execution.ocr_image_regions_to_text_results_with_timeout(
                image,
                requests,
                timeout=None,
            )
        backend = self.backend
        if backend is None:
            return [OcrTextResult("", None) for _ignored in requests]
        prepared = self.prepared_image(image)
        if prepared is None:
            native_requests = [
                (
                    ocr_execution.rectangle_for_backend_image(image, request.rectangle),
                    request.psm,
                    request.variables,
                )
                for request in requests
            ]
            return backend.image_regions_to_text_results(
                image,
                native_requests,
                image.resolution or ocr_execution.OCR_DEFAULT_DPI,
                buffer_cache=self._raw_buffer_cache,
            )
        results: list[OcrTextResult] = []
        for request in requests:
            results.append(
                self._image_to_text_result_from_prepared(
                    backend,
                    prepared,
                    image,
                    psm=request.psm,
                    variables=request.variables,
                    rectangle=request.rectangle,
                )
            )
        return results

    def image_subregion_to_text_results(
        self,
        source_image: OcrImage,
        rectangle: tuple[int, int, int, int],
        subregion_image: OcrImage,
        *,
        psms: list[int],
        variables: ocr_execution.OcrVariables = None,
    ) -> list[OcrTextResult] | None:
        if not psms:
            return []
        backend = self.backend
        if backend is None:
            return [OcrTextResult("", None) for _ignored in psms]
        prepared = self.prepared_subregion_image(
            source_image,
            rectangle,
            subregion_image,
        )
        if prepared is None:
            return None
        results: list[OcrTextResult] = []
        for psm in psms:
            results.append(
                self._image_to_text_result_from_prepared(
                    backend,
                    prepared,
                    subregion_image,
                    psm=psm,
                    variables=variables,
                )
            )
        return results

    def prepared_image(self, image: OcrImage) -> PreparedOcrImage | None:
        backend = self.backend
        if backend is None or not backend.should_use_pix(image):
            return None
        source_key = self.source_key_for_image(image)
        target_width = image.target_width or image.width
        target_height = image.target_height or image.height
        should_deskew = ocr_deskew.full_page_ocr_image_should_be_deskewed(image)
        deskew_signature: OcrSessionCacheTuple = ()
        if should_deskew:
            deskew_signature = (
                "deskew",
                image.source,
                ocr_deskew.OCR_DESKEW_BINARY_THRESHOLD,
                ocr_deskew.OCR_DESKEW_CONFIDENCE_THRESHOLD,
                ocr_deskew.OCR_DESKEW_MIN_ANGLE_DEGREES,
                ocr_deskew.OCR_DESKEW_MAX_ANGLE_DEGREES,
            )
        variant_key = (
            source_key,
            target_width,
            target_height,
            image.resolution or ocr_execution.OCR_DEFAULT_DPI,
            image.clockwise_quarter_turns % 4,
            *deskew_signature,
        )
        cached_variant = self._variant_pix_cache.get(variant_key)
        if cached_variant is not None:
            return cached_variant
        source_pix = self._source_pix_cache.get(source_key)
        if source_pix is None:
            source_pix = backend.source_pix_from_image(self.source_image_for_preparation(image))
            if not source_pix:
                return None
            self._source_pix_cache[source_key] = source_pix
        variant_pix = backend.scale_source_pix(source_pix, image)
        if not variant_pix:
            return None
        prepared_pix = variant_pix
        deskew_info = None
        if should_deskew:
            prepared_pix, deskew_info = backend.deskew_pix(
                variant_pix,
                source=image.source,
                width=target_width,
                height=target_height,
            )
            if not prepared_pix:
                prepared_pix = variant_pix
            if (
                prepared_pix != variant_pix
                and variant_pix != source_pix
                and backend.leptonica is not None
            ):
                variant_handle = ctypes.c_void_p(variant_pix)
                backend.leptonica.pixDestroy(ctypes.byref(variant_handle))
        prepared = PreparedOcrImage(
            pix=prepared_pix,
            owns_pix=prepared_pix != source_pix,
            source_key=source_key,
            variant_key=variant_key,
            deskew_info=deskew_info,
        )
        self._variant_pix_cache[variant_key] = prepared
        return prepared

    def prepared_subregion_image(
        self,
        source_image: OcrImage,
        rectangle: tuple[int, int, int, int],
        subregion_image: OcrImage,
    ) -> PreparedOcrImage | None:
        backend = self.backend
        if backend is None or not backend.should_use_pix(source_image):
            return None
        source_key = self.source_key_for_image(source_image)
        target_width = subregion_image.target_width or subregion_image.width
        target_height = subregion_image.target_height or subregion_image.height
        variant_key = (
            source_key,
            "subregion",
            rectangle,
            subregion_image.width,
            subregion_image.height,
            target_width,
            target_height,
            subregion_image.resolution or ocr_execution.OCR_DEFAULT_DPI,
        )
        cached_variant = self._variant_pix_cache.get(variant_key)
        if cached_variant is not None:
            return cached_variant
        source_pix = self._source_pix_cache.get(source_key)
        if source_pix is None:
            source_pix = backend.source_pix_from_image(
                self.source_image_for_preparation(source_image)
            )
            if not source_pix:
                return None
            self._source_pix_cache[source_key] = source_pix
        variant_pix = backend.clip_scale_source_pix(
            source_pix,
            source_width=source_image.width,
            source_height=source_image.height,
            rectangle=rectangle,
            target_width=target_width,
            target_height=target_height,
            clockwise_quarter_turns=source_image.clockwise_quarter_turns,
        )
        if not variant_pix:
            return None
        prepared = PreparedOcrImage(
            pix=variant_pix,
            owns_pix=True,
            source_key=source_key,
            variant_key=variant_key,
        )
        self._variant_pix_cache[variant_key] = prepared
        return prepared

    def source_key_for_image(self, image: OcrImage) -> OcrSessionCacheKey:
        if image.cache_key is not None:
            return cast(OcrSessionCacheKey, image.cache_key)
        if image.encoded is not None:
            return (
                "encoded",
                image.encoded,
                image.width,
                image.height,
                image.clockwise_quarter_turns % 4,
            )
        return (
            "raw",
            image.data,
            image.width,
            image.height,
            image.bytes_per_pixel,
            image.bytes_per_line,
            image.clockwise_quarter_turns % 4,
        )

    @staticmethod
    def source_image_for_preparation(image: OcrImage) -> OcrImage:
        return OcrImage(
            data=image.data,
            width=image.width,
            height=image.height,
            bytes_per_pixel=image.bytes_per_pixel,
            bytes_per_line=image.bytes_per_line,
            encoded=image.encoded,
            source=image.source,
            cache_key=image.cache_key,
            resolution=image.resolution,
            clockwise_quarter_turns=image.clockwise_quarter_turns,
            page_bbox=image.page_bbox,
            page_clockwise_quarter_turns=image.page_clockwise_quarter_turns,
        )

    def _image_to_text_result_from_prepared(
        self,
        backend: TesseractCtypesBackend,
        prepared: PreparedOcrImage,
        image: OcrImage,
        *,
        psm: int,
        variables: ocr_execution.OcrVariables,
        rectangle: tuple[int, int, int, int] | None = None,
    ) -> OcrTextResult:
        if rectangle is not None and not hasattr(backend.tesseract, "TessBaseAPISetRectangle"):
            return self._result_with_deskew_info(OcrTextResult("", None), prepared)
        pix_handle = ctypes.c_void_p(prepared.pix)
        resolution = image.resolution or ocr_execution.OCR_DEFAULT_DPI
        try:
            backend.tesseract.TessBaseAPISetPageSegMode(backend.api, psm)
            backend.apply_variables(variables)
            backend.tesseract.TessBaseAPISetImage2(backend.api, pix_handle)
            if backend.has_set_source_resolution and resolution > 0:
                backend.tesseract.TessBaseAPISetSourceResolution(backend.api, resolution)
            if rectangle is not None:
                source_rectangle = backend.clamp_rectangle(
                    rectangle,
                    image.width,
                    image.height,
                )
                if source_rectangle is None:
                    return self._result_with_deskew_info(OcrTextResult("", None), prepared)
                if prepared.deskew_info is not None:
                    clamped = ocr_deskew.source_rectangle_to_deskew_rectangle(
                        source_rectangle,
                        image,
                        prepared.deskew_info,
                    )
                else:
                    clamped = backend.rectangle_for_image(image, source_rectangle)
                if clamped is None:
                    return self._result_with_deskew_info(OcrTextResult("", None), prepared)
                x0, y0, x1, y1 = clamped
                backend.tesseract.TessBaseAPISetRectangle(
                    backend.api,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                )
            result = backend.current_image_text_result(image, resolution)
            return self._result_with_deskew_info(result, prepared)
        finally:
            if backend.has_clear_adaptive_classifier:
                backend.tesseract.TessBaseAPIClearAdaptiveClassifier(backend.api)
            if backend.has_clear:
                backend.tesseract.TessBaseAPIClear(backend.api)

    @staticmethod
    def _result_with_deskew_info(
        result: OcrTextResult,
        prepared: PreparedOcrImage,
    ) -> OcrTextResult:
        if prepared.deskew_info is None:
            return result
        return ocr_deskew.restore_ocr_result_geometry(result, prepared.deskew_info)
