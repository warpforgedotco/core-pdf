# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import ctypes
import ctypes.util
import math
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from core_ocr.impl import deskew as ocr_deskew
from core_ocr.impl.types import (
    BMP_MAX_BYTES,
    LEPTONICA_LIBRARY_FILENAMES,
    LEPTONICA_LIBRARY_NAMES,
    TESSERACT_DEFAULT_VARIABLES,
    TESSERACT_LIBRARY_FILENAMES,
    TESSERACT_LIBRARY_NAMES,
    TESSERACT_OEM_LSTM_ONLY,
    TESSERACT_RIL_BLOCK,
    TESSERACT_RIL_PARA,
    TESSERACT_RIL_SYMBOL,
    TESSERACT_RIL_TEXTLINE,
    TESSERACT_RIL_WORD,
    OcrComponentBox,
    OcrDeskewInfo,
    OcrImage,
    OcrIteratorLayout,
    OcrTextChoice,
    OcrTextResult,
    leptonica_pix_size_is_supported,
    ocr_observations_from_rows,
    raw_ocr_image_size_is_supported,
)


class TesseractCtypesBackend:
    __slots__ = (
        "api",
        "has_box_create",
        "has_box_destroy",
        "has_clear",
        "has_clear_adaptive_classifier",
        "has_init2",
        "has_pix_clip_rectangle",
        "has_pix_convert_to_1",
        "has_pix_find_skew",
        "has_pix_rotate",
        "has_set_image2",
        "has_set_source_resolution",
        "has_pix_scale_to_size",
        "has_pix_rotate_orth",
        "leptonica",
        "tesseract",
    )

    def __init__(
        self,
        tesseract: ctypes.CDLL,
        leptonica: ctypes.CDLL | None = None,
    ) -> None:
        self.tesseract = tesseract
        self.leptonica = leptonica
        self.configure_symbols(tesseract)
        if leptonica is not None:
            self.configure_leptonica_symbols(leptonica)
        self.has_box_create = bool(leptonica is not None and hasattr(leptonica, "boxCreate"))
        self.has_box_destroy = bool(leptonica is not None and hasattr(leptonica, "boxDestroy"))
        self.has_pix_clip_rectangle = bool(
            leptonica is not None and hasattr(leptonica, "pixClipRectangle")
        )
        self.has_pix_convert_to_1 = bool(
            leptonica is not None and hasattr(leptonica, "pixConvertTo1")
        )
        self.has_pix_find_skew = bool(leptonica is not None and hasattr(leptonica, "pixFindSkew"))
        self.has_pix_rotate = bool(leptonica is not None and hasattr(leptonica, "pixRotate"))
        self.has_pix_scale_to_size = bool(
            leptonica is not None and hasattr(leptonica, "pixScaleToSize")
        )
        self.has_pix_rotate_orth = bool(
            leptonica is not None and hasattr(leptonica, "pixRotateOrth")
        )
        api = tesseract.TessBaseAPICreate()
        if not api:
            raise RuntimeError("TessBaseAPICreate failed")
        self.has_clear = hasattr(tesseract, "TessBaseAPIClear")
        self.has_clear_adaptive_classifier = hasattr(
            tesseract, "TessBaseAPIClearAdaptiveClassifier"
        )
        self.has_init2 = hasattr(tesseract, "TessBaseAPIInit2")
        self.has_set_image2 = hasattr(tesseract, "TessBaseAPISetImage2")
        self.has_set_source_resolution = hasattr(tesseract, "TessBaseAPISetSourceResolution")
        if self.has_init2:
            init_status = tesseract.TessBaseAPIInit2(
                api,
                None,
                b"eng",
                TESSERACT_OEM_LSTM_ONLY,
            )
        else:
            init_status = tesseract.TessBaseAPIInit3(api, None, b"eng")
        if init_status != 0:
            tesseract.TessBaseAPIDelete(api)
            raise RuntimeError("TessBaseAPIInit3 failed")
        tesseract.TessBaseAPISetVariable(api, b"debug_file", b"/dev/null")
        tesseract.TessBaseAPISetVariable(api, b"classify_enable_learning", b"0")
        self.api = api

    @classmethod
    @lru_cache(maxsize=1)
    def from_system(cls) -> TesseractCtypesBackend | None:
        tesseract_path = cls.find_library(
            TESSERACT_LIBRARY_NAMES,
            executable="tesseract",
            filenames=TESSERACT_LIBRARY_FILENAMES,
        )
        if tesseract_path is None:
            return None
        leptonica = cls.load_optional_library(
            LEPTONICA_LIBRARY_NAMES,
            executable="tesseract",
            filenames=LEPTONICA_LIBRARY_FILENAMES,
        )
        try:
            return cls(ctypes.CDLL(tesseract_path), leptonica)
        except (AttributeError, OSError, RuntimeError):
            return None

    @staticmethod
    def find_library(
        names: tuple[str, ...],
        *,
        executable: str | None = None,
        filenames: tuple[str, ...] = (),
    ) -> str | None:
        for name in names:
            path = ctypes.util.find_library(name)
            if path:
                return path
        for directory in TesseractCtypesBackend.library_search_dirs(executable):
            for filename in filenames:
                candidate_path = directory / filename
                if candidate_path.exists():
                    return str(candidate_path)
        return None

    @staticmethod
    def library_search_dirs(executable: str | None) -> tuple[Path, ...]:
        if executable is None:
            return ()
        executable_path = shutil.which(executable)
        if executable_path is None:
            return ()
        path = Path(executable_path)
        prefixes = (path.parent.parent, path.resolve().parent.parent)
        seen: set[Path] = set()
        directories: list[Path] = []
        for prefix in prefixes:
            directory = prefix / "lib"
            if directory not in seen:
                seen.add(directory)
                directories.append(directory)
        return tuple(directories)

    @classmethod
    def load_optional_library(
        cls,
        names: tuple[str, ...],
        *,
        executable: str | None = None,
        filenames: tuple[str, ...] = (),
    ) -> ctypes.CDLL | None:
        path = cls.find_library(names, executable=executable, filenames=filenames)
        if path is None:
            return None
        try:
            return ctypes.CDLL(path)
        except OSError:
            return None

    @staticmethod
    def configure_symbols(tesseract: ctypes.CDLL) -> None:
        tesseract.TessBaseAPICreate.argtypes = []
        tesseract.TessBaseAPICreate.restype = ctypes.c_void_p
        tesseract.TessBaseAPIDelete.argtypes = [ctypes.c_void_p]
        tesseract.TessBaseAPIDelete.restype = None
        tesseract.TessBaseAPIInit3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        tesseract.TessBaseAPIInit3.restype = ctypes.c_int
        if hasattr(tesseract, "TessBaseAPIInit2"):
            tesseract.TessBaseAPIInit2.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            tesseract.TessBaseAPIInit2.restype = ctypes.c_int
        tesseract.TessBaseAPISetPageSegMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        tesseract.TessBaseAPISetPageSegMode.restype = None
        tesseract.TessBaseAPISetVariable.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        tesseract.TessBaseAPISetVariable.restype = ctypes.c_bool
        tesseract.TessBaseAPISetImage.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        tesseract.TessBaseAPISetImage.restype = None
        if hasattr(tesseract, "TessBaseAPISetImage2"):
            tesseract.TessBaseAPISetImage2.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            tesseract.TessBaseAPISetImage2.restype = None
        tesseract.TessBaseAPIGetUTF8Text.argtypes = [ctypes.c_void_p]
        tesseract.TessBaseAPIGetUTF8Text.restype = ctypes.c_void_p
        if hasattr(tesseract, "TessBaseAPIGetComponentImages"):
            tesseract.TessBaseAPIGetComponentImages.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
            ]
            tesseract.TessBaseAPIGetComponentImages.restype = ctypes.c_void_p
        if hasattr(tesseract, "TessBaseAPIGetComponentImages1"):
            tesseract.TessBaseAPIGetComponentImages1.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            tesseract.TessBaseAPIGetComponentImages1.restype = ctypes.c_void_p
        if hasattr(tesseract, "TessBaseAPISetRectangle"):
            tesseract.TessBaseAPISetRectangle.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            tesseract.TessBaseAPISetRectangle.restype = None
        if hasattr(tesseract, "TessBaseAPIMeanTextConf"):
            tesseract.TessBaseAPIMeanTextConf.argtypes = [ctypes.c_void_p]
            tesseract.TessBaseAPIMeanTextConf.restype = ctypes.c_int
        if hasattr(tesseract, "TessBaseAPIRecognize"):
            tesseract.TessBaseAPIRecognize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            tesseract.TessBaseAPIRecognize.restype = ctypes.c_int
        if hasattr(tesseract, "TessBaseAPIGetIterator"):
            tesseract.TessBaseAPIGetIterator.argtypes = [ctypes.c_void_p]
            tesseract.TessBaseAPIGetIterator.restype = ctypes.c_void_p
        if hasattr(tesseract, "TessResultIteratorDelete"):
            tesseract.TessResultIteratorDelete.argtypes = [ctypes.c_void_p]
            tesseract.TessResultIteratorDelete.restype = None
        if hasattr(tesseract, "TessResultIteratorCopy"):
            tesseract.TessResultIteratorCopy.argtypes = [ctypes.c_void_p]
            tesseract.TessResultIteratorCopy.restype = ctypes.c_void_p
        if hasattr(tesseract, "TessResultIteratorNext"):
            tesseract.TessResultIteratorNext.argtypes = [ctypes.c_void_p, ctypes.c_int]
            tesseract.TessResultIteratorNext.restype = ctypes.c_bool
        if hasattr(tesseract, "TessResultIteratorGetUTF8Text"):
            tesseract.TessResultIteratorGetUTF8Text.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            tesseract.TessResultIteratorGetUTF8Text.restype = ctypes.c_void_p
        if hasattr(tesseract, "TessResultIteratorConfidence"):
            tesseract.TessResultIteratorConfidence.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            tesseract.TessResultIteratorConfidence.restype = ctypes.c_float
        if hasattr(tesseract, "TessResultIteratorGetChoiceIterator"):
            tesseract.TessResultIteratorGetChoiceIterator.argtypes = [ctypes.c_void_p]
            tesseract.TessResultIteratorGetChoiceIterator.restype = ctypes.c_void_p
        if hasattr(tesseract, "TessChoiceIteratorDelete"):
            tesseract.TessChoiceIteratorDelete.argtypes = [ctypes.c_void_p]
            tesseract.TessChoiceIteratorDelete.restype = None
        if hasattr(tesseract, "TessChoiceIteratorNext"):
            tesseract.TessChoiceIteratorNext.argtypes = [ctypes.c_void_p]
            tesseract.TessChoiceIteratorNext.restype = ctypes.c_bool
        if hasattr(tesseract, "TessChoiceIteratorGetUTF8Text"):
            tesseract.TessChoiceIteratorGetUTF8Text.argtypes = [ctypes.c_void_p]
            tesseract.TessChoiceIteratorGetUTF8Text.restype = ctypes.c_char_p
        if hasattr(tesseract, "TessChoiceIteratorConfidence"):
            tesseract.TessChoiceIteratorConfidence.argtypes = [ctypes.c_void_p]
            tesseract.TessChoiceIteratorConfidence.restype = ctypes.c_float
        if hasattr(tesseract, "TessPageIteratorBoundingBox"):
            tesseract.TessPageIteratorBoundingBox.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            tesseract.TessPageIteratorBoundingBox.restype = ctypes.c_bool
        if hasattr(tesseract, "TessPageIteratorBaseline"):
            tesseract.TessPageIteratorBaseline.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            tesseract.TessPageIteratorBaseline.restype = ctypes.c_bool
        if hasattr(tesseract, "TessPageIteratorIsAtBeginningOf"):
            tesseract.TessPageIteratorIsAtBeginningOf.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            tesseract.TessPageIteratorIsAtBeginningOf.restype = ctypes.c_bool
        if hasattr(tesseract, "TessBaseAPIClear"):
            tesseract.TessBaseAPIClear.argtypes = [ctypes.c_void_p]
            tesseract.TessBaseAPIClear.restype = None
        if hasattr(tesseract, "TessBaseAPIClearAdaptiveClassifier"):
            tesseract.TessBaseAPIClearAdaptiveClassifier.argtypes = [ctypes.c_void_p]
            tesseract.TessBaseAPIClearAdaptiveClassifier.restype = None
        if hasattr(tesseract, "TessBaseAPISetSourceResolution"):
            tesseract.TessBaseAPISetSourceResolution.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            tesseract.TessBaseAPISetSourceResolution.restype = None
        tesseract.TessBaseAPIEnd.argtypes = [ctypes.c_void_p]
        tesseract.TessBaseAPIEnd.restype = None
        tesseract.TessDeleteText.argtypes = [ctypes.c_void_p]
        tesseract.TessDeleteText.restype = None

    @staticmethod
    def configure_leptonica_symbols(leptonica: ctypes.CDLL) -> None:
        if hasattr(leptonica, "boxCreate"):
            leptonica.boxCreate.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            leptonica.boxCreate.restype = ctypes.c_void_p
        if hasattr(leptonica, "boxDestroy"):
            leptonica.boxDestroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            leptonica.boxDestroy.restype = None
        leptonica.pixReadMem.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        leptonica.pixReadMem.restype = ctypes.c_void_p
        if hasattr(leptonica, "pixGetWidth"):
            leptonica.pixGetWidth.argtypes = [ctypes.c_void_p]
            leptonica.pixGetWidth.restype = ctypes.c_int
        if hasattr(leptonica, "pixGetHeight"):
            leptonica.pixGetHeight.argtypes = [ctypes.c_void_p]
            leptonica.pixGetHeight.restype = ctypes.c_int
        if hasattr(leptonica, "pixClipRectangle"):
            leptonica.pixClipRectangle.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            leptonica.pixClipRectangle.restype = ctypes.c_void_p
        if hasattr(leptonica, "pixScaleToSize"):
            leptonica.pixScaleToSize.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            leptonica.pixScaleToSize.restype = ctypes.c_void_p
        if hasattr(leptonica, "pixRotateOrth"):
            leptonica.pixRotateOrth.argtypes = [ctypes.c_void_p, ctypes.c_int]
            leptonica.pixRotateOrth.restype = ctypes.c_void_p
        if hasattr(leptonica, "pixConvertTo1"):
            leptonica.pixConvertTo1.argtypes = [ctypes.c_void_p, ctypes.c_int]
            leptonica.pixConvertTo1.restype = ctypes.c_void_p
        if hasattr(leptonica, "pixFindSkew"):
            leptonica.pixFindSkew.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            leptonica.pixFindSkew.restype = ctypes.c_int
        if hasattr(leptonica, "pixRotate"):
            leptonica.pixRotate.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            leptonica.pixRotate.restype = ctypes.c_void_p
        if hasattr(leptonica, "boxaGetCount"):
            leptonica.boxaGetCount.argtypes = [ctypes.c_void_p]
            leptonica.boxaGetCount.restype = ctypes.c_int
        if hasattr(leptonica, "boxaGetBoxGeometry"):
            leptonica.boxaGetBoxGeometry.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            leptonica.boxaGetBoxGeometry.restype = ctypes.c_int
        if hasattr(leptonica, "boxaDestroy"):
            leptonica.boxaDestroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            leptonica.boxaDestroy.restype = None
        if hasattr(leptonica, "pixaDestroy"):
            leptonica.pixaDestroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            leptonica.pixaDestroy.restype = None
        leptonica.pixDestroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        leptonica.pixDestroy.restype = None

    def image_to_text(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> str:
        result = self.image_to_text_result(
            image,
            psm,
            resolution,
            variables=variables,
        )
        return result.text

    def image_to_text_result(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> OcrTextResult:
        result = self.recognize_image(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=None,
            buffer_cache=buffer_cache,
        )
        return result

    def image_to_text_results(
        self,
        image: OcrImage,
        psms: list[int],
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> list[OcrTextResult]:
        if not psms:
            return []
        data = image.data
        width = image.width
        height = image.height
        bytes_per_pixel = image.bytes_per_pixel
        bytes_per_line = image.bytes_per_line
        pix = None
        pix_handle = ctypes.c_void_p()
        try:
            if self.should_use_pix(image):
                pix = self.image_to_pix(image, buffer_cache=buffer_cache)
            if pix:
                pix_handle = ctypes.c_void_p(pix)
                self.tesseract.TessBaseAPISetImage2(self.api, pix_handle)
            else:
                if not data:
                    return [OcrTextResult("", None) for _ignored in psms]
                if not raw_ocr_image_size_is_supported(
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                ):
                    return [OcrTextResult("", None) for _ignored in psms]
                buffer = self.raw_image_buffer(data, buffer_cache=buffer_cache)
                self.tesseract.TessBaseAPISetImage(
                    self.api,
                    ctypes.cast(buffer, ctypes.c_void_p),
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                )
            if self.has_set_source_resolution and resolution > 0:
                self.tesseract.TessBaseAPISetSourceResolution(self.api, resolution)
            results: list[OcrTextResult] = []
            self.apply_variables(variables)
            for psm in psms:
                self.tesseract.TessBaseAPISetPageSegMode(self.api, psm)
                results.append(self.current_image_text_result(image, resolution))
                if self.has_clear_adaptive_classifier:
                    self.tesseract.TessBaseAPIClearAdaptiveClassifier(self.api)
            return results
        finally:
            if pix and self.leptonica is not None:
                self.leptonica.pixDestroy(ctypes.byref(pix_handle))
            if self.has_clear:
                self.tesseract.TessBaseAPIClear(self.api)

    def image_region_to_text(
        self,
        image: OcrImage,
        region: tuple[int, int, int, int],
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> str:
        if not hasattr(self.tesseract, "TessBaseAPISetRectangle"):
            return ""
        clamped_region = self.clamp_rectangle(region, image.width, image.height)
        if clamped_region is None:
            return ""
        result = self.recognize_image(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=clamped_region,
            buffer_cache=buffer_cache,
        )
        return result.text

    def image_region_to_text_result(
        self,
        image: OcrImage,
        region: tuple[int, int, int, int],
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> OcrTextResult:
        if not hasattr(self.tesseract, "TessBaseAPISetRectangle"):
            return OcrTextResult("", None)
        clamped_region = self.clamp_rectangle(region, image.width, image.height)
        if clamped_region is None:
            return OcrTextResult("", None)
        return self.recognize_image(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=clamped_region,
            buffer_cache=buffer_cache,
        )

    def image_regions_to_text_results(
        self,
        image: OcrImage,
        requests: list[
            tuple[
                tuple[int, int, int, int],
                int,
                Mapping[str, str | int | float | bool] | None,
            ]
        ],
        resolution: int,
        *,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> list[OcrTextResult]:
        if not hasattr(self.tesseract, "TessBaseAPISetRectangle"):
            return [OcrTextResult("", None) for ignored in requests]
        if not requests:
            return []
        data = image.data
        width = image.width
        height = image.height
        bytes_per_pixel = image.bytes_per_pixel
        bytes_per_line = image.bytes_per_line
        pix = None
        pix_handle = ctypes.c_void_p()
        try:
            if self.should_use_pix(image):
                pix = self.image_to_pix(image, buffer_cache=buffer_cache)
            if pix:
                pix_handle = ctypes.c_void_p(pix)
                self.tesseract.TessBaseAPISetImage2(self.api, pix_handle)
            else:
                if not data:
                    return [OcrTextResult("", None) for ignored in requests]
                if not raw_ocr_image_size_is_supported(
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                ):
                    return [OcrTextResult("", None) for ignored in requests]
                buffer = self.raw_image_buffer(data, buffer_cache=buffer_cache)
                self.tesseract.TessBaseAPISetImage(
                    self.api,
                    ctypes.cast(buffer, ctypes.c_void_p),
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                )
            if self.has_set_source_resolution and resolution > 0:
                self.tesseract.TessBaseAPISetSourceResolution(self.api, resolution)
            results: list[OcrTextResult] = []
            for region, psm, variables in requests:
                clamped_region = self.clamp_rectangle(region, width, height)
                if clamped_region is None:
                    results.append(OcrTextResult("", None))
                    continue
                self.tesseract.TessBaseAPISetPageSegMode(self.api, psm)
                self.apply_variables(variables)
                x0, y0, x1, y1 = clamped_region
                self.tesseract.TessBaseAPISetRectangle(
                    self.api,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                )
                results.append(self.current_image_text_result(image, resolution))
                if self.has_clear_adaptive_classifier:
                    self.tesseract.TessBaseAPIClearAdaptiveClassifier(self.api)
            return results
        finally:
            if pix and self.leptonica is not None:
                self.leptonica.pixDestroy(ctypes.byref(pix_handle))
            if self.has_clear:
                self.tesseract.TessBaseAPIClear(self.api)

    def image_to_word_rows(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.supports_result_iterator_rows():
            return []
        return self.recognize_image_word_rows(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=None,
        )

    def image_to_textline_rows(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.supports_result_iterator_rows():
            return []
        return self.recognize_image_iterator_rows(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=None,
            level=TESSERACT_RIL_TEXTLINE,
        )

    def image_to_symbol_rows(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.supports_result_iterator_rows():
            return []
        return self.recognize_image_iterator_rows(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=None,
            level=TESSERACT_RIL_SYMBOL,
        )

    def image_to_iterator_layout(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> OcrIteratorLayout:
        if not self.supports_result_iterator_rows():
            return OcrIteratorLayout([], [], [])
        rows_by_level, text, confidence = self.recognize_image_iterator_layout(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=None,
            levels=(
                TESSERACT_RIL_TEXTLINE,
                TESSERACT_RIL_WORD,
                TESSERACT_RIL_SYMBOL,
            ),
        )
        return OcrIteratorLayout(
            rows_by_level.get(TESSERACT_RIL_TEXTLINE, []),
            rows_by_level.get(TESSERACT_RIL_WORD, []),
            rows_by_level.get(TESSERACT_RIL_SYMBOL, []),
            text,
            confidence,
        )

    def image_to_component_boxes(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        level: int,
        text_only: bool = True,
        variables: Mapping[str, str | int | float | bool] | None = None,
    ) -> list[OcrComponentBox]:
        if not self.supports_component_boxes():
            return []
        return self.recognize_image_component_boxes(
            image,
            psm,
            resolution,
            level=level,
            text_only=text_only,
            variables=variables,
            rectangle=None,
        )

    def recognize_image(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None,
        rectangle: tuple[int, int, int, int] | None,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> OcrTextResult:
        data = image.data
        width = image.width
        height = image.height
        bytes_per_pixel = image.bytes_per_pixel
        bytes_per_line = image.bytes_per_line
        pix = None
        pix_handle = ctypes.c_void_p()
        try:
            self.tesseract.TessBaseAPISetPageSegMode(self.api, psm)
            self.apply_variables(variables)
            if self.should_use_pix(image):
                pix = self.image_to_pix(image, buffer_cache=buffer_cache)
            if pix:
                pix_handle = ctypes.c_void_p(pix)
                self.tesseract.TessBaseAPISetImage2(self.api, pix_handle)
            else:
                if not data:
                    return OcrTextResult("", None)
                if not raw_ocr_image_size_is_supported(
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                ):
                    return OcrTextResult("", None)
                buffer = self.raw_image_buffer(data, buffer_cache=buffer_cache)
                self.tesseract.TessBaseAPISetImage(
                    self.api,
                    ctypes.cast(buffer, ctypes.c_void_p),
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                )
            if self.has_set_source_resolution and resolution > 0:
                self.tesseract.TessBaseAPISetSourceResolution(self.api, resolution)
            backend_rectangle = self.rectangle_for_image(image, rectangle)
            if backend_rectangle is not None and hasattr(self.tesseract, "TessBaseAPISetRectangle"):
                x0, y0, x1, y1 = backend_rectangle
                self.tesseract.TessBaseAPISetRectangle(
                    self.api,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                )
            return self.current_image_text_result(image, resolution)
        finally:
            if pix and self.leptonica is not None:
                self.leptonica.pixDestroy(ctypes.byref(pix_handle))
            if self.has_clear_adaptive_classifier:
                self.tesseract.TessBaseAPIClearAdaptiveClassifier(self.api)
            if self.has_clear:
                self.tesseract.TessBaseAPIClear(self.api)

    def current_image_text_result(
        self,
        image: OcrImage,
        resolution: int,
    ) -> OcrTextResult:
        iterator = None
        try:
            rows_by_level: dict[int, list[dict[str, Any]]] = {}
            if self.supports_result_iterator_rows():
                if self.tesseract.TessBaseAPIRecognize(self.api, None) < 0:
                    return OcrTextResult("", None)
                iterator = self.tesseract.TessBaseAPIGetIterator(self.api)
                if iterator:
                    rows_by_level = self.result_iterator_rows_by_level(
                        iterator,
                        (TESSERACT_RIL_TEXTLINE, TESSERACT_RIL_WORD),
                    )
            text, confidence = self.current_native_text_result()
            if not text:
                return OcrTextResult("", confidence)
            line_rows = tuple(rows_by_level.get(TESSERACT_RIL_TEXTLINE, ()))
            word_rows = tuple(rows_by_level.get(TESSERACT_RIL_WORD, ()))
            return OcrTextResult(
                text,
                confidence,
                line_rows=line_rows,
                word_rows=word_rows,
                observations=ocr_observations_from_rows(
                    [*line_rows, *word_rows],
                    source=image.source,
                    image_width=image.width,
                    image_height=image.height,
                    image_resolution=resolution,
                ),
            )
        finally:
            if iterator:
                self.tesseract.TessResultIteratorDelete(iterator)

    def current_native_text_result(self) -> tuple[str, int | None]:
        text_ptr = self.tesseract.TessBaseAPIGetUTF8Text(self.api)
        confidence = self.mean_text_confidence()
        if not text_ptr:
            return "", confidence
        try:
            text = ctypes.string_at(text_ptr).decode("utf-8", "replace").strip()
        finally:
            self.tesseract.TessDeleteText(text_ptr)
        return text, confidence

    def recognize_image_iterator_layout(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None,
        rectangle: tuple[int, int, int, int] | None,
        levels: tuple[int, ...],
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> tuple[dict[int, list[dict[str, Any]]], str, int | None]:
        data = image.data
        width = image.width
        height = image.height
        bytes_per_pixel = image.bytes_per_pixel
        bytes_per_line = image.bytes_per_line
        pix = None
        pix_handle = ctypes.c_void_p()
        iterator = None
        try:
            self.tesseract.TessBaseAPISetPageSegMode(self.api, psm)
            self.apply_variables(variables)
            if self.should_use_pix(image):
                pix = self.image_to_pix(image, buffer_cache=buffer_cache)
            if pix:
                pix_handle = ctypes.c_void_p(pix)
                self.tesseract.TessBaseAPISetImage2(self.api, pix_handle)
            else:
                if not data:
                    return {}, "", None
                if not raw_ocr_image_size_is_supported(
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                ):
                    return {}, "", None
                buffer = self.raw_image_buffer(data, buffer_cache=buffer_cache)
                self.tesseract.TessBaseAPISetImage(
                    self.api,
                    ctypes.cast(buffer, ctypes.c_void_p),
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                )
            if self.has_set_source_resolution and resolution > 0:
                self.tesseract.TessBaseAPISetSourceResolution(self.api, resolution)
            backend_rectangle = self.rectangle_for_image(image, rectangle)
            if backend_rectangle is not None and hasattr(self.tesseract, "TessBaseAPISetRectangle"):
                x0, y0, x1, y1 = backend_rectangle
                self.tesseract.TessBaseAPISetRectangle(
                    self.api,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                )
            if self.tesseract.TessBaseAPIRecognize(self.api, None) < 0:
                return {}, "", None
            iterator = self.tesseract.TessBaseAPIGetIterator(self.api)
            rows = self.result_iterator_rows_by_level(iterator, levels) if iterator else {}
            text, confidence = self.current_native_text_result()
            return rows, text, confidence
        finally:
            if iterator:
                self.tesseract.TessResultIteratorDelete(iterator)
            if pix and self.leptonica is not None:
                self.leptonica.pixDestroy(ctypes.byref(pix_handle))
            if self.has_clear_adaptive_classifier:
                self.tesseract.TessBaseAPIClearAdaptiveClassifier(self.api)
            if self.has_clear:
                self.tesseract.TessBaseAPIClear(self.api)

    def recognize_image_word_rows(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None,
        rectangle: tuple[int, int, int, int] | None,
    ) -> list[dict[str, Any]]:
        return self.recognize_image_iterator_rows(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=rectangle,
            level=TESSERACT_RIL_WORD,
        )

    def recognize_image_iterator_rows(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None,
        rectangle: tuple[int, int, int, int] | None,
        level: int,
    ) -> list[dict[str, Any]]:
        rows_by_level = self.recognize_image_iterator_rows_by_level(
            image,
            psm,
            resolution,
            variables=variables,
            rectangle=rectangle,
            levels=(level,),
        )
        return rows_by_level.get(level, [])

    def recognize_image_iterator_rows_by_level(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        variables: Mapping[str, str | int | float | bool] | None,
        rectangle: tuple[int, int, int, int] | None,
        levels: tuple[int, ...],
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> dict[int, list[dict[str, Any]]]:
        data = image.data
        width = image.width
        height = image.height
        bytes_per_pixel = image.bytes_per_pixel
        bytes_per_line = image.bytes_per_line
        pix = None
        pix_handle = ctypes.c_void_p()
        iterator = None
        iterator_copy = None
        try:
            self.tesseract.TessBaseAPISetPageSegMode(self.api, psm)
            self.apply_variables(variables)
            if self.should_use_pix(image):
                pix = self.image_to_pix(image, buffer_cache=buffer_cache)
            if pix:
                pix_handle = ctypes.c_void_p(pix)
                self.tesseract.TessBaseAPISetImage2(self.api, pix_handle)
            else:
                if not data:
                    return {}
                if not raw_ocr_image_size_is_supported(
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                ):
                    return {}
                buffer = self.raw_image_buffer(data, buffer_cache=buffer_cache)
                self.tesseract.TessBaseAPISetImage(
                    self.api,
                    ctypes.cast(buffer, ctypes.c_void_p),
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                )
            if self.has_set_source_resolution and resolution > 0:
                self.tesseract.TessBaseAPISetSourceResolution(self.api, resolution)
            backend_rectangle = self.rectangle_for_image(image, rectangle)
            if backend_rectangle is not None and hasattr(self.tesseract, "TessBaseAPISetRectangle"):
                x0, y0, x1, y1 = backend_rectangle
                self.tesseract.TessBaseAPISetRectangle(
                    self.api,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                )
            if self.tesseract.TessBaseAPIRecognize(self.api, None) < 0:
                return {}
            iterator = self.tesseract.TessBaseAPIGetIterator(self.api)
            if not iterator:
                return {}
            return self.result_iterator_rows_by_level(iterator, levels)
        finally:
            if iterator_copy:
                self.tesseract.TessResultIteratorDelete(iterator_copy)
            if iterator:
                self.tesseract.TessResultIteratorDelete(iterator)
            if pix and self.leptonica is not None:
                self.leptonica.pixDestroy(ctypes.byref(pix_handle))
            if self.has_clear_adaptive_classifier:
                self.tesseract.TessBaseAPIClearAdaptiveClassifier(self.api)
            if self.has_clear:
                self.tesseract.TessBaseAPIClear(self.api)

    def recognize_image_component_boxes(
        self,
        image: OcrImage,
        psm: int,
        resolution: int,
        *,
        level: int,
        text_only: bool,
        variables: Mapping[str, str | int | float | bool] | None,
        rectangle: tuple[int, int, int, int] | None,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> list[OcrComponentBox]:
        data = image.data
        width = image.width
        height = image.height
        bytes_per_pixel = image.bytes_per_pixel
        bytes_per_line = image.bytes_per_line
        pix = None
        pix_handle = ctypes.c_void_p()
        boxa_handle = ctypes.c_void_p()
        pixa_handle = ctypes.c_void_p()
        try:
            self.tesseract.TessBaseAPISetPageSegMode(self.api, psm)
            self.apply_variables(variables)
            if self.should_use_pix(image):
                pix = self.image_to_pix(image, buffer_cache=buffer_cache)
            if pix:
                pix_handle = ctypes.c_void_p(pix)
                self.tesseract.TessBaseAPISetImage2(self.api, pix_handle)
            else:
                if not data:
                    return []
                if not raw_ocr_image_size_is_supported(
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                ):
                    return []
                buffer = self.raw_image_buffer(data, buffer_cache=buffer_cache)
                self.tesseract.TessBaseAPISetImage(
                    self.api,
                    ctypes.cast(buffer, ctypes.c_void_p),
                    width,
                    height,
                    bytes_per_pixel,
                    bytes_per_line,
                )
            if self.has_set_source_resolution and resolution > 0:
                self.tesseract.TessBaseAPISetSourceResolution(self.api, resolution)
            backend_rectangle = self.rectangle_for_image(image, rectangle)
            if backend_rectangle is not None and hasattr(self.tesseract, "TessBaseAPISetRectangle"):
                x0, y0, x1, y1 = backend_rectangle
                self.tesseract.TessBaseAPISetRectangle(
                    self.api,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                )
            if hasattr(self.tesseract, "TessBaseAPIGetComponentImages"):
                boxa = self.tesseract.TessBaseAPIGetComponentImages(
                    self.api,
                    level,
                    bool(text_only),
                    ctypes.byref(pixa_handle),
                    None,
                )
            elif hasattr(self.tesseract, "TessBaseAPIGetComponentImages1"):
                boxa = self.tesseract.TessBaseAPIGetComponentImages1(
                    self.api,
                    level,
                    bool(text_only),
                    True,
                    0,
                    ctypes.byref(pixa_handle),
                    None,
                    None,
                )
            else:
                return []
            if not boxa:
                return []
            boxa_handle = ctypes.c_void_p(boxa)
            return self.component_boxes_from_boxa(boxa_handle, level)
        finally:
            if pixa_handle.value and self.leptonica is not None:
                self.leptonica.pixaDestroy(ctypes.byref(pixa_handle))
            if boxa_handle.value and self.leptonica is not None:
                self.leptonica.boxaDestroy(ctypes.byref(boxa_handle))
            if pix and self.leptonica is not None:
                self.leptonica.pixDestroy(ctypes.byref(pix_handle))
            if self.has_clear_adaptive_classifier:
                self.tesseract.TessBaseAPIClearAdaptiveClassifier(self.api)
            if self.has_clear:
                self.tesseract.TessBaseAPIClear(self.api)

    def component_boxes_from_boxa(self, boxa: ctypes.c_void_p, level: int) -> list[OcrComponentBox]:
        if self.leptonica is None:
            return []
        count = int(self.leptonica.boxaGetCount(boxa))
        boxes: list[OcrComponentBox] = []
        for index in range(max(0, count)):
            left = ctypes.c_int()
            top = ctypes.c_int()
            width = ctypes.c_int()
            height = ctypes.c_int()
            if self.leptonica.boxaGetBoxGeometry(
                boxa,
                index,
                ctypes.byref(left),
                ctypes.byref(top),
                ctypes.byref(width),
                ctypes.byref(height),
            ):
                continue
            if width.value <= 0 or height.value <= 0:
                continue
            boxes.append(
                OcrComponentBox(
                    level=level,
                    index=index,
                    left=int(left.value),
                    top=int(top.value),
                    width=int(width.value),
                    height=int(height.value),
                )
            )
        return boxes

    def result_iterator_rows_by_level(
        self,
        iterator: int,
        levels: tuple[int, ...],
    ) -> dict[int, list[dict[str, Any]]]:
        rows_by_level: dict[int, list[dict[str, Any]]] = {}
        if len(levels) == 1:
            level = levels[0]
            rows_by_level[level] = self.result_iterator_rows(iterator, level)
            return rows_by_level
        if not hasattr(self.tesseract, "TessResultIteratorCopy"):
            return rows_by_level
        for level in levels:
            iterator_copy = self.tesseract.TessResultIteratorCopy(iterator)
            if not iterator_copy:
                rows_by_level[level] = []
                continue
            try:
                rows_by_level[level] = self.result_iterator_rows(iterator_copy, level)
            finally:
                self.tesseract.TessResultIteratorDelete(iterator_copy)
        return rows_by_level

    def result_iterator_word_rows(self, iterator: int) -> list[dict[str, Any]]:
        return self.result_iterator_rows(iterator, TESSERACT_RIL_WORD)

    def result_iterator_rows(self, iterator: int, level: int) -> list[dict[str, Any]]:
        if level not in {
            TESSERACT_RIL_TEXTLINE,
            TESSERACT_RIL_WORD,
            TESSERACT_RIL_SYMBOL,
        }:
            return []
        rows: list[dict[str, Any]] = []
        block_num = 0
        par_num = 0
        line_num = 0
        word_num = 0
        symbol_num = 0
        while True:
            if self.tesseract.TessPageIteratorIsAtBeginningOf(iterator, TESSERACT_RIL_BLOCK):
                block_num += 1
                par_num = 0
                line_num = 0
                word_num = 0
                symbol_num = 0
            if self.tesseract.TessPageIteratorIsAtBeginningOf(iterator, TESSERACT_RIL_PARA):
                par_num += 1
                line_num = 0
                word_num = 0
                symbol_num = 0
            if self.tesseract.TessPageIteratorIsAtBeginningOf(iterator, TESSERACT_RIL_TEXTLINE):
                line_num += 1
                word_num = 0
                symbol_num = 0
            if level == TESSERACT_RIL_WORD:
                word_num += 1
            elif level == TESSERACT_RIL_SYMBOL:
                if self.tesseract.TessPageIteratorIsAtBeginningOf(iterator, TESSERACT_RIL_WORD):
                    word_num += 1
                    symbol_num = 0
                symbol_num += 1
            row = self.result_iterator_row(
                iterator,
                level=level,
                block_num=max(1, block_num),
                par_num=max(1, par_num),
                line_num=max(1, line_num),
                word_num=word_num,
                symbol_num=symbol_num,
            )
            if row is not None:
                rows.append(row)
            if not self.tesseract.TessResultIteratorNext(iterator, level):
                break
        return rows

    def result_iterator_word_row(
        self,
        iterator: int,
        *,
        block_num: int,
        par_num: int,
        line_num: int,
        word_num: int,
    ) -> dict[str, Any] | None:
        return self.result_iterator_row(
            iterator,
            level=TESSERACT_RIL_WORD,
            block_num=block_num,
            par_num=par_num,
            line_num=line_num,
            word_num=word_num,
            symbol_num=0,
        )

    def result_iterator_row(
        self,
        iterator: int,
        *,
        level: int,
        block_num: int,
        par_num: int,
        line_num: int,
        word_num: int,
        symbol_num: int,
    ) -> dict[str, Any] | None:
        text_ptr = self.tesseract.TessResultIteratorGetUTF8Text(iterator, level)
        if not text_ptr:
            return None
        try:
            text = ctypes.string_at(text_ptr).decode("utf-8", "replace").strip()
        finally:
            self.tesseract.TessDeleteText(text_ptr)
        if not text:
            return None
        left = ctypes.c_int()
        top = ctypes.c_int()
        right = ctypes.c_int()
        bottom = ctypes.c_int()
        if not self.tesseract.TessPageIteratorBoundingBox(
            iterator,
            level,
            ctypes.byref(left),
            ctypes.byref(top),
            ctypes.byref(right),
            ctypes.byref(bottom),
        ):
            return None
        width = max(0, int(right.value) - int(left.value))
        height = max(0, int(bottom.value) - int(top.value))
        if width <= 0 or height <= 0:
            return None
        confidence = int(round(float(self.tesseract.TessResultIteratorConfidence(iterator, level))))
        row: dict[str, Any] = {
            "level": level,
            "page_num": 1,
            "block_num": block_num,
            "par_num": par_num,
            "line_num": line_num,
            "word_num": word_num,
            "symbol_num": symbol_num,
            "left": int(left.value),
            "top": int(top.value),
            "width": width,
            "height": height,
            "conf": max(0, min(100, confidence)),
            "text": text,
        }
        baseline = self.result_iterator_baseline(iterator, level)
        if baseline is not None:
            row["baseline"] = baseline
        choices = self.result_iterator_choices(iterator, level)
        if choices:
            row["choices"] = choices
        return row

    def result_iterator_baseline(
        self, iterator: int, level: int
    ) -> tuple[int, int, int, int] | None:
        if not hasattr(self.tesseract, "TessPageIteratorBaseline"):
            return None
        x1 = ctypes.c_int()
        y1 = ctypes.c_int()
        x2 = ctypes.c_int()
        y2 = ctypes.c_int()
        if not self.tesseract.TessPageIteratorBaseline(
            iterator,
            level,
            ctypes.byref(x1),
            ctypes.byref(y1),
            ctypes.byref(x2),
            ctypes.byref(y2),
        ):
            return None
        return (int(x1.value), int(y1.value), int(x2.value), int(y2.value))

    def result_iterator_choices(self, iterator: int, level: int) -> tuple[OcrTextChoice, ...]:
        if level not in {TESSERACT_RIL_WORD, TESSERACT_RIL_SYMBOL}:
            return ()
        if not all(
            hasattr(self.tesseract, name)
            for name in (
                "TessResultIteratorGetChoiceIterator",
                "TessChoiceIteratorDelete",
                "TessChoiceIteratorNext",
                "TessChoiceIteratorGetUTF8Text",
                "TessChoiceIteratorConfidence",
            )
        ):
            return ()
        choice_iterator = self.tesseract.TessResultIteratorGetChoiceIterator(iterator)
        if not choice_iterator:
            return ()
        choices: list[OcrTextChoice] = []
        try:
            while True:
                text_ptr = self.tesseract.TessChoiceIteratorGetUTF8Text(choice_iterator)
                if text_ptr:
                    text = ctypes.string_at(text_ptr).decode("utf-8", "replace").strip()
                    if text:
                        confidence = int(
                            round(
                                float(self.tesseract.TessChoiceIteratorConfidence(choice_iterator))
                            )
                        )
                        choices.append(
                            OcrTextChoice(
                                text=text,
                                confidence=max(0, min(100, confidence)),
                            )
                        )
                if not self.tesseract.TessChoiceIteratorNext(choice_iterator):
                    break
        finally:
            self.tesseract.TessChoiceIteratorDelete(choice_iterator)
        return tuple(choices)

    def supports_result_iterator_rows(self) -> bool:
        return all(
            hasattr(self.tesseract, name)
            for name in (
                "TessBaseAPIRecognize",
                "TessBaseAPIGetIterator",
                "TessResultIteratorDelete",
                "TessResultIteratorNext",
                "TessResultIteratorGetUTF8Text",
                "TessResultIteratorConfidence",
                "TessPageIteratorBoundingBox",
                "TessPageIteratorIsAtBeginningOf",
            )
        )

    def supports_component_boxes(self) -> bool:
        if self.leptonica is None:
            return False
        if not hasattr(self.tesseract, "TessBaseAPIGetComponentImages"):
            return False
        return all(
            hasattr(self.leptonica, name)
            for name in (
                "boxaGetCount",
                "boxaGetBoxGeometry",
                "boxaDestroy",
                "pixaDestroy",
            )
        )

    def apply_variables(
        self,
        variables: Mapping[str, str | int | float | bool] | None,
    ) -> None:
        effective: dict[str, str | int | float | bool] = dict(TESSERACT_DEFAULT_VARIABLES)
        if variables:
            effective.update(variables)
        for name, value in effective.items():
            self.tesseract.TessBaseAPISetVariable(
                self.api,
                name.encode("utf-8"),
                self.format_variable_value(value),
            )

    @staticmethod
    def format_variable_value(value: str | int | float | bool) -> bytes:
        if isinstance(value, bool):
            return b"1" if value else b"0"
        return str(value).encode("utf-8")

    def mean_text_confidence(self) -> int | None:
        if not hasattr(self.tesseract, "TessBaseAPIMeanTextConf"):
            return None
        confidence = int(self.tesseract.TessBaseAPIMeanTextConf(self.api))
        return confidence if confidence >= 0 else None

    @staticmethod
    def clamp_rectangle(
        rectangle: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = rectangle
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(x0, min(width, x1))
        y1 = max(y0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    @classmethod
    def rectangle_for_image(
        cls,
        image: OcrImage,
        rectangle: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if rectangle is None:
            return None
        source_width = max(1, image.width)
        source_height = max(1, image.height)
        target_width = image.target_width or source_width
        target_height = image.target_height or source_height
        clamped = cls.clamp_rectangle(rectangle, source_width, source_height)
        if clamped is None:
            return None
        x0, y0, x1, y1 = clamped
        mapped = (
            round(x0 * target_width / source_width),
            round(y0 * target_height / source_height),
            round(x1 * target_width / source_width),
            round(y1 * target_height / source_height),
        )
        return cls.clamp_rectangle(mapped, target_width, target_height)

    def image_to_pix(
        self,
        image: OcrImage,
        *,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> int | None:
        pix = self.source_pix_from_image(image, buffer_cache=buffer_cache)
        if not pix:
            return None
        scaled = self.scale_source_pix(pix, image)
        if scaled and scaled != pix and self.leptonica is not None:
            pix_handle = ctypes.c_void_p(pix)
            self.leptonica.pixDestroy(ctypes.byref(pix_handle))
            return scaled
        return scaled if scaled else pix

    def source_pix_from_image(
        self,
        image: OcrImage,
        *,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> int | None:
        if self.leptonica is None:
            return None
        encoded = image.encoded
        if not leptonica_pix_size_is_supported(image.width, image.height):
            return None
        if encoded is None:
            encoded = rgba_image_to_bmp(image)
        if encoded is None:
            return None
        buffer = self.raw_image_buffer(encoded, buffer_cache=buffer_cache)
        pix = self.leptonica.pixReadMem(
            ctypes.cast(buffer, ctypes.c_void_p),
            len(encoded),
        )
        rotation = image.clockwise_quarter_turns % 4
        if pix and rotation:
            if not self.has_pix_rotate_orth:
                pix_handle = ctypes.c_void_p(pix)
                self.leptonica.pixDestroy(ctypes.byref(pix_handle))
                return None
            pix_handle = ctypes.c_void_p(pix)
            rotated = self.leptonica.pixRotateOrth(pix_handle, rotation)
            if rotated:
                self.leptonica.pixDestroy(ctypes.byref(pix_handle))
                pix = rotated
        return int(pix) if pix else None

    def raw_image_buffer(
        self,
        data: bytes | bytearray | memoryview,
        *,
        buffer_cache: dict[int, tuple[object, Any]] | None = None,
    ) -> Any:
        if type(data) is bytes:
            cache_key = id(data)
            if buffer_cache is not None:
                cached = buffer_cache.get(cache_key)
                if cached is not None and cached[0] is data:
                    return cached[1]
            buffer = ctypes.create_string_buffer(data)
            if buffer_cache is not None:
                buffer_cache[cache_key] = (data, buffer)
            return buffer
        raw_data = bytes(data)
        cache_key = id(raw_data)
        if buffer_cache is not None:
            cached = buffer_cache.get(cache_key)
            if cached is not None and cached[0] is raw_data:
                return cached[1]
        buffer = ctypes.create_string_buffer(raw_data)
        if buffer_cache is not None:
            buffer_cache[cache_key] = (raw_data, buffer)
        return buffer

    def scale_source_pix(self, source_pix: int, image: OcrImage) -> int | None:
        if not source_pix:
            return None
        if not self.should_scale_pix(image):
            return source_pix
        if self.leptonica is None:
            return None
        if not leptonica_pix_scale_is_supported(
            image.width,
            image.height,
            image.target_width,
            image.target_height,
        ):
            return None
        pix_handle = ctypes.c_void_p(source_pix)
        scaled = self.leptonica.pixScaleToSize(
            pix_handle,
            image.target_width,
            image.target_height,
        )
        return int(scaled) if scaled else None

    def deskew_pix(
        self,
        pix: int,
        *,
        source: str,
        width: int,
        height: int,
    ) -> tuple[int, OcrDeskewInfo]:
        unavailable = ocr_deskew.deskew_info(
            source=source,
            angle_degrees=None,
            confidence=None,
            applied=False,
            reason="unavailable",
            image_width=width,
            image_height=height,
        )
        if (
            not pix
            or self.leptonica is None
            or not self.has_pix_convert_to_1
            or not self.has_pix_find_skew
            or not self.has_pix_rotate
        ):
            return pix, unavailable

        try:
            binary_pix = self.leptonica.pixConvertTo1(
                ctypes.c_void_p(pix),
                ocr_deskew.OCR_DESKEW_BINARY_THRESHOLD,
            )
        except (AttributeError, OSError, ValueError):
            return pix, unavailable
        if not binary_pix:
            return pix, ocr_deskew.deskew_info(
                source=source,
                angle_degrees=None,
                confidence=None,
                applied=False,
                reason="binarization_failed",
                image_width=width,
                image_height=height,
            )

        angle = ctypes.c_float()
        confidence = ctypes.c_float()
        binary_handle = ctypes.c_void_p(binary_pix)
        try:
            try:
                status = self.leptonica.pixFindSkew(
                    binary_handle,
                    ctypes.byref(angle),
                    ctypes.byref(confidence),
                )
            except (AttributeError, OSError, ValueError):
                return pix, unavailable
        finally:
            self.leptonica.pixDestroy(ctypes.byref(binary_handle))

        angle_value = float(angle.value)
        confidence_value = float(confidence.value)
        if status != 0 or not math.isfinite(angle_value) or not math.isfinite(confidence_value):
            return pix, ocr_deskew.deskew_info(
                source=source,
                angle_degrees=None,
                confidence=None,
                applied=False,
                reason="invalid_measurement",
                image_width=width,
                image_height=height,
            )
        if confidence_value < ocr_deskew.OCR_DESKEW_CONFIDENCE_THRESHOLD:
            return pix, ocr_deskew.deskew_info(
                source=source,
                angle_degrees=angle_value,
                confidence=confidence_value,
                applied=False,
                reason="low_confidence",
                image_width=width,
                image_height=height,
            )
        if abs(angle_value) < ocr_deskew.OCR_DESKEW_MIN_ANGLE_DEGREES:
            return pix, ocr_deskew.deskew_info(
                source=source,
                angle_degrees=angle_value,
                confidence=confidence_value,
                applied=False,
                reason="below_minimum_angle",
                image_width=width,
                image_height=height,
            )
        if abs(angle_value) > ocr_deskew.OCR_DESKEW_MAX_ANGLE_DEGREES:
            return pix, ocr_deskew.deskew_info(
                source=source,
                angle_degrees=angle_value,
                confidence=confidence_value,
                applied=False,
                reason="angle_out_of_range",
                image_width=width,
                image_height=height,
            )

        try:
            rotated = self.leptonica.pixRotate(
                ctypes.c_void_p(pix),
                ctypes.c_float(math.radians(angle_value)),
                1,  # L_ROTATE_AREA_MAP
                1,  # L_BRING_IN_WHITE
                0,
                0,
            )
        except (AttributeError, OSError, ValueError):
            rotated = None
        if not rotated:
            return pix, ocr_deskew.deskew_info(
                source=source,
                angle_degrees=angle_value,
                confidence=confidence_value,
                applied=False,
                reason="rotation_failed",
                image_width=width,
                image_height=height,
            )
        return int(rotated), ocr_deskew.deskew_info(
            source=source,
            angle_degrees=angle_value,
            confidence=confidence_value,
            applied=True,
            reason="applied",
            image_width=width,
            image_height=height,
        )

    def clip_scale_source_pix(
        self,
        source_pix: int,
        *,
        source_width: int,
        source_height: int,
        rectangle: tuple[int, int, int, int],
        target_width: int | None = None,
        target_height: int | None = None,
        clockwise_quarter_turns: int = 0,
    ) -> int | None:
        if (
            not source_pix
            or self.leptonica is None
            or not self.has_pix_clip_rectangle
            or not self.has_box_create
            or not self.has_box_destroy
        ):
            return None
        # The decoded PIX is authoritative. Encoded image metadata can differ
        # after rotation or renderer scaling, and passing source-space bounds
        # directly to pixClipRectangle produces Leptonica's
        # ``box outside rectangle`` message.
        pix_width = source_width
        pix_height = source_height
        if hasattr(self.leptonica, "pixGetWidth") and hasattr(self.leptonica, "pixGetHeight"):
            pix_width = int(self.leptonica.pixGetWidth(ctypes.c_void_p(source_pix)))
            pix_height = int(self.leptonica.pixGetHeight(ctypes.c_void_p(source_pix)))
            if pix_width <= 0 or pix_height <= 0:
                return None
        x0, y0, x1, y1 = rectangle
        rotation = clockwise_quarter_turns % 4
        if rotation == 1:
            rectangle = (
                source_height - y1,
                x0,
                source_height - y0,
                x1,
            )
            source_width, source_height = source_height, source_width
        elif rotation == 2:
            rectangle = (
                source_width - x1,
                source_height - y1,
                source_width - x0,
                source_height - y0,
            )
        elif rotation == 3:
            rectangle = (
                y0,
                source_width - x1,
                y1,
                source_width - x0,
            )
            source_width, source_height = source_height, source_width
        if pix_width != source_width or pix_height != source_height:
            x0, y0, x1, y1 = rectangle
            rectangle = (
                round(x0 * pix_width / source_width),
                round(y0 * pix_height / source_height),
                round(x1 * pix_width / source_width),
                round(y1 * pix_height / source_height),
            )
        clamped = self.clamp_rectangle(rectangle, pix_width, pix_height)
        if clamped is None:
            return None
        x0, y0, x1, y1 = clamped
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            return None
        box = self.leptonica.boxCreate(x0, y0, width, height)
        if not box:
            return None
        box_handle = ctypes.c_void_p(box)
        try:
            clipped = self.leptonica.pixClipRectangle(
                ctypes.c_void_p(source_pix),
                box_handle,
                None,
            )
        finally:
            self.leptonica.boxDestroy(ctypes.byref(box_handle))
        if not clipped:
            return None
        clipped_pix = int(clipped)
        if target_width is None or target_height is None:
            return clipped_pix
        if target_width <= 0 or target_height <= 0:
            clipped_handle = ctypes.c_void_p(clipped_pix)
            self.leptonica.pixDestroy(ctypes.byref(clipped_handle))
            return None
        if target_width == width and target_height == height:
            return clipped_pix
        if not self.has_pix_scale_to_size:
            clipped_handle = ctypes.c_void_p(clipped_pix)
            self.leptonica.pixDestroy(ctypes.byref(clipped_handle))
            return None
        if not leptonica_pix_scale_is_supported(
            width,
            height,
            target_width,
            target_height,
        ):
            clipped_handle = ctypes.c_void_p(clipped_pix)
            self.leptonica.pixDestroy(ctypes.byref(clipped_handle))
            return None
        scaled = self.leptonica.pixScaleToSize(
            ctypes.c_void_p(clipped_pix),
            target_width,
            target_height,
        )
        clipped_handle = ctypes.c_void_p(clipped_pix)
        self.leptonica.pixDestroy(ctypes.byref(clipped_handle))
        return int(scaled) if scaled else None

    def should_scale_pix(self, image: OcrImage) -> bool:
        if not self.has_pix_scale_to_size:
            return False
        if image.target_width is None or image.target_height is None:
            return False
        if image.target_width <= 0 or image.target_height <= 0:
            return False
        if not leptonica_pix_size_is_supported(
            image.target_width,
            image.target_height,
        ):
            return False
        target_pixels = image.target_width * image.target_height
        source_pixels = image.width * image.height
        if target_pixels <= 0 or source_pixels <= 0:
            return False
        return image.target_width != image.width or image.target_height != image.height

    def should_use_pix(self, image: OcrImage) -> bool:
        if getattr(self, "leptonica", None) is None or not self.has_set_image2:
            return False
        if image.clockwise_quarter_turns % 4 and not self.has_pix_rotate_orth:
            return False
        if image.encoded is not None:
            return leptonica_pix_size_is_supported(image.width, image.height)
        if (
            ocr_deskew.full_page_ocr_image_should_be_deskewed(image)
            and image.bytes_per_pixel in {1, 3, 4}
            and raw_ocr_image_size_is_supported(
                image.width,
                image.height,
                image.bytes_per_pixel,
                image.bytes_per_line,
            )
        ):
            return True
        return self.should_scale_pix(image)

    def supports_encoded_pix(self) -> bool:
        return getattr(self, "leptonica", None) is not None and self.has_set_image2

    def __del__(self) -> None:
        api = getattr(self, "api", None)
        if api:
            self.tesseract.TessBaseAPIEnd(api)
            self.tesseract.TessBaseAPIDelete(api)
            self.api = None


def rgba_image_to_bmp(image: OcrImage) -> bytes | None:
    data = image.data
    width = image.width
    height = image.height
    bytes_per_pixel = image.bytes_per_pixel
    bytes_per_line = image.bytes_per_line
    if width <= 0 or height <= 0 or bytes_per_pixel not in {1, 3, 4} or bytes_per_line <= 0:
        return None
    if not leptonica_pix_size_is_supported(width, height):
        return None
    if len(data) < (height - 1) * bytes_per_line + width * bytes_per_pixel:
        return None
    row_size = ((width * 3 + 3) // 4) * 4
    pixel_bytes = row_size * height
    file_size = 14 + 40 + pixel_bytes
    if pixel_bytes > BMP_MAX_BYTES or file_size > BMP_MAX_BYTES:
        return None
    header = bytearray()
    header.extend(b"BM")
    header.extend(file_size.to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((54).to_bytes(4, "little"))
    header.extend((40).to_bytes(4, "little"))
    header.extend(width.to_bytes(4, "little", signed=True))
    header.extend(height.to_bytes(4, "little", signed=True))
    header.extend((1).to_bytes(2, "little"))
    header.extend((24).to_bytes(2, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend(pixel_bytes.to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    header.extend((0).to_bytes(4, "little"))
    rows = bytearray(pixel_bytes)
    if bytes_per_pixel == 4 and bytes_per_line == width * 4:
        for y in range(height):
            source_row = (height - 1 - y) * bytes_per_line
            source = source_row
            target = y * row_size
            end = source + bytes_per_line
            while source < end:
                rows[target] = data[source + 2]
                rows[target + 1] = data[source + 1]
                rows[target + 2] = data[source]
                source += 4
                target += 3
    elif bytes_per_pixel == 1:
        for y in range(height):
            source_row = (height - 1 - y) * bytes_per_line
            source = source_row
            target = y * row_size
            for _ in range(width):
                value = data[source]
                rows[target] = value
                rows[target + 1] = value
                rows[target + 2] = value
                source += 1
                target += 3
    else:
        for y in range(height):
            source_row = (height - 1 - y) * bytes_per_line
            source = source_row
            target = y * row_size
            for _ in range(width):
                rows[target] = data[source + 2]
                rows[target + 1] = data[source + 1]
                rows[target + 2] = data[source]
                source += bytes_per_pixel
                target += 3
    return bytes(header + rows)


def leptonica_pix_scale_is_supported(
    source_width: int,
    source_height: int,
    target_width: int | None,
    target_height: int | None,
) -> bool:
    if (
        source_width <= 1
        or source_height <= 1
        or target_width is None
        or target_height is None
        or target_width <= 1
        or target_height <= 1
    ):
        return False
    return leptonica_pix_size_is_supported(
        source_width, source_height
    ) and leptonica_pix_size_is_supported(target_width, target_height)
