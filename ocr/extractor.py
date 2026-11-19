import gc
import importlib.util
import logging
import math
import os
import threading
from typing import Any, Dict, List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# These limits protect the detector before RapidOCR decodes or resizes an
# image. They intentionally allow regular landscape, portrait, 4:3 and 16:9
# classroom images while rejecting panoramas with unsafe detector geometry.
OCR_MAX_IMAGE_PIXELS = 8_000_000
OCR_MAX_IMAGE_WIDTH = 4_000
OCR_MAX_IMAGE_HEIGHT = 4_000
OCR_MAX_ASPECT_RATIO = 3.0
OCR_MAX_DETECTOR_TENSOR_BYTES = 48 * 1024 * 1024
OCR_MAX_DETECTOR_WIDTH = 2_304
OCR_MAX_DETECTOR_HEIGHT = 2_304
# RapidOCR's detector normalization creates temporary float32 arrays. Budget
# up to three detector tensors while normalizing (input, scaled, normalized).
OCR_MAX_DETECTOR_WORKING_BYTES = 3 * OCR_MAX_DETECTOR_TENSOR_BYTES
# The upstream default (2,000 px) produces about 36 MiB for the float32
# detector tensor on an 8 MP 4:3 image. A 1,280 px cap limits detector
# activations and normalization buffers on the 512 MB worker.
RAPIDOCR_MAX_SIDE_LEN = 1_280
RAPIDOCR_MIN_SIDE_LEN = 30
RAPIDOCR_MIN_HEIGHT = 30
RAPIDOCR_WIDTH_HEIGHT_RATIO = 8.0
RAPIDOCR_DET_LIMIT_SIDE_LEN = 736
RAPIDOCR_DET_LIMIT_TYPE = "min"
OCR_GEOMETRY_REVIEW_MESSAGE = (
    "This image is too wide, too tall, or too large for safe text extraction. "
    "Retake it with the board upright and closer in frame, or crop the image "
    "around the writing. No study notes were generated from this image."
)

# RapidOCR uses ONNX Runtime on CPU. The model is lazy-loaded once per process.
_ocr_engine = None
_ocr_lock = threading.Lock()
_inference_lock = threading.Lock()


class OCREngineUnavailable(RuntimeError):
    """Raised when RapidOCR cannot initialize or complete inference."""


class OCRGeometryRejected(ValueError):
    """An image exceeds the memory-safe geometry supported by OCR."""

    def __init__(self, reason):
        self.reason = str(reason)
        self.user_message = OCR_GEOMETRY_REVIEW_MESSAGE
        super().__init__(self.reason)


def _round_to_32(value: int) -> int:
    """Match RapidOCR 1.4.4's nearest-multiple-of-32 resize rounding."""
    return int(round(int(value) / 32) * 32)


def _resize_dimensions(height: int, width: int, ratio: float):
    """Match RapidOCR's integer truncation followed by 32-pixel alignment."""
    return (
        _round_to_32(int(height * ratio)),
        _round_to_32(int(width * ratio)),
    )


def _active_resize_settings(height: int, width: int, engine=None):
    """Read the loaded RapidOCR resize settings without allocating image data.

    The defaults match the checked-in RapidOCR 1.4.4 package config. Once the
    singleton exists, inspect its live detector preprocessor so a package or
    config change cannot silently bypass the memory estimate.
    """
    if engine is None:
        return (
            RAPIDOCR_MAX_SIDE_LEN,
            RAPIDOCR_MIN_SIDE_LEN,
            RAPIDOCR_DET_LIMIT_SIDE_LEN,
            RAPIDOCR_DET_LIMIT_TYPE,
        )

    try:
        max_side_len = int(engine.max_side_len)
        min_side_len = int(engine.min_side_len)
        min_height = int(engine.min_height)
        width_height_ratio = float(engine.width_height_ratio)
        if min_height != RAPIDOCR_MIN_HEIGHT:
            raise ValueError("RapidOCR min_height differs from the verified configuration")
        if not math.isfinite(width_height_ratio) or (
            width_height_ratio != -1
            and width_height_ratio <= OCR_MAX_ASPECT_RATIO
        ):
            raise ValueError("RapidOCR may add unmodeled vertical padding")
        resized_height, resized_width = height, width
        if max(resized_height, resized_width) > max_side_len:
            ratio = max_side_len / float(max(resized_height, resized_width))
            resized_height, resized_width = _resize_dimensions(
                resized_height, resized_width, ratio
            )
        if min(resized_height, resized_width) < min_side_len:
            ratio = min_side_len / float(min(resized_height, resized_width))
            resized_height, resized_width = _resize_dimensions(
                resized_height, resized_width, ratio
            )

        detector = engine.text_det
        get_preprocess = detector.get_preprocess
        detector_preprocess = get_preprocess(max(resized_height, resized_width))
        return (
            max_side_len,
            min_side_len,
            int(detector_preprocess.limit_side_len),
            str(detector_preprocess.limit_type),
        )
    except Exception as exc:
        raise OCRGeometryRejected(
            "The active OCR resize configuration could not be verified."
        ) from exc


def estimate_ocr_detector_geometry(width: int, height: int, engine=None) -> dict:
    """Estimate RapidOCR's detector input shape and float32 working memory.

    This mirrors RapidOCR 1.4.4's resize rules, including its min-side detector
    upscaling and 32-pixel rounding. It only uses dimensions and integers; it
    never allocates an image or tensor.
    """
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OCRGeometryRejected("Image dimensions are invalid.") from exc
    if width <= 0 or height <= 0:
        raise OCRGeometryRejected("Image dimensions must be positive.")
    if width > OCR_MAX_IMAGE_WIDTH or height > OCR_MAX_IMAGE_HEIGHT:
        raise OCRGeometryRejected(
            f"Image dimensions {width}x{height} exceed the OCR width/height limits."
        )
    if width * height > OCR_MAX_IMAGE_PIXELS:
        raise OCRGeometryRejected(
            f"Image dimensions {width}x{height} exceed the OCR pixel limit."
        )
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > OCR_MAX_ASPECT_RATIO:
        raise OCRGeometryRejected(
            f"Image aspect ratio {aspect_ratio:.3f} exceeds the OCR limit."
        )

    max_side_len, min_side_len, det_limit_side_len, det_limit_type = (
        _active_resize_settings(height, width, engine)
    )
    resized_height, resized_width = height, width

    # RapidOCR main.preprocess(): reduce the longest side, then raise a very
    # short side to min_side_len. Both operations align dimensions to 32.
    if max(resized_height, resized_width) > max_side_len:
        ratio = max_side_len / float(max(resized_height, resized_width))
        resized_height, resized_width = _resize_dimensions(
            resized_height, resized_width, ratio
        )
    if min(resized_height, resized_width) < min_side_len:
        ratio = min_side_len / float(min(resized_height, resized_width))
        resized_height, resized_width = _resize_dimensions(
            resized_height, resized_width, ratio
        )

    if resized_height <= 0 or resized_width <= 0:
        raise OCRGeometryRejected(
            "RapidOCR would round an image dimension to zero."
        )

    # RapidOCR TextDetector.get_preprocess()/DetPreProcess.resize().
    if det_limit_type == "min":
        ratio = (
            det_limit_side_len / float(min(resized_height, resized_width))
            if min(resized_height, resized_width) < det_limit_side_len
            else 1.0
        )
    elif det_limit_type == "max":
        ratio = (
            det_limit_side_len / float(max(resized_height, resized_width))
            if max(resized_height, resized_width) > det_limit_side_len
            else 1.0
        )
    else:
        raise OCRGeometryRejected(
            f"Unsupported RapidOCR detector resize mode: {det_limit_type!r}."
        )

    detector_height, detector_width = _resize_dimensions(
        resized_height, resized_width, ratio
    )
    if detector_height <= 0 or detector_width <= 0:
        raise OCRGeometryRejected(
            "RapidOCR would create an invalid detector tensor shape."
        )

    tensor_bytes = detector_height * detector_width * 3 * 4
    working_bytes = tensor_bytes * 3
    if (
        detector_width > OCR_MAX_DETECTOR_WIDTH
        or detector_height > OCR_MAX_DETECTOR_HEIGHT
    ):
        raise OCRGeometryRejected(
            f"Estimated detector dimensions {detector_width}x{detector_height} exceed the OCR limit."
        )
    if tensor_bytes > OCR_MAX_DETECTOR_TENSOR_BYTES:
        raise OCRGeometryRejected(
            f"Estimated detector tensor is {tensor_bytes} bytes, over the OCR limit."
        )
    if working_bytes > OCR_MAX_DETECTOR_WORKING_BYTES:
        raise OCRGeometryRejected(
            f"Estimated detector working set is {working_bytes} bytes, over the OCR limit."
        )

    return {
        "source_width": width,
        "source_height": height,
        "detector_width": detector_width,
        "detector_height": detector_height,
        "detector_tensor_bytes": tensor_bytes,
        "detector_working_bytes": working_bytes,
        "aspect_ratio": aspect_ratio,
    }


def _elapsed_seconds(elapse) -> float:
    """Sum valid RapidOCR timing components without masking good OCR output."""
    components = elapse if isinstance(elapse, (list, tuple)) else (elapse,)
    total = 0.0
    for component in components:
        try:
            value = float(component)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value >= 0:
            total += value
    return total


def ocr_is_configured() -> bool:
    """Returns True if the rapidocr-onnxruntime package can be imported."""
    try:
        return importlib.util.find_spec("rapidocr_onnxruntime") is not None
    except Exception:
        return False


def ocr_is_initialized() -> bool:
    """Returns True if the RapidOCR singleton is already loaded in memory."""
    return _ocr_engine is not None


def ocr_is_operational() -> bool:
    """Returns True if a probe inference can be run (singleton warm)."""
    return _ocr_engine is not None


def get_reader():
    """Thread-safe lazy singleton for CPU-only RapidOCR ONNX inference."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine
        logger.info("[OCR] Initializing RapidOCR (ONNX Runtime, CPU-only) ...")
        try:
            from rapidocr_onnxruntime import RapidOCR
            # RapidOCR's packaged config defaults to -1 (all available CPU
            # threads). Pin both pools to one thread for predictable latency
            # and a smaller memory footprint on single-core, 512 MB hosts.
            # Set the supported 1.4.4 geometry values explicitly, then verify
            # the live instance before every inference. This keeps the source
            # resize and detector estimates tied to the actual engine config.
            _ocr_engine = RapidOCR(
                intra_op_num_threads=1,
                inter_op_num_threads=1,
                min_height=RAPIDOCR_MIN_HEIGHT,
                width_height_ratio=RAPIDOCR_WIDTH_HEIGHT_RATIO,
                max_side_len=RAPIDOCR_MAX_SIDE_LEN,
                min_side_len=RAPIDOCR_MIN_SIDE_LEN,
                det_limit_side_len=RAPIDOCR_DET_LIMIT_SIDE_LEN,
                det_limit_type=RAPIDOCR_DET_LIMIT_TYPE,
            )
            logger.info("[OCR] RapidOCR initialized successfully.")
        except Exception as exc:
            logger.exception("[OCR] RapidOCR initialization failed")
            _ocr_engine = None
            raise OCREngineUnavailable("RapidOCR could not initialize.") from exc
    return _ocr_engine


def _box_from_raw(raw_box, width, height):
    """Validate and clamp a quadrilateral to the source image dimensions."""
    if hasattr(raw_box, "tolist"):
        raw_box = raw_box.tolist()
    try:
        points = np.asarray(raw_box, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    # Permit small rounding drift, but reject coordinates that do not describe
    # a box on this image. Coordinates stay in source-image space.
    if (
        np.any(points[:, 0] < -5) or np.any(points[:, 0] > width + 5)
        or np.any(points[:, 1] < -5) or np.any(points[:, 1] > height + 5)
    ):
        return None
    points[:, 0] = np.clip(points[:, 0], 0, width)
    points[:, 1] = np.clip(points[:, 1], 0, height)
    area = abs(float(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
                       np.dot(points[:, 1], np.roll(points[:, 0], 1)))) / 2.0
    if area < 1.0:
        return None
    return [[float(x), float(y)] for x, y in points]


def _box_bounds(item):
    box = item.get("bbox")
    if not box:
        return None
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap_ratio(a, b):
    """Return intersection-over-union and intersection over the smaller box."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a, area_b = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1), max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0, intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0


def _deduplicate_overlapping(items):
    """Drop duplicate text detections that substantially overlap spatially."""
    kept = []
    for item in sorted(items, key=lambda value: value["confidence"], reverse=True):
        bounds = _box_bounds(item)
        normalized = "".join(ch.lower() for ch in item["text"] if ch.isalnum())
        duplicate = False
        if bounds and normalized:
            for prior in kept:
                prior_bounds = _box_bounds(prior)
                prior_text = "".join(ch.lower() for ch in prior["text"] if ch.isalnum())
                if not prior_bounds or not prior_text:
                    continue
                same_text = normalized == prior_text
                if same_text:
                    iou, smaller_overlap = _overlap_ratio(bounds, prior_bounds)
                    if iou >= 0.45 or smaller_overlap >= 0.82:
                        duplicate = True
                        break
        if not duplicate:
            kept.append(item)
    return sorted(kept, key=lambda item: (
        min(p[1] for p in item["bbox"]), min(p[0] for p in item["bbox"])
    ))


def _parse_results(result, width, height):
    if not result:
        return []
    if not isinstance(result, (list, tuple)):
        raise OCREngineUnavailable("RapidOCR returned an unsupported result format.")

    extracted = []
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        if not isinstance(item[1], str):
            logger.warning("[OCR] Ignoring item with non-text content: %r", item[1])
            continue
        text = item[1].strip()
        try:
            confidence = float(item[2])
        except (ValueError, TypeError, OverflowError):
            logger.warning("[OCR] Ignoring item with invalid confidence: %r", item[2])
            continue
        if not text or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            continue
        # Keep the previous public API threshold. NLP applies stricter checks
        # later, while the returned string remains the recognizer's raw text.
        if confidence < 0.30:
            continue
        box = _box_from_raw(item[0], width, height)
        if box is None:
            logger.warning("[OCR] Ignoring detection with invalid/out-of-image bounding box")
            continue
        extracted.append({"text": text, "confidence": round(confidence, 4), "bbox": box})
    return _deduplicate_overlapping(extracted)


def _extract_from_input(image_input, width, height, label):
    # Check a conservative estimate before initializing models or asking
    # RapidOCR to decode an image path. Then verify against the live settings.
    estimate_ocr_detector_geometry(width, height)
    engine = get_reader()
    if engine is None:
        raise OCREngineUnavailable("RapidOCR could not initialize.")
    estimate_ocr_detector_geometry(width, height, engine=engine)
    logger.info("[OCR] Running RapidOCR on: %s", label)
    try:
        # RapidOCR's call path mutates its text-score parameter when passed
        # keyword options, so inference stays serialized on the shared instance.
        with _inference_lock:
            result, elapse = engine(image_input)
        logger.info("[OCR] RapidOCR inference complete in %.2fs", _elapsed_seconds(elapse))
        return _parse_results(result, width, height)
    except OCREngineUnavailable:
        raise
    except Exception as exc:
        logger.exception("[OCR] RapidOCR inference failed on %s", label)
        raise OCREngineUnavailable("RapidOCR inference failed.") from exc
    finally:
        gc.collect()


def extract_text(image_path: str) -> List[Dict[str, Any]]:
    """Return raw OCR regions with validated boxes and original confidences."""
    if not os.path.exists(image_path):
        logger.error("[OCR] Input image not found: %s", image_path)
        raise FileNotFoundError(f"OCR input image not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        raise OCREngineUnavailable("OCR could not read image dimensions.") from exc
    return _extract_from_input(image_path, width, height, image_path)


def extract_text_from_image(image: np.ndarray, label="<image>") -> List[Dict[str, Any]]:
    """Run the same CPU OCR and validation on a preprocessed OpenCV image array."""
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("OCR input must be a non-empty image array")
    height, width = image.shape[:2]
    return _extract_from_input(image, width, height, label)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    test_img = sys.argv[1] if len(sys.argv) > 1 else "public/samples/sample_text.jpg"
    print(f"Running RapidOCR extraction on: {test_img}")
    res = extract_text(test_img)
    print(f"Found {len(res)} text segments:")
    for item in res:
        print(f"  [{item['confidence']:.3f}] {item['text']}")
