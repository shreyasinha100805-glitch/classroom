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

# RapidOCR uses ONNX Runtime on CPU. The model is lazy-loaded once per process.
_ocr_engine = None
_ocr_lock = threading.Lock()
_inference_lock = threading.Lock()


class OCREngineUnavailable(RuntimeError):
    """Raised when RapidOCR cannot initialize or complete inference."""


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
            _ocr_engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
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
    engine = get_reader()
    if engine is None:
        raise OCREngineUnavailable("RapidOCR could not initialize.")
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
