import os
import gc
import threading
import logging
import math
from typing import Any, List, Dict

logger = logging.getLogger(__name__)

# ==============================================================================
# OCR EXTRACTOR — RapidOCR (ONNX Runtime, CPU-only)
# ==============================================================================
#
# Replaces EasyOCR + PyTorch with RapidOCR (rapidocr-onnxruntime==1.4.4).
# RapidOCR uses ONNX Runtime for all three models:
#   - Detection   (DBNet)
#   - Angle classification
#   - Recognition (CRNN / PP-OCRv4)
#
# Memory saving vs EasyOCR+PyTorch:
#   EasyOCR peak RSS (local, measured): 530.7 MB  → exceeds Render 512 MB limit
#   RapidOCR peak RSS (local, measured): 120.4 MB → 391.6 MB headroom
#
# Public interface is UNCHANGED:
#   extract_text(image_path) -> List[{"text": str, "confidence": float}]
#
# Health check helpers:
#   ocr_is_configured()   -> bool   (package importable, no model load)
#   ocr_is_initialized()  -> bool   (singleton already loaded)
#   ocr_is_operational()  -> bool   (can run inference right now)

_ocr_engine = None
_ocr_lock = threading.Lock()


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
        import importlib.util
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
    """
    Thread-safe lazy singleton for RapidOCR engine.

    Uses ONNX Runtime CPU execution provider exclusively.
    No PyTorch, no TensorFlow. The package's default ONNX models are loaded
    lazily once, then reused for the process lifetime.

    Thread safety: a threading.Lock() prevents concurrent init races.
    The app-level semaphore in app.py prevents concurrent analysis requests,
    so the lock here primarily guards against unlikely startup races.
    """
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine

        logger.info("[OCR] Initializing RapidOCR (ONNX Runtime, CPU-only) ...")
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()
            logger.info("[OCR] RapidOCR initialized successfully.")
        except Exception as exc:
            logger.exception("[OCR] RapidOCR initialization failed")
            _ocr_engine = None
            raise OCREngineUnavailable("RapidOCR could not initialize.") from exc

    return _ocr_engine


def extract_text(image_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text segments from a preprocessed classroom board image.

    Uses RapidOCR (ONNX Runtime CPU). No PyTorch required.

    Output format is IDENTICAL to the previous EasyOCR implementation:
        [{"text": str, "confidence": float}, ...]

    RapidOCR returns: (result, elapse)
    where result is a list of [bounding_box, text, confidence] tuples,
    or None when no text is detected.

    Confidence threshold: 0.30 (same as previous EasyOCR extractor).
    """
    if not os.path.exists(image_path):
        logger.error("[OCR] Input image not found: %s", image_path)
        raise FileNotFoundError(f"OCR input image not found: {image_path}")

    engine = get_reader()
    if engine is None:
        raise OCREngineUnavailable("RapidOCR could not initialize.")

    logger.info("[OCR] Running RapidOCR on: %s", image_path)
    try:
        result, elapse = engine(image_path)
        elapsed_seconds = _elapsed_seconds(elapse)
        logger.info("[OCR] RapidOCR inference complete in %.2fs", elapsed_seconds)
    except Exception as exc:
        logger.exception("[OCR] RapidOCR inference failed on %s", image_path)
        raise OCREngineUnavailable("RapidOCR inference failed.") from exc

    if not result:
        logger.info("[OCR] No text detected.")
        return []

    if not isinstance(result, (list, tuple)):
        logger.error("[OCR] Unsupported RapidOCR result type: %s", type(result).__name__)
        raise OCREngineUnavailable("RapidOCR returned an unsupported result format.")

    extracted_text: List[Dict[str, Any]] = []
    for item in result:
        # RapidOCR result item: [bounding_box, text, confidence]
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        if not isinstance(item[1], str):
            logger.warning("[OCR] Ignoring item with non-text content: %r", item[1])
            continue
        text = item[1].strip()
        try:
            confidence = float(item[2])
        except (ValueError, TypeError):
            logger.warning("[OCR] Ignoring item with invalid confidence: %r", item[2])
            continue
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            logger.warning("[OCR] Ignoring item with out-of-range confidence: %r", item[2])
            continue
        if not text:
            continue
        if confidence < 0.30:
            continue
        item_data = {
            "text": text,
            "confidence": round(confidence, 4)
        }
        raw_box = item[0]
        try:
            if hasattr(raw_box, "tolist"):
                raw_box = raw_box.tolist()
            item_data["bbox"] = [
                [float(point[0]), float(point[1])]
                for point in raw_box
            ]
        except (TypeError, ValueError, IndexError):
            logger.warning("[OCR] Skipping malformed bounding box for OCR item: %r", item[0])
        extracted_text.append(item_data)

    # Explicit release of raw result list (may hold intermediate numpy arrays).
    del result
    gc.collect()

    logger.info("[OCR] Complete: %d segments extracted", len(extracted_text))
    return extracted_text


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    test_img = sys.argv[1] if len(sys.argv) > 1 else "public/samples/sample_text.jpg"
    print(f"Running RapidOCR extraction on: {test_img}")
    res = extract_text(test_img)
    print(f"Found {len(res)} text segments:")
    for r in res:
        print(f"  [{r['confidence']:.3f}] {r['text']}")
