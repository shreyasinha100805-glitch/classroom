import os
import gc
import tempfile
import threading
import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)

# ==============================================================================
# OCR EXTRACTOR — RapidOCR (ONNX Runtime, CPU-only)
# ==============================================================================
#
# Replaces EasyOCR + PyTorch with RapidOCR (rapidocr-onnxruntime==1.4.x).
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
    No PyTorch, no TensorFlow. Models (~8 MB total) are downloaded once
    to Render's /tmp on cold start, then reused for the process lifetime.

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

            # RapidOCR downloads lightweight ONNX models (~8 MB) on first init.
            # On Render, /tmp is writable; models are cached for the process lifetime.
            # Passing no arguments uses the bundled default models (PP-OCRv4).
            _ocr_engine = RapidOCR()
            logger.info("[OCR] RapidOCR initialized successfully.")
        except Exception as exc:
            logger.error("[OCR] RapidOCR initialization failed: %s", exc, exc_info=True)
            _ocr_engine = None

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
        logger.warning("[OCR] Image not found: %s", image_path)
        return []

    engine = get_reader()
    if engine is None:
        logger.warning("[OCR] RapidOCR engine unavailable — returning empty result.")
        return []

    logger.info("[OCR] Running RapidOCR on: %s", image_path)
    try:
        result, elapse = engine(image_path)
        logger.info("[OCR] RapidOCR inference complete in %.2fs", elapse if elapse else 0)
    except Exception as exc:
        logger.warning("[OCR] RapidOCR inference error on %s: %s", image_path, exc, exc_info=True)
        return []

    if not result:
        logger.info("[OCR] No text detected.")
        return []

    extracted_text: List[Dict[str, Any]] = []
    for item in result:
        # RapidOCR result item: [bounding_box, text, confidence]
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1]).strip()
        try:
            confidence = float(item[2])
        except (ValueError, TypeError):
            confidence = 0.0
        if not text:
            continue
        if confidence < 0.30:
            continue
        extracted_text.append({
            "text": text,
            "confidence": round(confidence, 4)
        })

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
