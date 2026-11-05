import os
import tempfile
import threading
from typing import Any, List, Dict

# ============================================
# EASY OCR (PRODUCTION SINGLETON INITIALIZATION)
# ============================================

_reader = None
_reader_lock = threading.Lock()


def get_model_storage_directory() -> str:
    """
    Finds or creates a writable directory for EasyOCR models.
    Supports local user cache, project-relative cache, or system temp directory.
    """
    configured = os.environ.get("EASYOCR_MODEL_DIR")
    if configured and os.path.exists(configured):
        return configured

    # Check local workspace cache first (pre-cached models)
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_cache", "easyocr"))
    if os.path.exists(local_dir):
        try:
            files = os.listdir(local_dir)
            if any(f.endswith(".pth") for f in files):
                return local_dir
        except Exception:
            pass

    # Check default user location
    user_dir = os.path.expanduser("~/.EasyOCR/model")
    try:
        os.makedirs(user_dir, exist_ok=True)
        test_file = os.path.join(user_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return user_dir
    except (OSError, PermissionError):
        pass

    try:
        os.makedirs(local_dir, exist_ok=True)
        return local_dir
    except (OSError, PermissionError):
        pass

    # Fallback to system temp directory
    tmp_dir = os.path.join(tempfile.gettempdir(), "easyocr_models")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def get_reader():
    """
    Thread-safe lazy initializer for EasyOCR Reader.
    """
    global _reader
    if _reader is not None:
        return _reader

    with _reader_lock:
        if _reader is not None:
            return _reader

        model_dir = get_model_storage_directory()
        try:
            import easyocr
            print(f"Initializing EasyOCR reader with model dir: {model_dir}")
            _reader = easyocr.Reader(
                ['en'],
                gpu=False,
                model_storage_directory=model_dir,
                user_network_directory=model_dir,
                download_enabled=True,
                verbose=False
            )
            print("[OK] EasyOCR reader successfully initialized.")
        except Exception as exc:
            print(f"Warning: EasyOCR reader initialization attempt failed: {exc}")
            try:
                import easyocr
                # Fallback to default reader params
                _reader = easyocr.Reader(['en'], gpu=False, download_enabled=True, verbose=False)
                print("[OK] EasyOCR fallback initialization succeeded.")
            except Exception as exc2:
                print(f"Error: EasyOCR reader could not be initialized: {exc2}")
                _reader = None

    return _reader


# ============================================
# OCR EXTRACTION
# ============================================

def extract_text(image_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text tokens and bounding regions from a preprocessed classroom board image.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found for OCR: {image_path}")
        return []

    reader = get_reader()
    if reader is None:
        print("Warning: EasyOCR reader unavailable.")
        return []

    try:
        results = reader.readtext(
            image_path,
            detail=1,
            paragraph=False,
            min_size=10,
            text_threshold=0.50,
            low_text=0.25,
            link_threshold=0.30,
            mag_ratio=1.5
        )
    except Exception as exc:
        print(f"Warning: EasyOCR extraction error on {image_path}: {exc}")
        return []

    extracted_text = []

    for result in results:
        if not isinstance(result, (list, tuple)) or len(result) < 3:
            continue

        text = str(result[1]).strip()
        try:
            confidence = float(result[2])
        except (ValueError, TypeError):
            continue

        if not text:
            continue

        # Keep results with reasonable confidence threshold
        if confidence < 0.30:
            continue

        extracted_text.append({
            "text": text,
            "confidence": round(confidence, 4)
        })

    return extracted_text


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "public/samples/sample_text.jpg"
    print(f"Running OCR extraction on: {test_img}")
    res = extract_text(test_img)
    print(f"Found {len(res)} text lines:")
    for r in res:
        print(f"  * {r['text']} (conf: {r['confidence']})")
